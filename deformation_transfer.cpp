#include <iostream>
#include <filesystem>
#include <string>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/CholmodSupport>
#include "TriMesh.h"
#include "TriMesh_algo.h"
#include "ICP.h"
#include "mytimer.h"

using namespace std;

Eigen::Matrix3d CalculateV(trimesh::TriMesh *mesh, int i)
{
    int idx_v1 = mesh->faces[i][0];
    int idx_v2 = mesh->faces[i][1];
    int idx_v3 = mesh->faces[i][2];
    trimesh::point p_v1 = mesh->vertices[idx_v1];
    trimesh::point p_v2 = mesh->vertices[idx_v2];
    trimesh::point p_v3 = mesh->vertices[idx_v3];
    Eigen::Vector3d v1(p_v1[0], p_v1[1], p_v1[2]);
    Eigen::Vector3d v2(p_v2[0], p_v2[1], p_v2[2]);
    Eigen::Vector3d v3(p_v3[0], p_v3[1], p_v3[2]);
    // Eigen::Vector3d v4 = v1 + (v2-v1).cross(v3-v1)/((v2-v1).cross(v3-v1)).norm();
    Eigen::Vector3d v4 = v1 + ( (v2 - v1).cross(v3 - v1) ).normalized();
    Eigen::Matrix3d V;
    V << v2 - v1, v3 - v1, v4 - v1;
    return V;
}

/* 
    夫夫哥暴力法:直接求Q+d并应用至对应三角形
    不同三角形共享的相同顶点的多个位置求平均值来作为最终顶点位置
*/
void def_transfer(
    trimesh::TriMesh *dfm,
    trimesh::TriMesh *src,
    trimesh::TriMesh *tgt)
{

    Eigen::MatrixX3d Q(3 * src->faces.size(), 3);
    Eigen::Matrix3Xd d(3, 3 * src->vertices.size());

    for (int i = 0; i < src->faces.size(); i++)
    {
        Eigen::Matrix3d Qi, V, V_hat;
        V_hat = CalculateV(dfm, i);
        V = CalculateV(src, i);
        Qi = V_hat * V.inverse();
        Q.block<3, 3>(3 * i, 0) = Qi;

        int idx_v1 = src->faces[i][0];
        trimesh::point p_v1 = src->vertices[idx_v1];
        Eigen::Vector3d v(p_v1[0], p_v1[1], p_v1[2]);

        int idx_v_hat1 = dfm->faces[i][0];
        trimesh::point p_v_hat1 = dfm->vertices[idx_v_hat1];
        Eigen::Vector3d v_hat(p_v_hat1[0], p_v_hat1[1], p_v_hat1[2]);

        d.col(i) = v_hat - Qi * v;
    }

    // count zero cols in d
    // int zerocnt = 0;
    // for(int i = 0; i < src->faces.size(); i++){
    //     if(d.col(i) == Eigen::Vector3d(0,0,0))
    //         zerocnt++;
    // }
    // cout<<"d has "<<zerocnt<<" zero col"<<endl;

    // check if the calculated Q&d will make triangles fail
    std::vector<::std::vector<int>> adjacentfaces = src->adjacentfaces;
    // for(int i = 0; i < adjacentfaces.size(); i++){
    //     Eigen::Vector3d result(0,0,0);
    //     trimesh::point p = src->vertices[i];
    //     Eigen::Vector3d v(p[0], p[1], p[2]);
    //     for(auto& face_idx:adjacentfaces[i]){
    //         Eigen::Vector3d tmp;
    //         tmp = Q.block<3,3>(face_idx*3,0) * v + d.col(face_idx);
    //         if(i == 0)
    //             result = tmp;
    //         else if(result != tmp)
    //             cout<<"FUCK!!!!!!"<<endl;
    //             exit(1);
    //     }
    // }

    // trimesh::TriMesh* tgt_dfm = new trimesh::TriMesh;

    for (int i = 0; i < adjacentfaces.size(); i++)
    {
        vector<int> faces = adjacentfaces[i];
        Eigen::Vector3d result(0, 0, 0);
        trimesh::point p = tgt->vertices[i];
        Eigen::Vector3d v(p[0], p[1], p[2]);
        for (auto &face_idx : faces)
        {
            Eigen::Vector3d deformed;
            deformed = Q.block<3, 3>(face_idx * 3, 0) * v + d.col(face_idx);
            result += deformed;
        }
        result /= faces.size();
        // tgt_dfm->vertices.push_back(trimesh::point(result[0],result[1],result[2]));
        tgt->vertices[i] = trimesh::point(result[0], result[1], result[2]);
    }

    // tgt_dfm->need_neighbors();
    // tgt_dfm->need_normals();
    // tgt_dfm->need_faces();
}
Eigen::Matrix<double,3,2> CalculateW(trimesh::TriMesh *mesh, int i)
{
    int idx_v1 = mesh->faces[i][0];
    int idx_v2 = mesh->faces[i][1];
    int idx_v3 = mesh->faces[i][2];
    trimesh::point p_v1 = mesh->vertices[idx_v1];
    trimesh::point p_v2 = mesh->vertices[idx_v2];
    trimesh::point p_v3 = mesh->vertices[idx_v3];
    Eigen::Vector3d v1(p_v1[0], p_v1[1], p_v1[2]);
    Eigen::Vector3d v2(p_v2[0], p_v2[1], p_v2[2]);
    Eigen::Vector3d v3(p_v3[0], p_v3[1], p_v3[2]);
    Eigen::Matrix<double, 3, 2> W;
    W << v2 - v1, v3 - v1;
    return W;
}
void AXEqulsToF(
    trimesh::TriMesh *dfm,
    trimesh::TriMesh *src,
    trimesh::TriMesh *tgt)
{
    // Solve AX=F

    /*
        A(3mxn): sparse matrix

        a   d   (-a-d)      v2.T
        b   e   (-b-e)  *   v1.T      =       S.T
        c   f   (-c-f)      v3.T

    */
    Eigen::SparseMatrix<double> A(3 * tgt->faces.size(), tgt->vertices.size());
    std::vector<Eigen::Triplet<double>> coefficients;
    coefficients.reserve(3 * tgt->faces.size() * 3);
    for (int i = 0; i < tgt->faces.size(); i++)
    {
        Eigen::Matrix<double, 3, 2> W = CalculateW(tgt, i);
        Eigen::HouseholderQR<Eigen::MatrixXd> qr;
        qr.compute(W);
        Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
        Eigen::MatrixXd Q = qr.householderQ();
        Eigen::Matrix2d R_alpha = R.block<2,2>(0,0);
        Eigen::Matrix<double,3,2>Q_alpha = Q.block<3,2>(0,0); 
        // std::cout<<R<<endl;
        // std::cout<<Q<<endl;
        // std::cout<<R_alpha<<endl;
        // std::cout<<Q_alpha<<endl;
        // std::cout<<R_alpha.inverse()<<endl;
        // std::cout<<Q_alpha.transpose()<<endl;
        Eigen::Matrix<double, 2, 3> W_inv = R_alpha.inverse() * Q_alpha.transpose();

        double a = W_inv(0, 0);
        double b = W_inv(0, 1);
        double c = W_inv(0, 2);
        double d = W_inv(1, 0);
        double e = W_inv(1, 1);
        double f = W_inv(1, 2);

        trimesh::TriMesh::Face face = tgt->faces[i];

        coefficients.push_back(Eigen::Triplet<double>(3 * i, face[1], a));
        coefficients.push_back(Eigen::Triplet<double>(3 * i, face[2], d));
        coefficients.push_back(Eigen::Triplet<double>(3 * i, face[0], -a - d));

        coefficients.push_back(Eigen::Triplet<double>(3 * i + 1, face[1], b));
        coefficients.push_back(Eigen::Triplet<double>(3 * i + 1, face[2], e));
        coefficients.push_back(Eigen::Triplet<double>(3 * i + 1, face[0], -b - e));

        coefficients.push_back(Eigen::Triplet<double>(3 * i + 2, face[1], c));
        coefficients.push_back(Eigen::Triplet<double>(3 * i + 2, face[2], f));
        coefficients.push_back(Eigen::Triplet<double>(3 * i + 2, face[0], -c - f));
    }
    A.setFromTriplets(coefficients.begin(), coefficients.end());

    /*
                |v_hat1.T|
        X(nx3)= |   :    |
                |v_hatn.T|

    */
    Eigen::MatrixX3d X = Eigen::MatrixX3d::Zero(tgt->vertices.size(), 3);

    /*
                |S1.T|
        F(3mx3)=|  : |
                |Sn.T|
    */
    Eigen::MatrixX3d F(3 * src->faces.size(), 3);

    for (int i = 0; i < src->faces.size(); i++){
        Eigen::Matrix3d Si, V, V_hat;
        V_hat = CalculateV(dfm, i);
        V = CalculateV(src, i);
        Si = V_hat * V.inverse();
        F.block<3, 3>(3 * i, 0) = Si.transpose();
    }
    
    Eigen::SparseMatrix<double> AtA = Eigen::SparseMatrix<double>(A.transpose()) * A;
    Eigen::MatrixX3d AtF = Eigen::SparseMatrix<double>(A.transpose()) * F;
    // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    // Eigen::ConjugateGradient< Eigen::SparseMatrix<double> > solver;
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double> > solver;
    // Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(AtA);
    X = solver.solve(AtF);

    for(int i = 0; i < tgt->vertices.size(); i++){
        tgt->vertices[i] = trimesh::point(X(i,0),X(i,1),X(i,2));
    }

}

trimesh::xform RigidICP(trimesh::TriMesh* m_src, trimesh::TriMesh* m_tgt){
		trimesh::ICP_xform_type xform_type = trimesh::ICP_RIGID;
		int verbose = 1;
		trimesh::KDtree *kd_src = new trimesh::KDtree(m_src->vertices);
		trimesh::KDtree *kd_tgt = new trimesh::KDtree(m_tgt->vertices);
		trimesh::xform xf_src;
		trimesh::xform xf_tgt;
		float err = trimesh::ICP(m_src, m_tgt, xf_src, xf_tgt, kd_src, kd_tgt, 0.0f, verbose, xform_type);

		// only xform2 will be changed & can be applied
		trimesh::apply_xform(m_tgt, xf_tgt);
		if (err < 0.0f) {
			trimesh::TriMesh::eprintf("ICP failed\n");
			exit(1);
		}
		trimesh::TriMesh::eprintf("ICP succeeded - distance = %f\n", err);
        return xf_tgt;
}
int main()
{

    MyTimer timer;
    timer.start();

    string rootPath = "/home/pcd/vscodes/pcd_nricp";
    string srcFile = rootPath + "/data/source.obj";
    string tgtFile = rootPath + "/result/result.obj";
    string dfmFileFolder = rootPath + "/data/blendshapes";
    string outfile = rootPath + "/result/blendshapes";
    string testfile = rootPath + "/result/blendshapes_test";
    vector<string> dfmFiles;
    for (auto &blendshapes : std::filesystem::directory_iterator(dfmFileFolder))
        dfmFiles.push_back(blendshapes.path());

    trimesh::TriMesh *src, *tgt, *tgt_tmp;
    src = trimesh::TriMesh::read(srcFile);
    tgt = trimesh::TriMesh::read(tgtFile);
    tgt_tmp = new trimesh::TriMesh;
    src->need_adjacentfaces();

    // for (auto &dfmFile : dfmFiles)
    // {
    //     trimesh::TriMesh *dfm = trimesh::TriMesh::read(dfmFile);
    //     tgt_tmp->vertices = tgt->vertices;
    //     tgt_tmp->faces = tgt->faces;
    //     dfm->need_adjacentfaces();
        
    //     def_transfer(dfm, src, tgt_tmp);

    //     tgt_tmp->need_neighbors();
    //     tgt_tmp->need_normals();
    //     int pos = dfmFile.rfind("/");
    //     tgt_tmp->write(outfile + dfmFile.substr(pos));
    // }

    bool IsRigidAligned = false;
    trimesh::xform xform_to_tgt;
    for (auto &dfmFile : dfmFiles)
    {
        trimesh::TriMesh *dfm = trimesh::TriMesh::read(dfmFile);
        tgt_tmp->vertices = tgt->vertices;
        tgt_tmp->faces = tgt->faces;
        
        AXEqulsToF(dfm, src, tgt_tmp);

        tgt_tmp->need_neighbors();
        tgt_tmp->need_normals();

        int pos = dfmFile.rfind("/");

        if(!IsRigidAligned){
            xform_to_tgt = RigidICP(tgt, tgt_tmp);
            IsRigidAligned = true;
        }
        else{
            trimesh::apply_xform(tgt_tmp, xform_to_tgt);
        }

        tgt_tmp->write(outfile + dfmFile.substr(pos));
    }

    timer.end();
    return 0;
}