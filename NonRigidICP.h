#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/CholmodSupport>

#include <flann/flann.hpp>

#include "TriMesh.h"
#include "TriMesh_algo.h"
#include "ICP.h"

#include "mytimer.h"

#define VERBOSE 1
#define NUM_LANDMARKS 49
#define THRESHOLD_DIST 20     //less
#define THRESHOLD_NORM 0.5     //greater

using namespace std;
using namespace trimesh;

MyTimer mytimer;

class NonRigidICP
{
private:
	trimesh::TriMesh* 					m_src;
	trimesh::TriMesh*              	 	m_tgt;
	
	
	int 								src_vertsNum;		
	int 								tgt_vertsNum;					
	std::vector<trimesh::TriMesh::Face> m_faces;
	std::vector<std::pair<int,int> >    src_edges;
	
	
	//cv::Mat 							m_image;
	//Eigen::Matrix<double,3,4> 		m_projMatrix;
	
	//for landmark term
    //source:
	std::vector<int>					src_lm_idx;  
	std::vector<Eigen::Vector3d> 		src_lm_verts;  
	
    //target:
	std::vector<int>					tgt_lm_idx;
	std::vector<Eigen::Vector3d> 		tgt_lm_verts;  
	
	std::vector<float>					m_weights;
	std::vector<std::pair<int,int> >    m_soft_corres;
	
	// kdtre
	double* 							m_dataset;
	flann::Index<flann::L2<double> >* 	m_kd_flann_index;

	//translations
	trimesh::point 						centroid_src;

	// eye_cornor indexs 
	map<int,bool>						eye_socket;
	map<int,bool>						mesh_boundry;
public:
    NonRigidICP(trimesh::TriMesh* _src, trimesh::TriMesh* _tgt,
				std::vector<int> _src_lm_idx, std::vector<Eigen::Vector3d>  _src_lm_verts,
				std::vector<int> _tgt_lm_idx, std::vector<Eigen::Vector3d>  _tgt_lm_verts)
	{
        m_src = _src;
		m_tgt = _tgt;
		
        //used for?
		m_src->clear_colors();

        src_lm_idx = _src_lm_idx;
        src_lm_verts = _src_lm_verts;

        tgt_lm_idx = _tgt_lm_idx;
        tgt_lm_verts = _tgt_lm_verts;

        src_vertsNum = m_src->vertices.size();
		tgt_vertsNum = m_tgt->vertices.size();
        m_faces = m_src->faces;

        for(int i = 0; i < m_faces.size(); i++){
            trimesh::TriMesh::Face face=m_faces[i];
			// makes every edge[v1,v2] has v1 < v2
			// easier to remove repeat verts
			sort(face.begin(),face.end());
			int a=face[0];
			int b=face[1];
			int c=face[2];
			src_edges.push_back(std::pair<int,int>(a,b));
			src_edges.push_back(std::pair<int,int>(a,c));
			src_edges.push_back(std::pair<int,int>(b,c));
        }
		
		// remove repeated edges
		set< std::pair<int,int> > s(src_edges.begin(),src_edges.end());
		cout << "Repeat Edges Num is:" << src_edges.size() - s.size() << endl;
		src_edges.assign(s.begin(),s.end());
		cout << "Edges Num Now is " << src_edges.size() << endl;

		std::string eye_socket_file = "/home/pcd/vscodes/pcd_nricp/data/eye_socket.txt"; 
		ifstream fin(eye_socket_file,ios::in);
		if (!fin.is_open()) 
        	cout << "File cannot be imported!" << endl;
		std::string line;
		while(getline(fin,line)){
			int index = atoi(line.c_str());
			// cout<< index <<endl;
			eye_socket[index] = true;
		}

		std::string boundry_file = "/home/pcd/vscodes/pcd_nricp/data/boundry.txt"; 
		ifstream fin_boundry(boundry_file,ios::in);
		if (!fin_boundry.is_open()) 
        	cout << "File cannot be imported!" << endl;
		while(getline(fin_boundry,line)){
			int index = atoi(line.c_str());
			// cout<< index <<endl;
			mesh_boundry[index] = true;
		}
    }

    ~NonRigidICP()
	{
		releaseKdTree();
	}
    void buildKdTree(trimesh::TriMesh* mesh)
	{
		std::vector<trimesh::point> verts = mesh->vertices;
		
		int vertNum=verts.size();
		m_dataset= new double[3*vertNum];
		
		for(int i=0; i<vertNum; i++)
		{
			m_dataset[3*i]=verts[i][0];
			m_dataset[3*i+1]=verts[i][1];
			m_dataset[3*i+2]=verts[i][2];
		}
		
		flann::Matrix<double> flann_dataset(m_dataset,vertNum,3);
		m_kd_flann_index=new flann::Index<flann::L2<double> >(flann_dataset,flann::KDTreeIndexParams(1));
		m_kd_flann_index->buildIndex();
		return;
	}

    void releaseKdTree()
	{
		delete[] m_dataset;
		delete m_kd_flann_index;
		m_kd_flann_index=NULL;
	}

	// move both src & tgt to original point
	// record src original centroid 
	// !!! if only move tgt to src, result mesh is bad
	void MoveToOriginUseAllPoints()
	{
		trimesh::point p_src,p_tgt;
		p_src = mesh_center_of_mass(m_src); 
		for(int i = 0; i < src_vertsNum; i++)
			m_src->vertices[i] -= p_src;
		centroid_src = p_src;

		p_tgt = mesh_center_of_mass(m_tgt);
		for(int i = 0; i < tgt_vertsNum; i++)
			m_tgt->vertices[i] -= p_tgt;

		// assign new verts value to landmarks verts
		for(int i = 0; i < NUM_LANDMARKS; i++){
			int idx = src_lm_idx[i];
			trimesh::point p = m_src->vertices[idx];
			src_lm_verts[i] = Eigen::Vector3d(p[0],p[1],p[2]);
		}
		for(int i = 0; i < NUM_LANDMARKS; i++){
			int idx = tgt_lm_idx[i];
			trimesh::point p = m_tgt->vertices[idx];
			tgt_lm_verts[i] = Eigen::Vector3d(p[0],p[1],p[2]);
		}
	}
	void MoveToOriginUseLandmarks(){
		Eigen::Vector3d mean_src_lm(0,0,0);
		Eigen::Vector3d mean_tgt_lm(0,0,0);
		for(size_t i=0;i<NUM_LANDMARKS;i++)
		{
			mean_src_lm += src_lm_verts[i];
			mean_tgt_lm += tgt_lm_verts[i];
		}
		mean_src_lm /= NUM_LANDMARKS;
		mean_tgt_lm /= NUM_LANDMARKS;
		trimesh::point p_mean_src,p_mean_tgt;
		p_mean_src = (mean_src_lm[0],mean_src_lm[1],mean_src_lm[2]);
		p_mean_tgt = (mean_tgt_lm[0],mean_tgt_lm[1],mean_tgt_lm[2]);
		for(int i = 0; i < src_vertsNum; i++){
			m_src->vertices[i] -= p_mean_src;
		}
		for(int i = 0; i < tgt_vertsNum; i++){
			m_tgt->vertices[i] -= p_mean_tgt;
		}

		// assign new verts value to landmarks verts
		for(int i = 0; i < NUM_LANDMARKS; i++){
			int idx = src_lm_idx[i];
			trimesh::point p = m_src->vertices[idx];
			src_lm_verts[i] = Eigen::Vector3d(p[0],p[1],p[2]);
		}
		for(int i = 0; i < NUM_LANDMARKS; i++){
			int idx = tgt_lm_idx[i];
			trimesh::point p = m_tgt->vertices[idx];
			tgt_lm_verts[i] = Eigen::Vector3d(p[0],p[1],p[2]);
		}

	}
	// resize tgt to src size
	// based on their landmarks
	void resizeUseAllPoints()
	{
		trimesh::point p_src,p_tgt;
		p_src = mesh_center_of_mass(m_src); 
		p_tgt = mesh_center_of_mass(m_tgt);
		Eigen::Vector3d AA,BB;
		double meanRadius_AA=0, meanRadius_BB=0;
		for (int i = 0; i < src_vertsNum; i++){
			trimesh::point p= m_src->vertices[i] - p_src;
			meanRadius_AA+=sqrt(pow(p[0],2) + pow(p[1],2) +pow(p[1],2));
		}
		for (int i = 0; i < tgt_vertsNum; i++){
			trimesh::point p= m_tgt->vertices[i] - p_tgt;
			meanRadius_BB+=sqrt(pow(p[0],2) + pow(p[1],2) +pow(p[1],2));
		}
		meanRadius_AA/=src_vertsNum;
		meanRadius_BB/=tgt_vertsNum;
		float ratio = meanRadius_AA/meanRadius_BB;
		trimesh::scale(m_tgt,ratio);
	}
    void resizeUseLandmarks()
	{
		Eigen::Vector3d mean_src_lm(0,0,0);
		Eigen::Vector3d mean_tgt_lm(0,0,0);
		for(size_t i=0;i<NUM_LANDMARKS;i++)
		{
			mean_src_lm += src_lm_verts[i];
			mean_tgt_lm += tgt_lm_verts[i];
		}
		
		mean_src_lm /= NUM_LANDMARKS;
		mean_tgt_lm /= NUM_LANDMARKS;
		
		Eigen::Vector3d AA,BB;
		double meanRadius_AA, meanRadius_BB;
		for(size_t i = 0; i < src_lm_verts.size(); i++)
		{
			AA = src_lm_verts[i] - mean_src_lm;
			BB = tgt_lm_verts[i] - mean_tgt_lm;

			meanRadius_AA += sqrt(pow(AA(0),2)+pow(AA(1),2)+pow(AA(2),2));
			meanRadius_BB += sqrt(pow(BB(0),2)+pow(BB(1),2)+pow(BB(2),2));
		}

		float ratio = meanRadius_AA / meanRadius_BB; 
		cout<<"Scale ratio: "<<ratio<<endl;
		
		trimesh::scale(m_tgt, ratio);
		
		//use the scaled landmark vertices
		for(size_t i=0; i<tgt_lm_idx.size(); i++)
		{
			int idx = tgt_lm_idx[i];
			trimesh::point xyz= m_tgt->vertices[idx];
			tgt_lm_verts[i]=Eigen::Vector3d(xyz[0],xyz[1],xyz[2]);
		}
	}
	
	// transform tgt to align with src
	void RigidICP(){
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

		// reset target landmarks vertices
		for(int i = 0; i < NUM_LANDMARKS; i++){
			int idx = tgt_lm_idx[i];
			trimesh::point xyz= m_tgt->vertices[idx];
			tgt_lm_verts[i]=Eigen::Vector3d(xyz[0],xyz[1],xyz[2]);
		}
	}
	// ICP use landmarks' vertex-wise relationship
	// align tgt to src
	void InitialICP()
	{
		Eigen::Vector3d mean_src_lms(0,0,0);
		Eigen::Vector3d mean_tgt_lms(0,0,0);
		for( int i = 0; i < NUM_LANDMARKS; i++){
			mean_src_lms += src_lm_verts[i];
			mean_tgt_lms += tgt_lm_verts[i];
		}
		mean_src_lms /= NUM_LANDMARKS;
		mean_tgt_lms /= NUM_LANDMARKS;

		Eigen::Matrix<double,NUM_LANDMARKS,3>AA,BB;
		for(int i = 0; i < NUM_LANDMARKS; i++){
			AA(i,0) = tgt_lm_verts[i][0] - mean_tgt_lms[0];
			AA(i,1) = tgt_lm_verts[i][1] - mean_tgt_lms[1];
			AA(i,2) = tgt_lm_verts[i][2] - mean_tgt_lms[2];

			BB(i,0) = src_lm_verts[i][0] - mean_src_lms[0];
			BB(i,1) = src_lm_verts[i][1] - mean_src_lms[1];
			BB(i,2) = src_lm_verts[i][2] - mean_src_lms[2];
		}
		Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
		for(int i = 0; i < NUM_LANDMARKS; i++){
			W += Eigen::Vector3d(BB(i,0),BB(i,1),BB(i,2)) * Eigen::Vector3d(AA(i,0),AA(i,1),AA(i,2)).transpose();
		}
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU| Eigen::ComputeFullV);
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d V = svd.matrixV();
		
		Eigen::Matrix3d ini_R = U * V.transpose();
		Eigen::Vector3d ini_t = -ini_R * mean_tgt_lms + mean_src_lms;
		
		if(ini_R.determinant()<0)
		{
			for(int j=0; j<3;j++) U(2,j) *= -1;
			ini_R = U*V.transpose();
		}
		for(int i=0;i<tgt_vertsNum;i++)
		{
			trimesh::point xyz= m_tgt->vertices[i];
			Eigen::Vector3d point(xyz[0], xyz[1], xyz[2]);
			Eigen::Vector3d icp_result = ini_R * point + ini_t;
			
			for(int d=0; d<3; d++)
				m_tgt->vertices[i][d] = icp_result(d);
		}
		cout<<"Initial Rigid ICP finished"<<endl;
	}

	void Init(){
		MoveToOriginUseAllPoints();
		// resizeUseAllPoints();

		// MoveToOriginUseLandmarks();
		resizeUseLandmarks();

		// detailed Rigid ICP: rotate & translation
		// RigidICP();
		InitialICP();
		buildKdTree(m_tgt);
	}

	void ResetSrc(){
		for(int i = 0; i < src_vertsNum; i++){
			m_src->vertices[i] += centroid_src;
		}
	}
	void getKdCorrespondences()
	{
		m_weights.resize(src_vertsNum);
		for(size_t i = 0; i < m_weights.size(); i++)
			m_weights[i] = 1.0f;
		const int knn = 1;

		flann::Matrix<double> query(new double[3],1,3);
		flann::Matrix<int>    indices(new int[query.rows*knn],query.rows,knn);
		flann::Matrix<double> dists(new double[query.rows*knn],query.rows,knn);

		m_soft_corres.clear();
		m_soft_corres.resize(src_vertsNum);

		m_src->need_normals();
		m_tgt->need_normals();

		for(size_t i = 0; i < src_vertsNum; i++){
			for(int j = 0; j < 3; j++)
				query[0][j] = m_src->vertices[i][j];

			m_kd_flann_index->knnSearch(query,indices,dists,1,flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
			
			m_soft_corres[i]=std::make_pair(i,indices[0][0]);

			Eigen::Vector3d n1(m_src->normals[i][0],m_src->normals[i][1],m_src->normals[i][2]);
			Eigen::Vector3d n2(m_tgt->normals[indices[0][0]][0],m_tgt->normals[indices[0][0]][1],m_tgt->normals[indices[0][0]][2]);

			/* below are 3 ways to set weights */

			// threshold set by hand
			// if( dists[0][0] > THRESHOLD_DIST  || abs(n1.dot(n2))<=THRESHOLD_NORM )  
			// threshold set by eye corner
			// if(dists[0][0] > (src_lm_verts[1] - src_lm_verts[3]).norm()/2)
			if(dists[0][0] > 0.0001 )
				m_weights[i]=0.0f;
			
			// if included angle between normals > 90
			// if(normal1.dot(normal2)<0) 
			// 	m_src->normals[i]*=-1;
			// if(n1.dot(n2) < 0)
			// 	m_weights[i] = 0.0f;

			// if v is on mesh boundry
			if( m_tgt->is_bdy(indices[0][0]) || m_src->is_bdy(i))  
				m_weights[i]=0.0f;
			if(i < 40)
				m_weights[i] = 0.0f;
			// eye_cornor set to stable in order to set the eyeball position easily
			if( eye_socket[i] == true || mesh_boundry[i] == true)
				m_weights[i]=0.0f;
		}

		delete[] query.ptr();
		delete[] indices.ptr();
		delete[] dists.ptr();
		
		double sum=0;
		for (size_t i = 0; i < m_weights.size(); ++i) 
			sum+=m_weights[i];
			
		cout<<"Find matches:"<<sum<<endl;
		return;
			
	}
	Eigen::MatrixX3d compute(double alpha, double beta, double gamma)
	{

		int n = src_vertsNum;		//vertices
		int m = src_edges.size();		//edges
		int l = NUM_LANDMARKS;		//landmarks

		Eigen::MatrixX3d X(4*n,3);
		X.setZero();

		bool loop = true;
		int iter = 0;
		
		while(loop){
			mytimer.start();
			getKdCorrespondences();
			mytimer.end();

			mytimer.start();
			cout<<"Construct AtA and Atb"<<endl;

			Eigen::SparseMatrix<double> A(4*m + n + l, 4*n);

			// 1.alpha_M_G
			std::vector< Eigen::Triplet<double> > alpha_M_G;
			for (int i = 0; i < m; i++)
			{
				int a = src_edges[i].first;
				int b = src_edges[i].second;

				for (int j = 0; j < 3; j++) 
				{
					alpha_M_G.push_back(Eigen::Triplet<double>(i*4 + j, a*4 + j, alpha));
					alpha_M_G.push_back(Eigen::Triplet<double>(i*4 + j, b*4 + j, -alpha));
				}

				alpha_M_G.push_back(Eigen::Triplet<double>(i*4 + 3, a*4 + 3, alpha * gamma));	
				alpha_M_G.push_back(Eigen::Triplet<double>(i*4 + 3, b*4 + 3, -alpha * gamma));
			}

			// 2. W_D
			std::vector< Eigen::Triplet<double> > W_D;
			for(int i = 0;i < n; i++)
			{
				trimesh::point xyz = m_src->vertices[i];
				double weight = m_weights[i];

				// if weight = 0, corresponding U will be set the same as D, so weight can be 1
				// see line #462
				if(weight == 0) weight = 1;

				for(int j = 0; j < 3; j++)
					W_D.push_back(Eigen::Triplet<double>(4*m + i, i*4 + j, weight * xyz[j]));
				
				W_D.push_back(Eigen::Triplet<double>(4*m + i, i*4 + 3, weight * 1));
			}

			// 3. beta_D_L
			std::vector<Eigen::Triplet<double> > beta_D_L;
			for(int i = 0 ; i < l; i++)
			{
				for(int j = 0; j < 3; j++)
					beta_D_L.push_back(Eigen::Triplet<double>(
						4*m + n + i, src_lm_idx[i]*4 + j, beta * src_lm_verts[i][j]));
				beta_D_L.push_back(Eigen::Triplet<double>(
					4*m + n + i, src_lm_idx[i]*4 + 3, beta));
			}

			std::vector< Eigen::Triplet<double> > _A = alpha_M_G;
			_A.insert(_A.end(), W_D.begin(), W_D.end());
			_A.insert(_A.end(), beta_D_L.begin(), beta_D_L.end());
			A.setFromTriplets(_A.begin(), _A.end());

			// B
			Eigen::MatrixX3d B = Eigen::MatrixX3d::Zero(4*m + n + l, 3);

			// B: W_U
			for (int i = 0 ; i < n; i++)
			{
				int idx = 0;
				trimesh::point xyz;

				double weight = m_weights[i];
				// if weight is 1, set U the knn result verts
				if(weight == 1) 				
				{
					idx = m_soft_corres[i].second;
					xyz = m_tgt->vertices[idx];
				}
				// if weight is 0, set U the src verts same as D
				else 
				{
					weight = 1;
					idx = m_soft_corres[i].first;
					xyz = m_src->vertices[idx];
				}
				
				for (int j = 0; j < 3; j++)  
					B(4*m + i, j) = weight * xyz[j];
			}
			// B: beta_U_L
			for(int i = 0; i < l; i++)
			{
				for( int j = 0; j < 3 ;j++)
					B(4*m + n + i, j) = beta * tgt_lm_verts[i][j];

			}

			// Cal AtA & AtB
			Eigen::SparseMatrix<double> AtA = Eigen::SparseMatrix<double>(A.transpose()) * A;
			Eigen::MatrixX3d AtB = Eigen::SparseMatrix<double>(A.transpose()) * B;

			// Eigen::ConjugateGradient< Eigen::SparseMatrix<double> > solver;
			// Eigen::SimplicialCholesky<Eigen::SparseMatrix<double> > solver;
			Eigen::CholmodSupernodalLLT< Eigen::SparseMatrix<double> > solver;
			solver.compute(AtA);

			mytimer.end();

			Eigen::MatrixX3d TmpX(4*n,3);
			TmpX=X;

			cout<<"Solve Ax=b  "<<endl;
			mytimer.start();
			X = solver.solve(AtB);
			mytimer.end();

			Eigen::Matrix3Xd Xt = X.transpose();
			for (int i = 0; i < n; ++i)
			{
				trimesh::point xyz=m_src->vertices[i];
				
				Eigen::Vector4d point(xyz[0], xyz[1], xyz[2], 1.0f);
				Eigen::Vector3d result = Xt.block<3, 4>(0, 4*i) * point;
			
				for(int d=0; d<3; d++)
				{
					m_src->vertices[i][d]=result[d];
				}
			}

			for(size_t i=0; i<src_lm_idx.size(); i++)
			{
				int idx = src_lm_idx[i];
				trimesh::point xyz= m_src->vertices[idx];
				src_lm_verts[i]=Eigen::Vector3d(xyz[0],xyz[1],xyz[2]);
			}

			cout<<"X Change:"<<(X-TmpX).norm()<<endl<<endl;
			if((X-TmpX).norm() < 2) loop = false;

			iter++;

		}

		cout<<"Sub iter:"<<iter<<endl;
		return X;
	}

	vector<std::pair<int, int> > GetFinalCorrespondence(){
		vector<std::pair<int, int> >Correspondence;
		for( int i = 0; i < src_vertsNum; i++ ){
			// if( m_weights[i] )
				Correspondence.push_back(m_soft_corres[i]);	
		}
		return Correspondence;
	}
};
