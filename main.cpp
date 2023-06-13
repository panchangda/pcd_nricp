#include "NonRigidICP.h"

/* arg parse header */
#ifdef _WIN32
# include "wingetopt.h"
#else
# include <unistd.h>
# endif

using namespace std;

std::vector<int> LoadLandMarksIdx(string filepath)
{
    std::vector<int> landmarksIdx;
    int idx;
    ifstream fin(filepath);
    for (int i = 0; i < NUM_LANDMARKS; i++)
    {
        fin >> idx;
        landmarksIdx.push_back(idx);
    }
    fin.close();
    return landmarksIdx;
}
std::vector<Eigen::Vector3d> GetLandMarkVertices(std::vector<int>& landmark_index, trimesh::TriMesh *mesh)
{   
    std::vector<Eigen::Vector3d> landmark_vertices;
    for (int i = 0; i < landmark_index.size(); i++)
    {
        int idx = landmark_index[i];
        trimesh::point xyz = mesh->vertices[idx];
        landmark_vertices.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
    }
    return landmark_vertices;
}
int main(int argc, char **argv)
{

    MyTimer timer;
    timer.start();

    std::string rootPath = "/home/pcd/vscodes/pcd_nricp";
    std::string dataPath = rootPath + "/data";
    std::string srcFile = dataPath + "/source.obj";
    std::string tgtFile = dataPath + "/DECA.obj";
    std::string srcLandmarksFile = dataPath + "/source_landmarks_index.txt";
    std::string tgtLandmarksFile = dataPath + "/target_landmarks_index.txt";
    std::string savePath = rootPath + "/result/";

    
    int c;
    if(argc == 3){
        tgtFile = std::string(argv[1]);
        cout << "specify target file path as " << tgtFile << endl;
        tgtLandmarksFile = dataPath + "/" + std::string(argv[2])+"_landmarks_index.txt";
        cout << "specify landmarks file path as " << tgtLandmarksFile << endl;
    }

    std::string command = "mkdir -p " + savePath;
    system(command.c_str());

    trimesh::TriMesh* src,* tgt;
    src = trimesh::TriMesh::read(srcFile);
    tgt = trimesh::TriMesh::read(tgtFile);

    src->need_neighbors();
    src->need_normals();
    cout<<"has normals: "<<src->normals.size()<<endl;
    tgt->need_neighbors();
    tgt->need_normals();

    std::vector<int> src_landmark_index = LoadLandMarksIdx(srcLandmarksFile);
    std::vector<int> tgt_landmark_index = LoadLandMarksIdx(tgtLandmarksFile);
    std::vector<Eigen::Vector3d> src_landmark_vertices = GetLandMarkVertices(src_landmark_index,src);
    std::vector<Eigen::Vector3d> tgt_landmark_vertices = GetLandMarkVertices(tgt_landmark_index, tgt);

    NonRigidICP nricp(src, tgt,
    src_landmark_index, src_landmark_vertices,
    tgt_landmark_index, tgt_landmark_vertices);
    nricp.Init();

    std::string srcf=savePath+"srcAfterAlign.obj";
	src->write(srcf);
    std::string tgtf=savePath+"tgtAfterAlign.obj";
    tgt->write(tgtf);

    double max_alpha=200;
	double min_alpha=1;
    // beta beta beta
	double beta = 0;
	double gamma = 1.0;
	int step = 10;

    // initialize X {4n x 3}
    Eigen::MatrixX3d X(4 * src->vertices.size(), 3);
	X.setZero();

    for(int i = 1; i <= step; ++i)
    {
        double alpha = max_alpha - i * (max_alpha - min_alpha) / step;
        std::cout << "*********************************"<<endl;
		std::cout<<"Iteration:" <<i<<"  alpha:"<<alpha<<endl;

        X = nricp.compute(alpha, beta, gamma);
        // std::string filename = savePath + std::to_string(i)+".obj";
		// src->write(filename);

    }

    // move src to its initial location
    nricp.ResetSrc();

    // export correspondences between src & tgt
    vector<std::pair<int, int> >Correspondence = nricp.GetFinalCorrespondence();
    ofstream fout(savePath+"Corr.txt",ios::out);
    for(int i = 0 ; i < Correspondence.size(); i++)
        fout << Correspondence[i].first << " " << Correspondence[i].second << endl;
    fout.close();


    
    trimesh::TriMesh* original_src = trimesh::TriMesh::read(srcFile);

    vector<int>boundry = LoadLandMarksIdx(dataPath+"/boundry.txt");
    for(auto& i:boundry){
        trimesh::point p = original_src->vertices[i];
        trimesh::point _p = src->vertices[i];
        src->vertices[i] = (original_src->vertices[i] - src->vertices[i]) / 2 + src->vertices[i];
        // src->vertices[i] = original_src->vertices[i];
    }

    for(int i = 0; i < 40; i++){
        trimesh::point p = original_src->vertices[i];
        trimesh::point _p = src->vertices[i];
        // printf("original:%f %f %f ",p[0],p[1],p[2]);
        // printf("deformed:%f %f %f\n",_p[0],_p[1],_p[2]);
        src->vertices[i] = original_src->vertices[i];
    }


    // export src mesh
    src->need_neighbors();
    src->need_normals();

    std::string filename=savePath+"result.obj";
	src->write(filename);

	cout<<"All Cost time:";
	timer.end();
    
    return 0;
}