#include <iostream>
#include <Eigen/Core>
#include <string>
#include <filesystem>
using namespace std;
int main(){

	// Eigen norm() normalized() normalize()
	// Eigen::Vector3d v1,v2,v3;
	// v1 << 1, 1, 1;
	// v2 << 2, 2, 2;
	// v3 << 3, 3, 3;
	// Eigen::Matrix3d M;
	// M << v1, v2, v3;
	// cout << v2.norm()<<endl;
	// cout << v2.normalized()<<endl;

	// 遍历文件夹下所有文件名
	string out = "../data/result";
	string dfmFileFolder = "../data/blendshapes";
	for (const auto & entry : std::filesystem::directory_iterator(dfmFileFolder))
    {
	    std::cout << entry.path() << std::endl;
		string p = entry.path();
		int pos = p.rfind("/");
		string result  = out+p.substr(pos);
		cout<<result<<endl;
	}
}