#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <random>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
using namespace std;
using namespace Eigen;
using namespace pcl;

void random_transform_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, string file, string token){
    //https://stackoverflow.com/questions/2704521/generate-random-double-numbers-in-c
    cout << cloud->points.size()<<endl;
    double min_angle_change = 0;
    double max_angle_change = 30 * M_PI / 180;
    uniform_real_distribution<double> unif_angle(min_angle_change, max_angle_change);
    default_random_engine re_angle;
    double roll = unif_angle(re_angle);
    double pitch = unif_angle(re_angle);
    double yaw = unif_angle(re_angle);

    double min_dist_change = 0.01;
    double max_dist_change = 0.2;
    uniform_real_distribution<double> unif_translate(min_dist_change, max_dist_change);
    default_random_engine re_translate;
    double x = unif_translate(re_translate);
    double y = unif_translate(re_translate);
    double z = unif_translate(re_translate);

    cout <<"translation xyz: " << x << " " << y << " " << z << endl;
    cout<< "angles: " << roll << " " << pitch << " " << yaw<<endl; 

    //https://stackoverflow.com/questions/21412169/creating-a-rotation-matrix-with-pitch-yaw-roll-using-eigen
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());
    Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3d rotation_matrix = q.matrix();

    Eigen::Vector3d translation_vector(x, y, z);

    Matrix4d transform = Matrix4d::Identity();
    transform.block<3,3>(0,0) = rotation_matrix;
    transform.block<3,1>(0,3) = translation_vector;

    //apply transform
    transformPointCloud (*cloud, *cloud, transform);

    //https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c  
    string filename = file.substr(0, file.find(".pcd"));
    string extension = file.substr(file.find(".pcd") + 1);

    string result = filename + token + "."  + extension;
    cout<<result<<endl;
    pcl::io::savePCDFileASCII (result, *cloud);

    cout<<"saved ICP-CPU output to " << result <<endl;
}
int main(int argc, char** argv){

    if(argc != 3){
        cout<<"Usage: ./random_transform [pcd source] [token] "<<endl;
        return 0;
    }
    PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        
    if (io::loadPCDFile<PointXYZ> (argv[1], *cloud) == -1){
        string s = argv[1];
        cout<< "Couldn't read file " + s + "\n" << endl;
        return (-1);
    }
    cout<<"loaded pcl"<<endl;
    random_transform_cloud(cloud, argv[1], argv[2]);
    return 0;
}