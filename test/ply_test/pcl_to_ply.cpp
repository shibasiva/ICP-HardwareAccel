#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <Eigen/Dense>
#include <stdio.h>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d.h>

using namespace std;
using namespace pcl;
int main(int argc, char** argv){
    
    if(argc < 2){
        cout<<"Usage: ./icp_cpp [pcd source] [pcd referece]"<<endl;
        return 0;
    }
    else{
        for(int i = 1; i < argc; i++){
            cout<<"working on: " << argv[i] << endl;
            PointCloud<PointXYZ>::Ptr source (new PointCloud<PointXYZ>);
            PointCloud<PointNormal>::Ptr dest (new PointCloud<PointNormal>);
            string file = argv[i];
            if (io::loadPCDFile<PointXYZ> (file, *source) == -1){
                cout<< "Couldn't read file " + file + "\n" << endl;
                return (-1);
            }
            
            //https://stackoverflow.com/questions/34400656/how-can-i-compute-a-normal-for-each-point-in-cloud
            NormalEstimation<PointXYZ, Normal> ne;
            ne.setInputCloud (source);

            search::KdTree<PointXYZ>::Ptr tree (new search::KdTree<PointXYZ> ());
            ne.setSearchMethod (tree);

            PointCloud<Normal>::Ptr cloud_normals (new PointCloud<Normal>);
            ne.setRadiusSearch (5);
            ne.compute (*cloud_normals);

            dest->height = source->height;
            dest->width = source->width;
            dest->is_dense = source->is_dense;
            dest->points.resize(dest->width * dest->height);

            for (int i = 0; i < source->points.size(); i++)
            {
                dest->points[i].x = source->points[i].x;
                dest->points[i].y = source->points[i].y;
                dest->points[i].z = source->points[i].z;

                // dest->points[i].curvature = cloud_normals->points[i].curvature;

                dest->points[i].normal_x = cloud_normals->points[i].normal_x;
                dest->points[i].normal_y = cloud_normals->points[i].normal_y;
                dest->points[i].normal_z = cloud_normals->points[i].normal_z;
            }

            string filename = file.substr(0, file.find(".pcd"));

            string result = filename + "." + "ply";
            io::savePLYFile(result, *dest);
        }
        
        return 0;
    }
}