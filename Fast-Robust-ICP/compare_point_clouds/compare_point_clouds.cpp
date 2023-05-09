#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <fstream>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree_search.h>

using namespace std;
using namespace Eigen;
using namespace pcl;

int main(int argc, char** argv)
{
    if(argc != 4){
        cout<<"Usage: ./icp_cpp [pcd source] [pcd reference] [transform.txt]"<<endl;
        return 0;
    }
    else{
        PointCloud<PointXYZ>::Ptr source (new PointCloud<PointXYZ>);
        PointCloud<PointXYZ>::Ptr reference (new PointCloud<PointXYZ>);

        if (io::loadPCDFile<PointXYZ> (argv[1], *source) == -1){
            string s = argv[1];
            cout<< "Couldn't read file " + s + "\n" << endl;
            return (-1);
        }
        if (io::loadPCDFile<PointXYZ> (argv[2], *reference) == -1){
            string s = argv[2];
            cout<< "Couldn't read file " + s + "\n" << endl;
            return (-1);
        }
        Matrix4d transform = Matrix4d::Identity();
        ifstream infile(argv[3]);
        double a, b, c, d;
        int i = 0;
        while(infile >> a >> b >> c >> d){
            Vector4d vec(a, b, c, d);
            transform.row(i) = vec;
            i++;
        }

        transformPointCloud (*reference, *reference, transform);
        
        float resolution = 128.0; 
        octree::OctreePointCloudSearch<PointXYZ> octree (resolution);
        octree.setInputCloud (reference);
        octree.addPointsFromInputCloud();
        
        double rms = 0.0;
        for(int i = 0; i < source->points.size(); i++){
            int closest_point_index;
            float closest_point_distance;

            octree.approxNearestSearch(source->points[i], closest_point_index, closest_point_distance);

            float sx = source->points[i].x;
            float sy = source->points[i].y;
            float sz = source->points[i].z;

            float rx = reference->points[closest_point_index].x;
            float ry = reference->points[closest_point_index].y;
            float rz = reference->points[closest_point_index].z;

            rms += pow((sx - rx), 2) + pow((sy - ry), 2) + pow((sz - rz), 2);
        }

        rms = sqrt(rms/source->points.size());
        cout<< "final rms: " << rms << endl;
        *source += *reference;
        pcl::io::savePCDFileASCII ("result.pcd", *source);
        cout<<"saved ICP-CPU output to result.pcd"<<endl;

        return 0;
    }

    
}