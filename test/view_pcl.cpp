#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
using namespace std;

void view_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    viewer.showCloud (cloud);
    while (!viewer.wasStopped ()){
    }
}
int main(){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
 
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("bunny.pcd", *cloud) == -1){
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }
    cout<<"loaded pcl"<<endl;
    view_cloud(cloud);
    return 0;
}