#include <iostream>

#ifndef EDGE_H
#define EDGE_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Core>

using namespace std;

void edge_detection(pcl::PointCloud<pcl::PointXYZ>::Ptr reference, pcl::PointCloud<pcl::PointXYZ>::Ptr edgePoints, int k, double lambda);
#endif