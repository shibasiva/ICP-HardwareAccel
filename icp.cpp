#include <iostream>
#include <vector>
#include <cmath>
#include <pcl/kdtree/kdtree_flann.h>
#include "Point.h"
using namespace std;

extern double distance(Point p1, Point p2);
extern void icp(vector<Point>& source, vector<Point>& target);

double distance(Point p1, Point p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
}

void ICP(vector<Point>& source, vector<Point>& target)
{
    int max_iter = 100; // max iterations
    double convergence_criteria = 0.001;
    int size = source.size();

    // Create a KD-Tree from the target points
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < size; i++) {
        pcl::PointXYZ p(target[i].x, target[i].y, target[i].z);
        target_cloud->push_back(p);
    }
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(target_cloud);

    for (int i = 0; i < max_iter; i++) // iterations
    {
        vector<int> closest_points(size);
        vector<float> distance_squared(size);
        for (int j = 0; j < size; j++)
        {
        pcl::PointXYZ search_point(source[j].x, source[j].y, source[j].z);
        //kdtree.nearestKSearch(search_point, 1, closest_points[j], distance_squared[j]);
    }

        // Compute the transformation to align source with target
        double tx = 0, ty = 0, tz = 0;
        for (int j = 0; j < size; j++)
        {
            tx += target[closest_points[j]].x - source[j].x;
            ty += target[closest_points[j]].y - source[j].y;
            tz += target[closest_points[j]].z - source[j].z;
        }
        tx /= size;
        ty /= size;
        tz /= size;

        for (int j = 0; j < size; j++)
        {
            source[j].x += tx;
            source[j].y += ty;
            source[j].z += tz;
        }

        // Check for convergence; if rms is lower than convergenace than break
        double rms_error = 0;
        for (int j = 0; j < size; j++)
        {
            rms_error += pow(distance(source[j], target[closest_points[j]]), 2);
        }
        rms_error = sqrt(rms_error / size);
        if (rms_error < convergence_criteria)
        {
            break;
        }
    }
}

int main()
{
    // Sample usage
    vector<Point> source;
    vector<Point> target;

    source.push_back(Point(1, 2, 3));
    source.push_back(Point(4, 5, 6));
    source.push_back(Point(7, 8, 9));

    target.push_back(Point(1, 2, 4));
    target.push_back(Point(4, 5, 6));
    target.push_back(Point(7, 9, 10));

    ICP(source, target);

    // Print the transformed source points
    for (auto p : source)
    {
        cout << "(" << p.x << ", " << p.y << ", " << p.z << ")" << endl;
    }

    return 0;
}