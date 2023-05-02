#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
using namespace std;
using namespace Eigen;
using namespace pcl;
//map the source onto the reference
void ICP(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference);
//edge detection
void edge_detection_test(PointCloud<PointXYZ>::Ptr reference, PointCloud<PointXYZ>::Ptr edgePoints, int k, double lambda);



void edge_detection_test(PointCloud<PointXYZ>::Ptr reference, PointCloud<PointXYZ>::Ptr edgePoints, int k, double lambda)
{
    for(int i=0; i<reference->points.size(); i++){
        vector<int> indices(k);
        vector<float> sqrDistances(k);
        
        //calculate nearestKNeighbors
        KdTreeFLANN<PointXYZ> kdtree;
        kdtree.setInputCloud(reference); 
        
        //|V_i| nearest neighbors
        kdtree.nearestKSearch(reference->points[i], k, indices, sqrDistances); 

        //Calculate centroid
        Vector3d centroid = Vector3d(reference->points[i].x, reference->points[i].y, reference->points[i].z);
        
        double resolution = sqrt(sqrDistances[k-1]);
        //summing closest neighbors to form centroid
        for(int j=0; j<k; j++){
            centroid += Vector3d(reference->points[indices[j]].x, reference->points[indices[j]].y, reference->points[indices[j]].z);
        }

        //Centroid = s(1/|V_i|) 
        centroid =centroid/(k+1);
        
        //shift == ‖𝐶𝑖 − 𝑝𝑖‖
        double shift = (centroid - Vector3d(reference->points[i].x, reference->points[i].y, reference->points[i].z)).norm();
        
        //if ‖𝐶𝑖 − 𝑝𝑖‖ > 𝜆 ∙ 𝑍𝑖 -> found edge
        if(shift > lambda * resolution){
            edgePoints->push_back(reference->points[i]);
        }
    }
}
void ICP(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference)
{
    int max_iter = 100; // max iterations
    double convergence_criteria = 0.001;
    float resolution = 128.0; 

    //Create Octree
    octree::OctreePointCloudSearch<PointXYZ> octree (resolution);
    octree.setInputCloud (reference);
    octree.addPointsFromInputCloud();


    for (int i = 0; i < max_iter; i++) // iterations
    { 
        cout<<"iter: "<<i<<endl;
        cout<<"source cloud size: "<< source->points.size()<<endl;
        MatrixXd source_cloud_matrix(3, source->points.size()); //X
        MatrixXd matched_cloud_matrix(3, source->points.size()); //P

        //for every point in the source, find the closest point in the reference
        //calculate the center of mass
        //break if rms is low enough
        double rms = 0.0;
        for(int index = 0; index < source->points.size(); index++){
            int closest_point_index;
            float closest_point_distance;
            octree.approxNearestSearch(index, closest_point_index, closest_point_distance); //faster than actual nearest search

            Vector3d source_point(source->points[index].x, source->points[index].y, source->points[index].z);
            source_cloud_matrix.col(i) = source_point;
            Vector3d matched_point(reference->points[closest_point_index].x, reference->points[closest_point_index].y, reference->points[closest_point_index].z);
            matched_cloud_matrix.col(i) = matched_point;

            rms += closest_point_distance;
        }
        cout<<"found closest points"<<endl;

        rms = sqrt(rms/source->points.size());
        cout<<"rms: " <<rms<<endl;
        if(rms < convergence_criteria){
            cout<<"final rms: " <<rms<<endl;
            break;
        }
        Vector3d source_center_of_mass = source_cloud_matrix.rowwise().mean();
        source_cloud_matrix = source_cloud_matrix.colwise() - source_center_of_mass; //TODO: check this math: https://stackoverflow.com/questions/42811084/eigen-subtracting-vector-from-matrix-columns
        
        Vector3d matched_center_of_mass = matched_cloud_matrix.rowwise().mean();
        matched_cloud_matrix = matched_cloud_matrix.colwise() - matched_center_of_mass; //TODO: check this math

        cout<<"found center of masses"<<endl;

        //compute dxd matrix of covariances W
        Matrix3d covariances = Matrix3d::Zero();
        for(int i = 0; i < source->points.size(); i++){
            covariances = covariances + (source_cloud_matrix.col(i) * matched_cloud_matrix.col(i).transpose());
        }

        cout<<"found W"<<endl;
        // cout<<covariances.rows()<<endl;
        // cout<<covariances.cols()<<endl;

        //compute singular value decomposition U and V
        JacobiSVD<MatrixXd> svd; //this is different from the documentation, likely due to a bug: https://stackoverflow.com/questions/72749955/unable-to-compile-the-example-for-eigen-svd
        svd.compute(covariances, ComputeThinU | ComputeThinV);
        cout<<"found U and V"<<endl;

        //compute rotation and translation
        Matrix3d rotation = svd.matrixU() * svd.matrixV().transpose();
        Vector3d translation = source_center_of_mass - rotation * matched_center_of_mass;

        //create transform
        Matrix4d transform = Matrix4d::Identity();
        transform.block<3,3>(0,0) = rotation;
        transform.block<3,1>(0,3) = translation;

        cout<<"found transform"<<endl;

        //apply transform
        transformPointCloud (*source, *source, transform); //TODO: check if this is actually correct
        cout<<"applied transform"<<endl;

        // Check for convergence; if rms is lower than convergence than break
        // double rms_error = 0;
        // for (int j = 0; j < size; j++)
        // {
        //     rms_error += pow(distance(source[j], target[closest_points[j]]), 2);
        // }
        // rms_error = sqrt(rms_error / size);
        // if (rms_error < convergence_criteria)
        // {
        //     break;
        // }
    }
}

int main(int argc, char** argv)
{
    if(argc != 3){
        cout<<"Usage: ./icp_cpp [pcd source] [pcd referece]"<<endl;
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
        PointCloud<PointXYZ>::Ptr edges (new PointCloud<PointXYZ>);

        edge_detection_test(reference, edges, 20, 0.5); // 20 nearest neighbors and 0.5 lambda

        ICP(source, reference);
        
        cout<<"Number of edges: "<<edges->points.size()<<endl;

        return 0;
    }

    
}