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

using namespace std;
using namespace Eigen;
using namespace pcl;

void ICP(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference);

//map the source onto the reference
void ICP(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference)
{
    int max_iter = 100; // max iterations
    double convergence_criteria = 0.01;
    float resolution = 128.0; 
    Matrix3d total_rotation = Matrix3d::Identity();
    Vector3d total_translation = Vector3d::Zero();

    cout<< "reference clouid size: " << reference->points.size()<<endl;
    //Create Octree
    octree::OctreePointCloudSearch<PointXYZ> octree (resolution);
    octree.setInputCloud (reference);
    octree.addPointsFromInputCloud();

    cout<<"octree created" <<endl;


    for (int iter = 0; iter < max_iter; iter++) // iterations
    { 
        cout<<"iter: "<<iter<<endl;
        // cout<<"source cloud size: "<< source->points.size()<<endl;
        MatrixXd source_cloud_matrix(3, source->points.size()); //X
        MatrixXd matched_cloud_matrix(3, source->points.size()); //P

        //for every point in the source, find the closest point in the reference
        //calculate the center of mass
        //break if rms is low enough
        double rms = 0.0;
        for(int index = 0; index < source->points.size(); index++){
            int closest_point_index;
            float closest_point_distance;
            Vector3d search_point (source->points[index].x, source->points[index].y, source->points[index].z);
            Vector3d transformed_point = total_rotation * search_point + total_translation;
            PointXYZ transformed_pointXYZ = PointXYZ(transformed_point(0), transformed_point(1), transformed_point(2));
            octree.approxNearestSearch(transformed_pointXYZ, closest_point_index, closest_point_distance); //faster than actual nearest search

            Vector3d source_point(transformed_pointXYZ.x, transformed_pointXYZ.y, transformed_pointXYZ.z);
            source_cloud_matrix.col(index) = source_point;
            Vector3d matched_point(reference->points[closest_point_index].x, reference->points[closest_point_index].y, reference->points[closest_point_index].z);
            // cout<<"matched point: " << matched_point << endl;
            matched_cloud_matrix.col(index) = matched_point;

            rms += closest_point_distance;
        }
        // cout<<"found closest points"<<endl;

        rms = sqrt(rms/source->points.size());
        cout<<"rms: " <<rms<<endl;
        if(rms < convergence_criteria){
            cout<<"final rms: " <<rms<<endl;
            break;
        }
        // cout<<"source cloud matrix" << endl;
        // cout<<source_cloud_matrix<<endl;
        // cout<<"matched cloud matrix" <<endl;
        // cout<<matched_cloud_matrix<<endl;

        Vector3d source_center_of_mass = source_cloud_matrix.rowwise().mean();
        // cout<<source_center_of_mass<<endl;
        source_cloud_matrix = source_cloud_matrix.colwise() - source_center_of_mass; //TODO: check this math: https://stackoverflow.com/questions/42811084/eigen-subtracting-vector-from-matrix-columns
        
        Vector3d matched_center_of_mass = matched_cloud_matrix.rowwise().mean();
        // cout<<matched_center_of_mass<<endl;
        matched_cloud_matrix = matched_cloud_matrix.colwise() - matched_center_of_mass; //TODO: check this math
        // cout<<"found center of masses"<<endl;

        //compute dxd matrix of covariances W
        Matrix3d covariances = Matrix3d::Zero();
        for(int col = 0; col < source->points.size(); col++){
            covariances = covariances + (source_cloud_matrix.col(col) * matched_cloud_matrix.col(col).transpose());
        }

        // cout<<covariances<<endl;
        // cout<<covariances.rows()<<endl;
        // cout<<covariances.cols()<<endl;

        //compute singular value decomposition U and V
        JacobiSVD<MatrixXd> svd; //this is different from the documentation, likely due to a bug: https://stackoverflow.com/questions/72749955/unable-to-compile-the-example-for-eigen-svd
        svd.compute(covariances, ComputeThinU | ComputeThinV);
        // cout<<"found U and V"<<endl;

        //compute rotation and translation
        Matrix3d rotation = svd.matrixU() * (svd.matrixV().transpose());
        Vector3d translation = source_center_of_mass - rotation * matched_center_of_mass;

        
        total_rotation *= rotation.transpose();
        total_translation -= translation;
    }

    //create transform
    Matrix4d transform = Matrix4d::Identity();
    transform.block<3,3>(0,0) = total_rotation;
    transform.block<3,1>(0,3) = total_translation;
    transformPointCloud (*source, *source, transform);
    //write result as pcd
    *source += *reference;

    pcl::io::savePCDFileASCII ("result.pcd", *source);

    cout<<"saved ICP-CPU output to result.pcd"<<endl;
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

        ICP(source, reference);
        return 0;
    }

    
}