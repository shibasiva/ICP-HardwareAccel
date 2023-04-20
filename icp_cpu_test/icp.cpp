#define PCL_NO_PRECOMPILE
#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <limits>

//Eigen
#include <Eigen/Dense>

//PCL
#include <pcl/memory.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>
#include <pcl/octree/impl/octree_search.hpp>



using namespace std;
using namespace Eigen;
using namespace pcl;

//https://pointclouds.org/documentation/tutorials/adding_custom_ptype.html
struct EIGEN_ALIGN16 PointXYZW{
    PCL_ADD_POINT4D;
    float w;
    PCL_MAKE_ALIGNED_OPERATOR_NEW; 
    PointXYZW(){
        x = 0.0;
        y = 0.0;
        z = 0.0;
        w = 0.0;
    }
    PointXYZW(float x_set, float y_set, float z_set, float w_set){
        x = x_set;
        y = y_set;
        z = z_set;
        w = w_set;
    }
};

//custom 4-d points; adding w weighted points
// Article Source: Robust iterative closest point algorithm based on global reference point for rotation invariant registration
// Du S, Xu Y, Wan T, Hu H, Zhang S, et al. (2017) Robust iterative closest point algorithm based on global reference point for rotation invariant registration. 
// PLOS ONE 12(11): e0188039. https://doi.org/10.1371/journal.pone.0188039
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZW,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, w, w))

// PCL_INSTANTIATE(KdTree, PointXYZW);
// PCL_INSTANTIATE(OctreePointCloudSearch, PointXYZW);
PCL_INSTANTIATE_PointCloud(PointXYZW);
PCL_INSTANTIATE(OctreePointCloudSearch, PointXYZW);

MatrixXf CreateMatrixFromCloud(PointCloud<PointXYZ>::Ptr cloud, Matrix3f rotation, Vector3f translation);
MatrixXf CreateWeightedCloudAndMatrix(PointCloud<PointXYZW>::Ptr cloud, PointCloud<PointXYZW>::Ptr weighted_cloud);
void ComputeW(PointCloud<PointXYZW>::Ptr cloud, Vector3f centroid, float w);
void ICP(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference);

MatrixXf CreateMatrixFromCloud(PointCloud<PointXYZ>::Ptr cloud, Matrix3f rotation, Vector3f translation){
    MatrixXf new_matrix = MatrixXf::Zero(3, cloud->points.size());
    for(int i = 0; i < cloud->points.size(); i++){
        Vector3f point(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
        new_matrix.col(i) = rotation.transpose()*point - translation;
    }
    return new_matrix;
}

MatrixXf CreateWeightedCloudAndMatrix(PointCloud<PointXYZ>::Ptr cloud, PointCloud<PointXYZW>::Ptr weighted_cloud){
    MatrixXf weighted_matrix = MatrixXf::Zero(4, cloud->points.size());
    for(int i = 0; i < cloud->points.size(); i++){

        Vector4f point(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 0);
        weighted_matrix.col(i) = point;

        PointXYZW point_XYZW = PointXYZW(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 0.0);//
        weighted_cloud->push_back(point_XYZW);
    }
    return weighted_matrix;
}


//compute W with pointer to cloud
void ComputeW(PointCloud<PointXYZW>::Ptr cloud, Vector3f centroid, float w){
    for(int point = 0; point < cloud->points.size(); point++){

        Vector3f current_point(cloud->points[point].x, cloud->points[point].y, cloud->points[point].z);

        float a = current_point(0) - centroid(0);
        float b = current_point(1) - centroid(1);
        float c = current_point(2) - centroid(2);
        cloud->points[point].w = w * sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
        // cout<<"assigned w" << endl;
    } 
}

//map the source onto the reference
void ICP(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference)
{

    int max_iter = 1; // max iterations
    float convergence_criteria = 0.001;
    float resolution = 128.0;  

    Matrix3f total_rotation = Matrix3f::Identity();
    Vector3f total_translation = Vector3f::Zero();
    float w = 1000000000;
    float T = 0.025;
    float alpha = 20;


    cout<<"initialized constants" <<endl;

    PointCloud<PointXYZW>::Ptr source_weighted_cloud(new PointCloud<PointXYZW>); //TODO: remove this from code
    PointCloud<PointXYZW>::Ptr reference_weighted_cloud(new PointCloud<PointXYZW>);

    MatrixXf source_weighted_matrix = CreateMatrixFromCloud(source, total_rotation, total_translation);
    MatrixXf reference_weighted_matrix = CreateWeightedCloudAndMatrix(reference, reference_weighted_cloud);
    cout<<"created custom point cloud"<<endl;
    cout<<"point clouds with w created"<<endl;

    //compute centroid of both source and reference set
    Vector3f source_centroid = source_weighted_matrix.rowwise().mean();
    Vector3f reference_centroid = reference_weighted_matrix.rowwise().mean()(seqN(0, 3));

    
    cout<<"computed W"<<endl;

    // //Create Octree
    // octree::OctreePointCloudSearch<PointXYZW> octree (resolution);
    // octree.setInputCloud (reference_weighted_cloud);
    // octree.addPointsFromInputCloud();

    //create kd-tree
    KdTreeFLANN<PointXYZW> kdtree;
    kdtree.setInputCloud(reference_weighted_cloud);
    // KdTree<PointXYZW> kdtree;

    for (int iter = 0; iter < max_iter; iter++) // iterations
    { 

        cout<<"iter: "<<iter<<endl;
        // cout<<"source cloud size: "<< source->points.size()<<endl;
        MatrixXf source_cloud_matrix(3, source->points.size()); //X
        MatrixXf matched_cloud_matrix(3, source->points.size()); //P

        //recalculate w for reference set
        ComputeW(reference_weighted_cloud, reference_centroid, w);

        source_weighted_matrix = CreateMatrixFromCloud(source, total_rotation, total_translation);
        source_centroid = source_weighted_matrix.rowwise().mean();
        cout<<"source centroid and weighted matrix recomputed"<<endl;
        cout<<source_centroid<<endl;

        //for every point in the source, find the closest point in the reference
        //calculate the center of mass
        //break if rms is low enough
        float rms = 0.0;
        int k = 1;
        for(int index = 0; index < source_weighted_matrix.cols(); index++){
            vector<int> closest_point_index(k);
            vector<float> closest_point_distance(k);
            // octree.approxNearestSearch(source_weighted_cloud->points[index], closest_point_index, closest_point_distance); //faster than actual nearest search
            // PointXYZW point = source_weighted_cloud->points[index];   
            float x = source_weighted_matrix.coeff(0, index);
            float y = source_weighted_matrix.coeff(1, index);
            float z = source_weighted_matrix.coeff(2, index);
            float w_index = w * sqrt(pow(x - source_centroid(0), 2) + 
                                     pow(y - source_centroid(1), 2) + 
                                     pow(z - source_centroid(2), 2));

            PointXYZW search_point (x, y, z, w);
            // cout<<"search point found"<<endl;
            kdtree.nearestKSearch(search_point, k, closest_point_index, closest_point_distance);
            // cout<<"kd search found"<<endl;
            Vector3f source_point(x, y, z);
            // cout<<source_point<<endl;
            
            source_cloud_matrix.col(index) = source_point;

            Vector3f matched_point(reference_weighted_cloud->points[closest_point_index[0]].x,
                                   reference_weighted_cloud->points[closest_point_index[0]].y,
                                   reference_weighted_cloud->points[closest_point_index[0]].z
                                   );
            matched_cloud_matrix.col(index) = matched_point;

            rms += pow(closest_point_distance[0],2);
        }

        rms = sqrt(rms/source_weighted_matrix.rows());
        if(rms < T){
            w = rms / T;
        }
        cout<<"rms: " << rms <<endl;
        if(rms < convergence_criteria){
            cout<<"final rms: " <<rms<<endl;
            break;
        }
        // cout<<"source cloud matrix" << endl;
        // cout<<source_cloud_matrix<<endl;
        // cout<<"matched cloud matrix" <<endl;
        // cout<<matched_cloud_matrix<<endl;

        // cout<<"computed centroid source"<<endl;
        Vector3f source_center_of_mass = source_cloud_matrix.rowwise().mean();
        cout<<source_center_of_mass<<endl;
        // cout<<source_centroid<<endl;
        source_cloud_matrix = source_cloud_matrix.colwise() - source_centroid; //TODO: check this math: https://stackoverflow.com/questions/42811084/eigen-subtracting-vector-from-matrix-columns
        // cout<<"computed centroid matched"<<endl;
        Vector3f matched_center_of_mass = matched_cloud_matrix.rowwise().mean();
        cout<<matched_center_of_mass<<endl;
        matched_cloud_matrix = matched_cloud_matrix.colwise() - matched_center_of_mass; //TODO: check this math
        // cout<<"found center of masses"<<endl;

        //compute dxd matrix of covariances W
        Matrix3f covariances = Matrix3f::Zero();
        for(int col = 0; col < source_weighted_matrix.rows(); col++){
            covariances = covariances + (source_cloud_matrix.col(col)*(matched_cloud_matrix.col(col).transpose()));
        }


        //compute singular value decomposition U and V
        JacobiSVD<MatrixXf> svd; //this is different from the documentation, likely due to a bug: https://stackoverflow.com/questions/72749955/unable-to-compile-the-example-for-eigen-svd
        svd.compute(covariances, ComputeThinU | ComputeThinV);
        // cout<<"found U and V"<<endl;

        //compute rotation and translation
        Matrix3f rotation = svd.matrixU() * (svd.matrixV().transpose());
        Vector3f translation = source_center_of_mass - rotation * matched_center_of_mass;
        //create transform
        // Matrix4f transform = Matrix4f::Identity();
        // if(iter != 0){
        //     transform.block<3,3>(0,0) = rotation.transpose();
        // }
        // transform.block<3,1>(0,3) = -translation;
        // transformPointCloud (*source, *source, transform);
        
        total_rotation *= rotation;
        total_translation += translation;
    }
    Matrix4f transform = Matrix4f::Identity();
    transform.block<3,3>(0,0) = total_rotation.transpose();
    transform.block<3,1>(0,3) = -total_translation;
    transformPointCloud (*source, *source, transform);

    
    //write result as pcd
    *reference += *source;

    pcl::io::savePCDFileASCII ("result.pcd", *reference);

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