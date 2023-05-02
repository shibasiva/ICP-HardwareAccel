#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <Eigen/Dense>
#include <stdio.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree_search.h>

// #include <pcl/gpu/octree/octree.hpp>
// #include <pcl/gpu/containers/device_array.hpp>

// #include "../..pcl/gpu/octree/octree.hpp"
// #include "../pcl/gpu/octree/include/pcl/gpu/octree/octree.hpp"

// #include "cuda_runtime.h"
extern "C"{
    #include "./cuPCL/cuOctree/lib/cudaOctree.h"
}


// #pragma diag_suppress 20012
// add in cmd to supress eigen warnings
//--diag-suppress 20012
using namespace std;
using namespace Eigen;
using namespace pcl;

void ICP(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference);

void GetInfo(void)
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

// __global__ 
// void NearestNeighborSearch(vector<PointXYZ>* search_cloud, octree::OctreePointCloudSearch<PointXYZ>* octree, MatrixXd* result)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
//     int closest_point_index;
//     float closest_point_distance;
//     octree->approxNearestSearch((*search_cloud)[idx], closest_point_index, closest_point_distance);  
//     result->col(idx) = Vector2d(closest_point_index, closest_point_distance);
//     return;
// }

//map the source onto the reference
void ICP(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference)
{
    int max_iter = 100; // max iterations
    double convergence_criteria = 0.01;
    // float resolution = 128.0; 

    Matrix3d total_rotation = Matrix3d::Identity();
    Vector3d total_translation = Vector3d::Zero();

    for (int iter = 0; iter < max_iter; iter++) // iterations
    { 
        cout<<"iter: "<<iter<<endl;
        // cout<<"source cloud size: "<< source->points.size()<<endl;
        MatrixXd source_cloud_matrix(3, source->points.size()); //X
        MatrixXd matched_cloud_matrix(3, source->points.size()); //P

        //for every point in the source, find the closest point in the reference
        //calculate the center of mass
        //break if rms is low enough
        
        //following from this code: https://github.com/NVIDIA-AI-IOT/cuPCL/blob/main/cuOctree/main.cpp
        cudaStream_t stream = NULL;
        cudaStreamCreate ( &stream );
        
        unsigned int nCount = source->width * source->height;
        float *inputData = (float *)source->points.data();

        unsigned int nDstCount = reference->width * reference->height;
        float *outputData = (float *)reference->points.data();

        float *input = NULL;//points cloud source which be searched
        cudaMallocManaged(&input, sizeof(float) * 4 * nCount, cudaMemAttachHost);
        cudaStreamAttachMemAsync (stream, input);
        cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        float *output = NULL;// Dst is the targets points
        cudaMallocManaged(&output, sizeof(float) * 4 *nDstCount, cudaMemAttachHost);
        cudaStreamAttachMemAsync (stream, output);
        cudaMemsetAsync(output, 0, sizeof(unsigned int), stream);
        cudaMemcpyAsync(output, outputData, sizeof(float) * 4 * nDstCount, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        float *search = NULL;//search point (one point)
        cudaMallocManaged(&search, sizeof(float) * 4, cudaMemAttachHost);
        cudaStreamAttachMemAsync (stream, search);
        cudaStreamSynchronize(stream);

        unsigned int *selectedCount = NULL;//count of points selected
        checkCudaErrors(cudaMallocManaged(&selectedCount, sizeof(unsigned int)*nDstCount, cudaMemAttachHost));
        checkCudaErrors(cudaStreamAttachMemAsync(stream, selectedCount) );
        checkCudaErrors(cudaMemsetAsync(selectedCount, 0, sizeof(unsigned int)*nDstCount, stream));

        int *index = NULL;//index selected by search
        cudaMallocManaged(&index, sizeof(int) * nCount, cudaMemAttachHost);
        cudaStreamAttachMemAsync (stream, index);
        cudaMemsetAsync(index, 0, sizeof(unsigned int), stream);
        cudaStreamSynchronize(stream);

        float *distance = NULL;//suqure distance between points selected by search
        cudaMallocManaged(&distance, sizeof(float) * nCount, cudaMemAttachHost);
        cudaStreamAttachMemAsync (stream, distance);
        cudaMemsetAsync(distance, 0, sizeof(unsigned int), stream);
        cudaStreamSynchronize(stream);

        float resolution = 0.03f;
        cudaTree treeTest(input, nCount, resolution, stream);

        cudaMemsetAsync(index, 0, sizeof(unsigned int), stream);
        cudaMemsetAsync(distance, 0xFF, sizeof(unsigned int), stream);
        cudaMemsetAsync(selectedCount, 0, sizeof(unsigned int), stream);
        cudaStreamSynchronize(stream);

        cudaMemsetAsync(index, 0, sizeof(unsigned int), stream);
        cudaMemsetAsync(distance, 0xFF, sizeof(unsigned int), stream);
        cudaMemsetAsync(selectedCount, 0, sizeof(unsigned int), stream);
        cudaStreamSynchronize(stream);

        int *pointIdxANSearch = index;
        float *pointANSquaredDistance = distance;
        int status = 0;
        *selectedCount = nDstCount;

        cudaDeviceSynchronize();

        status = treeTest.approxNearestSearch(output, pointIdxANSearch, pointANSquaredDistance, selectedCount);

        cudaDeviceSynchronize();

        // cout<<"did all the cuda stuff"<<endl;
        if (status != 0){
            cerr<<"Failed to find approx nearest search"<<endl;
            return;
        } 
        double rms = 0.0;
        for(int i = 0; i < *selectedCount; i ++) {
            rms += ( *(((unsigned int*)pointANSquaredDistance) + i) )/1e9;
        }

        rms = sqrt(rms/(*selectedCount));
        cout<<"rms: " <<rms<<endl;
        if(rms < convergence_criteria){
            cout<<"final rms: " <<rms<<endl;
            break;
        }
        // cout<<"source cloud matrix" << endl;
        // cout<<source_cloud_matrix<<endl;
        // cout<<"matched cloud matrix" <<endl;
        // cout<<matched_cloud_matrix<<endl;

        for(int i = 0; i < *selectedCount; i++){
            Vector3d source_point (source->points[i].x, source->points[i].y, source->points[i].z);
            source_cloud_matrix.col(i) = source_point;

            int matched_index = *(((unsigned int*)pointIdxANSearch) + i);
            Vector3d matched_point (reference->points[matched_index].x, reference->points[matched_index].y, reference->points[matched_index].z);
            matched_cloud_matrix.col(i) = matched_point;
        }

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
        JacobiSVD<MatrixXd, ComputeThinU | ComputeThinV> svd(covariances); //this is different from the documentation, likely due to a bug: https://stackoverflow.com/questions/72749955/unable-to-compile-the-example-for-eigen-svd
        // svd.compute(covariances);
        // cout<<"found U and V"<<endl;

        //compute rotation and translation
        Matrix3d rotation = svd.matrixU() * (svd.matrixV().transpose());
        Vector3d translation = source_center_of_mass - rotation * matched_center_of_mass;

        
        // total_rotation *= rotation.transpose();
        // total_translation -= translation;

        //create transform
        Matrix4d transform = Matrix4d::Identity();
        transform.block<3,3>(0,0) = rotation.transpose();
        transform.block<3,1>(0,3) = -translation;
        transformPointCloud (*source, *source, transform);
    }
    //write result as pcd
    *source += *reference;
    pcl::io::savePCDFileASCII ("result.pcd", *source);

    cout<<"saved ICP-CPU output to result.pcd"<<endl;
}

int main(int argc, char** argv){
    
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
        GetInfo();
        ICP(source, reference);
        return 0;
    }
}