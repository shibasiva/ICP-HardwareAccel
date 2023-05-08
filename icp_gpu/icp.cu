#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <Eigen/Dense>
#include <stdio.h>
#include <chrono>
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

//https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
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

__global__ 
void NearestNeighborSearch(float* source, float* reference, int source_len, int reference_len, Matrix3f rotation, Vector3f translation, int* matched_indices, float* matched_distances)
{

    //grid stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < source_len; i += blockDim.x * gridDim.x) {
        int best_matched_index = 0;
        float s_x = source[i*4];
        float s_y = source[i*4 + 1];
        float s_z = source[i*4 + 2];

        // if(i == 0){
        //     printf("first source pt\n");
        //     printf("%f\n", s_x);
        //     printf("%f\n", s_y);
        //     printf("%f\n", s_z);
        // }
        Vector3f sp (s_x, s_y, s_z);
        Vector3f tsp = rotation * sp + translation;

        float best_matched_distance = ((reference[0] - tsp(0))*(reference[0] - tsp(0)) + 
                                       (reference[1] - tsp(1))*(reference[1] - tsp(1)) + 
                                       (reference[2] - tsp(2))*(reference[2] - tsp(2)));
        
        for(int reference_point_index = 1; reference_point_index < reference_len; reference_point_index++){
            float r_x = reference[reference_point_index*4];
            float r_y = reference[reference_point_index*4 + 1];
            float r_z = reference[reference_point_index*4 + 2];
            // printf("examining point: %f %f %f\n", r_x, r_y, r_z);
            float new_matched_distance = ((r_x - tsp(0))*(r_x - tsp(0))+
                                          (r_y - tsp(1))*(r_y - tsp(1))+
                                          (r_z - tsp(2))*(r_z - tsp(2)));

            if(new_matched_distance < best_matched_distance){
                // printf("i: %i | old distance: %f | new matching distance: %f\n", reference_point_index, best_matched_distance, new_matched_distance);

                best_matched_distance = new_matched_distance;
                best_matched_index = reference_point_index;
                // printf("new matching index: %i\n", reference_point);
            }
        }
        // printf("i: %i | matched index: %i | matched distance: %f\n", i, best_matched_index, best_matched_distance);
        matched_indices[i] = best_matched_index;
        matched_distances[i] = best_matched_distance;   
    }
}

//map the source onto the reference
void ICP(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference)
{   
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double, std::ratio<1, 1000>> time_span =
    chrono::duration_cast<chrono::duration<double, ratio<1, 1000>>>(t2 - t1);

    int max_iter = 100; // max iterations
    double convergence_criteria = 0.001;
    // float resolution = 128.0; 

    Matrix3f total_rotation = Matrix3f::Identity();
    Vector3f total_translation = Vector3f::Zero();
    // //following from this code: https://github.com/NVIDIA-AI-IOT/cuPCL/blob/main/cuOctree/main.cpp
    cudaStream_t stream = NULL;
    cudaStreamCreate ( &stream );
    
    //load data onto GPU
    unsigned int nCount = reference->width * reference->height;
    float *referenceData = (float *)reference->points.data();

    unsigned int nDstCount = source->width * source->height;
    float *sourceData = (float *)source->points.data();

    // cout<<"input data size: " << nCount << endl;
    // cout<<"output data size: " << nDstCount << endl;
    // cout<<"reference data:"<<endl;
    // // cout<<referenceData<<endl;
    // cout<<referenceData[0]<<endl;
    // cout<<referenceData[1]<<endl;
    // cout<<referenceData[2]<<endl;
    // cout<<referenceData[3]<<endl;
    // cout<<referenceData[4]<<endl;
    // cout<<referenceData[5]<<endl;
    // cout<<referenceData[6]<<endl;
    // cout<<referenceData[7]<<endl;

    // cout<<"sourceData"<<endl;
    // cout<<sourceData[0]<<endl;
    // cout<<sourceData[1]<<endl;
    // cout<<sourceData[2]<<endl;
    // cout<<sourceData[3]<<endl;
    // cout<<sourceData[4]<<endl;
    // cout<<sourceData[5]<<endl;
    // cout<<sourceData[6]<<endl;
    // cout<<sourceData[7]<<endl;

    t1 = chrono::steady_clock::now(); //time from loading in point clouds into GPU

    float *cuda_source = NULL;//points cloud source which be searched
    gpuErrchk(cudaMallocManaged(&cuda_source, sizeof(float) * 4 * nCount, cudaMemAttachHost));
    gpuErrchk(cudaStreamAttachMemAsync (stream, cuda_source));
    gpuErrchk(cudaMemcpyAsync(cuda_source, sourceData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    
    float *cuda_reference = NULL;// Dst is the targets points
    gpuErrchk(cudaMallocManaged(&cuda_reference, sizeof(float) * 4 *nDstCount, cudaMemAttachHost));
    gpuErrchk(cudaStreamAttachMemAsync (stream, cuda_reference));
    gpuErrchk(cudaMemcpyAsync(cuda_reference, referenceData, sizeof(float) * 4 * nDstCount, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    // float resolution = 0.03f;

    int blockSize = 1024;
    int numBlocks = (nCount + blockSize - 1) / blockSize;
    // cout<<"block size: "<< blockSize<< endl;
    // cout<<"numBlocks: " << numBlocks << endl;


    for (int iter = 0; iter < max_iter; iter++) // iterations
    { 
        cout<<"iter: "<<iter<<endl;
        // cout<<"source cloud size: "<< source->points.size()<<endl;
        MatrixXf source_cloud_matrix(3, source->points.size()); //X
        MatrixXf matched_cloud_matrix(3, source->points.size()); //P

        //for every point in the source, find the closest point in the reference
        //calculate the center of mass
        //break if rms is low enough
        
        
        int *matched_indices;
        gpuErrchk(cudaMallocManaged(&matched_indices, sizeof(int) * nCount, cudaMemAttachHost));
        gpuErrchk(cudaStreamAttachMemAsync (stream, matched_indices));
        gpuErrchk(cudaMemsetAsync(matched_indices, 0, sizeof(unsigned int), stream));
        gpuErrchk(cudaStreamSynchronize(stream));

        float *matched_distances;
        gpuErrchk(cudaMallocManaged(&matched_distances, sizeof(float) * nCount, cudaMemAttachHost));
        gpuErrchk(cudaStreamAttachMemAsync (stream, matched_distances));
        gpuErrchk(cudaMemsetAsync(matched_distances, 0, sizeof(unsigned int), stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        
        NearestNeighborSearch<<<numBlocks, blockSize>>>(cuda_source, cuda_reference, nCount, nDstCount, total_rotation, total_translation, matched_indices, matched_distances);
        gpuErrchk(cudaDeviceSynchronize());

        double rms = 0.0;
        for(int i = 0; i < nCount; i ++) {
            rms += matched_distances[i];
        }

        // rms /= nCount;
        rms = sqrt(rms/nCount);
        cout<<"rms: " <<rms<<endl;
        if(rms < convergence_criteria){
            cout<<"final rms: " <<rms<<endl;
            break;
        }
        
        //cout<<"size of indices: " << sizeof(matched_indices)/sizeof(matched_indices[0])<<endl;
        // for(int i = 0; i < nCount; i++){
        //     cout <<  *(matched_indices + i) << " ";
        // }
        // cout << endl;

        // for(int i = 0; i < nCount; i++){
        //     cout <<  *(matched_distances + i) << " ";
        // }
        // cout << endl;

        // cin.get();   

        for(int i = 0; i < nCount; i++){
            Vector3f source_point (source->points[i].x, source->points[i].y, source->points[i].z);
            source_cloud_matrix.col(i) = total_rotation * source_point + total_translation;

            int matched_index = *(matched_indices + i);
            Vector3f matched_point (reference->points[matched_index].x, reference->points[matched_index].y, reference->points[matched_index].z);
            matched_cloud_matrix.col(i) = matched_point;
        }


        Vector3f source_center_of_mass = source_cloud_matrix.rowwise().mean();
        // cout<<source_center_of_mass<<endl;
        source_cloud_matrix = source_cloud_matrix.colwise() - source_center_of_mass; //TODO: check this math: https://stackoverflow.com/questions/42811084/eigen-subtracting-vector-from-matrix-columns
        
        Vector3f matched_center_of_mass = matched_cloud_matrix.rowwise().mean();
        // cout<<matched_center_of_mass<<endl;
        matched_cloud_matrix = matched_cloud_matrix.colwise() - matched_center_of_mass; //TODO: check this math
        // cout<<"found center of masses"<<endl;

        //compute dxd matrix of covariances W
        Matrix3f covariances = Matrix3f::Zero();
        for(int col = 0; col < source->points.size(); col++){
            covariances = covariances + (source_cloud_matrix.col(col) * matched_cloud_matrix.col(col).transpose());
        }

        // cout<<covariances<<endl;
        // cout<<covariances.rows()<<endl;
        // cout<<covariances.cols()<<endl;

        //compute singular value decomposition U and V
        JacobiSVD<MatrixXf> svd; //this is different from the documentation, likely due to a bug: https://stackoverflow.com/questions/72749955/unable-to-compile-the-example-for-eigen-svd
        svd.compute(covariances, ComputeThinU | ComputeThinV);
        // cout<<"found U and V"<<endl;

        //compute rotation and translation
        Matrix3f rotation = svd.matrixU() * (svd.matrixV().transpose());
        Vector3f translation = source_center_of_mass - rotation * matched_center_of_mass;

        // cout<< "rotation: " << rotation << endl;
        // cout<< "translation: " << translation << endl;
        // cout<< "source centroid: " << total_rotation *  source_center_of_mass + total_translation << endl;
        // cout<< "matched centroid: " << matched_center_of_mass << endl;
        total_rotation *= rotation.transpose();
        total_translation -= translation;

        // //create transform
        // Matrix4f transform = Matrix4f::Identity();
        // transform.block<3,3>(0,0) = rotation.transpose();
        // transform.block<3,1>(0,3) = -translation;
        // transformPointCloud (*source, *source, transform);

        // cudaDeviceSynchronize();
        // cudaFree(search);
        // cudaFree(index);
        // cudaFree(output);
        // cudaFree(distance);
        // cudaFree(selectedCount);
        cudaFree(matched_indices);
        cudaFree(matched_distances);
    }
    cudaFree(cuda_source);
    cudaFree(cuda_reference);
    cudaStreamDestroy(stream);
    //write result as pcd
    t2 = chrono::steady_clock::now();

    time_span = chrono::duration_cast<chrono::duration<double, ratio<1, 1000>>>(t2 - t1);
    cout << "ICP-GPU costs : " << time_span.count() << " ms."<< endl;

    Matrix4f transform = Matrix4f::Identity();
    transform.block<3,3>(0,0) = total_rotation;
    transform.block<3,1>(0,3) = total_translation;
    transformPointCloud (*source, *source, transform);

    *source += *reference;
    pcl::io::savePCDFileASCII ("result.pcd", *source);

    cout<<"saved ICP-GPU output to result.pcd"<<endl;
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