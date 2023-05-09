#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <fstream>
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
// extern "C"{
//     #include "./cuPCL/cuOctree/lib/cudaOctree.h"
// }

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


using namespace std;
using namespace Eigen;
using namespace pcl;

void ICP(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference, map<int, bool>& edge_points);

//https://github.com/NVIDIA-AI-IOT/cuPCL/blob/main/cuOctree/main.cpp
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
void NearestNeighborSearch(
                           float* source, 
                           float* reference, 
                           float* search_indices,
                           int search_indices_len, 
                           int reference_len, 
                           Matrix3f rotation, 
                           Vector3f translation, 
                           int* matched_indices, 
                           float* matched_distances
                           )
{
    if(search_indices_len <= 0) return;
    
    //grid stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < search_indices_len; i += blockDim.x * gridDim.x) {
        int best_matched_index = 0;
        // if(i == 0){
        //     for(int j = 0; j < search_indices_len; j++){
        //         // printf("search index: %i\n", search_indices[j]);
        //         printf("search index: %i\n", *((unsigned int*)search_indices + j));
        //     }
        // }
        int sp_index = *((unsigned int*)search_indices + i);
        
        float s_x = source[sp_index*4];
        float s_y = source[sp_index*4 + 1];
        float s_z = source[sp_index*4 + 2];

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
void ICP(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference, map<int, bool>& edge_points)
{   
    //following from this code: https://github.com/NVIDIA-AI-IOT/cuPCL/blob/main/cuOctree/main.cpp
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double, std::ratio<1, 1000>> time_span =
    chrono::duration_cast<chrono::duration<double, ratio<1, 1000>>>(t2 - t1);

    int max_iter = 100; // max iterations
    double convergence_criteria = 0.001;
    // float resolution = 128.0; 

    Matrix3f total_rotation = Matrix3f::Identity();
    Vector3f total_translation = Vector3f::Zero();

    t1 = chrono::steady_clock::now();

    int regular_priority = 2;
    int higher_priority = 1;

    cudaStream_t stream = NULL;
    // cudaStreamCreate(&stream);
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, regular_priority);
    cudaStream_t priority_stream = NULL;
    cudaStreamCreateWithPriority(&priority_stream, cudaStreamNonBlocking, higher_priority);
    
    //load data onto GPU
    unsigned int nCount = reference->width * reference->height;
    float *referenceData = (float *)reference->points.data();

    unsigned int nDstCount = source->width * source->height;
    float *sourceData = (float *)source->points.data();

    float *cuda_source = NULL;
    gpuErrchk(cudaMallocManaged(&cuda_source, sizeof(float) * 4 * nCount, cudaMemAttachHost));
    gpuErrchk(cudaStreamAttachMemAsync (stream, cuda_source));
    // gpuErrchk(cudaMemcpyAsync(cuda_source, sourceData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(cuda_source, sourceData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    
    float *cuda_reference = NULL;
    gpuErrchk(cudaMallocManaged(&cuda_reference, sizeof(float) * 4 *nDstCount, cudaMemAttachHost));
    gpuErrchk(cudaStreamAttachMemAsync (stream, cuda_reference));
    // gpuErrchk(cudaMemcpyAsync(cuda_reference, referenceData, sizeof(float) * 4 * nDstCount, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(cuda_reference, referenceData, sizeof(float) * 4 * nDstCount, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    // float resolution = 0.03f;
    vector<int> edge_matched_indices;
    vector<int> nonedge_matched_indices;

    for(int i = 0; i < nDstCount; i++){
        nonedge_matched_indices.push_back(i);
    }

    int blockSize = 1024;
    int numBlocks = (nCount + blockSize - 1) / blockSize;
    // cout<<"block size: "<< blockSize<< endl;
    // cout<<"numBlocks: " << numBlocks << endl;
    float rms_max = 5;
    float rms_min = 0.0005;
    double old_rms = 10000;
    float rms_diff = 0.000001;
    float w = 1;

    for (int iter = 0; iter < max_iter; iter++) // iterations
    { 
        cout<<"iter: "<<iter<<endl;
        MatrixXf source_cloud_matrix(3, source->points.size()); //X
        MatrixXf matched_cloud_matrix(3, source->points.size()); //P

        int num_previous_matched_edges = edge_matched_indices.size();
        int num_previous_matched_nonedges = nonedge_matched_indices.size();
        
        //create two data streams; one for a list of edge matched indices to search for and another for nonedge
        float* previous_matched_edges = NULL;
        float* previous_edge_ptr = (float *)edge_matched_indices.data();
        int *edge_matched_indices_results;
        float *edge_matched_distances_results;

        if(num_previous_matched_edges > 0){
            gpuErrchk(cudaMallocManaged(&previous_matched_edges, sizeof(float) * num_previous_matched_edges, cudaMemAttachHost));
            gpuErrchk(cudaStreamAttachMemAsync (priority_stream, previous_matched_edges));
            gpuErrchk(cudaMemcpyAsync(previous_matched_edges, previous_edge_ptr, sizeof(float) *num_previous_matched_edges, cudaMemcpyHostToDevice, priority_stream));
            gpuErrchk(cudaStreamSynchronize(priority_stream));
           
            gpuErrchk(cudaMallocManaged(&edge_matched_indices_results, sizeof(int) * nCount, cudaMemAttachHost));
            gpuErrchk(cudaStreamAttachMemAsync (priority_stream, edge_matched_indices_results));
            gpuErrchk(cudaMemsetAsync(edge_matched_indices_results, 0, sizeof(unsigned int), priority_stream));
            gpuErrchk(cudaStreamSynchronize(priority_stream));
            
            gpuErrchk(cudaMallocManaged(&edge_matched_distances_results, sizeof(float) * nCount, cudaMemAttachHost));
            gpuErrchk(cudaStreamAttachMemAsync (priority_stream, edge_matched_distances_results));
            gpuErrchk(cudaMemsetAsync(edge_matched_distances_results, 0, sizeof(unsigned int), priority_stream));
            gpuErrchk(cudaStreamSynchronize(priority_stream));
            NearestNeighborSearch<<<numBlocks, blockSize, 0, priority_stream>>>(
                                                                          cuda_source, 
                                                                          cuda_reference, 
                                                                          previous_matched_edges, 
                                                                          num_previous_matched_edges,
                                                                          nDstCount, 
                                                                          total_rotation, 
                                                                          total_translation, 
                                                                          edge_matched_indices_results,
                                                                          edge_matched_distances_results
                                                                          
                                                                          );
        }

        float* previous_matched_nonedges = NULL;
        float* previous_non_edges_ptr = (float *)nonedge_matched_indices.data();
        int *matched_indices_results;
        float *matched_distances_results;

        if(num_previous_matched_nonedges > 0){
            // cout<<"copying over nonedges"<<endl;
            gpuErrchk(cudaMallocManaged(&previous_matched_nonedges, sizeof(float) * num_previous_matched_nonedges, cudaMemAttachHost));
            gpuErrchk(cudaStreamAttachMemAsync (stream, previous_matched_nonedges));
            gpuErrchk(cudaMemcpyAsync(previous_matched_nonedges, previous_non_edges_ptr, sizeof(float) * num_previous_matched_nonedges, cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaStreamSynchronize(stream));

            gpuErrchk(cudaMallocManaged(&matched_indices_results, sizeof(int) * nCount, cudaMemAttachHost));
            gpuErrchk(cudaStreamAttachMemAsync (stream, matched_indices_results));
            gpuErrchk(cudaMemsetAsync(matched_indices_results, 0, sizeof(unsigned int), stream));
            gpuErrchk(cudaStreamSynchronize(stream));
           
            gpuErrchk(cudaMallocManaged(&matched_distances_results, sizeof(float) * nCount, cudaMemAttachHost));
            gpuErrchk(cudaStreamAttachMemAsync (stream, matched_distances_results));
            gpuErrchk(cudaMemsetAsync(matched_distances_results, 0, sizeof(unsigned int), stream));
            gpuErrchk(cudaStreamSynchronize(stream));

            
            NearestNeighborSearch<<<numBlocks, blockSize, 0, stream>>>(
                                                                    cuda_source, 
                                                                    cuda_reference, 
                                                                    previous_matched_nonedges, 
                                                                    num_previous_matched_nonedges,
                                                                    nDstCount, 
                                                                    total_rotation, 
                                                                    total_translation, 
                                                                    matched_indices_results, 
                                                                    matched_distances_results
                                                                    );
        }
        
        gpuErrchk(cudaStreamSynchronize(stream));
        gpuErrchk(cudaStreamSynchronize(priority_stream));
        int num_edge_matched = 0;
        int num_points = 0;
        vector<int> new_edge_matched;
        vector<int> new_nonedge_matched;
        double rms = 0.0;
        // cout<< "nDstCount * w: " << nDstCount * w << endl;
        // cout<< "num previous edges: " << num_previous_matched_edges<<endl;
        // cout<< "num previous nonedges: " << num_previous_matched_nonedges << endl;
        
        bool stop_adding_to_matrix = false;
        for(int i = 0; i < num_previous_matched_edges + num_previous_matched_nonedges; i++){

            //stop adding to pose matrix if we have collected enough matched edge points
            if(num_edge_matched > nDstCount * w){
                stop_adding_to_matrix = true;
            }
            int matched_index = 0;
            int selected_index = 0;
            if(i <  num_previous_matched_edges){
                matched_index = edge_matched_indices_results[i];
                rms += edge_matched_distances_results[i];
                selected_index = edge_matched_indices[i];

                if(edge_points[matched_index]){
                    num_edge_matched += 1;
                    // cout<<"pushing back on edge indices"<<endl;
                    // cout<<edge_matched_indices[i]<<endl;
                    new_edge_matched.push_back(edge_matched_indices[i]);
                }
                else{
                    new_nonedge_matched.push_back(edge_matched_indices[i]);
                }
            }
            else{
                matched_index = matched_indices_results[i - num_previous_matched_edges];
                rms += matched_distances_results[i - num_previous_matched_edges];
                selected_index = nonedge_matched_indices[i - num_previous_matched_edges];
                if(edge_points[matched_index]){
                    num_edge_matched += 1;
                    // cout<<"pushing back on edge indices"<<endl;
                    // cout<<edge_matched_indices[i]<<endl;
                    new_edge_matched.push_back(nonedge_matched_indices[i-  num_previous_matched_edges]);
                }
                else{
                    new_nonedge_matched.push_back(nonedge_matched_indices[i - num_previous_matched_edges]);
                }
            }

            if(!stop_adding_to_matrix){
                Vector3f source_point (source->points[selected_index].x, source->points[selected_index].y, source->points[selected_index].z);
                source_cloud_matrix.col(i) = total_rotation * source_point + total_translation;

                
                Vector3f matched_point (reference->points[matched_index].x, reference->points[matched_index].y, reference->points[matched_index].z);
                matched_cloud_matrix.col(i) = matched_point;
                num_points++;
            }
            
        }

        edge_matched_indices = new_edge_matched;
        nonedge_matched_indices = new_nonedge_matched;

        source_cloud_matrix = source_cloud_matrix(seqN(0,3), seqN(0,num_previous_matched_edges + num_previous_matched_nonedges));
        matched_cloud_matrix = matched_cloud_matrix(seqN(0,3), seqN(0,num_previous_matched_edges + num_previous_matched_nonedges));

        rms = sqrt(rms/num_points);
        cout<<"rms: " <<rms<<endl;
        if(rms < convergence_criteria || abs(old_rms - rms) < rms_diff){
            cout<<"final rms: " <<rms<<endl;
            break;
        }
        
        //set w
        if(rms < rms_min){ //if less than min, set w to a small number of points
            w = 0.001;
        }
        else if(rms < rms_max){
            w = rms/rms_max;
        }
        else{ //rms too large, no confidence
            w = 1;
        }
        
        old_rms = rms;

        // cin.get();   

        Vector3f source_center_of_mass = source_cloud_matrix.rowwise().mean();
        // cout<<source_center_of_mass<<endl;
        source_cloud_matrix = source_cloud_matrix.colwise() - source_center_of_mass; //TODO: check this math: https://stackoverflow.com/questions/42811084/eigen-subtracting-vector-from-matrix-columns
        
        Vector3f matched_center_of_mass = matched_cloud_matrix.rowwise().mean();
        // cout<<matched_center_of_mass<<endl;
        matched_cloud_matrix = matched_cloud_matrix.colwise() - matched_center_of_mass; //TODO: check this math
        // cout<<"found center of masses"<<endl;

        //compute dxd matrix of covariances W
        Matrix3f covariances = Matrix3f::Zero();
        for(int col = 0; col < source_cloud_matrix.cols(); col++){
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
        if(num_previous_matched_nonedges > 0){
            cudaFree(matched_indices_results);
            cudaFree(matched_distances_results);
            cudaFree(previous_matched_nonedges);
        }
        
        if(num_previous_matched_edges > 0){
            cudaFree(edge_matched_indices_results);
            cudaFree(edge_matched_distances_results);
            cudaFree(previous_matched_edges);
        }

        
    }
    cudaFree(cuda_source);
    cudaFree(cuda_reference);
    cudaStreamDestroy(stream);
    t2 = chrono::steady_clock::now();

    time_span = chrono::duration_cast<chrono::duration<double, ratio<1, 1000>>>(t2 - t1);
    cout << "ICP-GPU Priority Scheduler costs : " << time_span.count() << " ms."<< endl;

    //write result as pcd
    Matrix4f transform = Matrix4f::Identity();
    transform.block<3,3>(0,0) = total_rotation;
    transform.block<3,1>(0,3) = total_translation;
    transformPointCloud (*source, *source, transform);

    *source += *reference;
    pcl::io::savePCDFileASCII ("result.pcd", *source);

    cout<<"saved ICP-GPU Priority Scheduler output to result.pcd"<<endl;
}

int main(int argc, char** argv){
    
    if(argc != 4){
        cout<<"Usage: ./icp_gpu_priority [pcd source] [pcd reference] [pcd reference edges.txt]"<<endl;
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
        map<int, bool> edge_points;
        std::ifstream infile(argv[3]);
        int a;
        while (infile >> a)
        {
            edge_points[a] = true;
        }
        // cout<<edge_points[8968]<<endl;
        // cout<<edge_points[7]<<endl;
        // cout<<edge_points[1]<<endl;
        GetInfo();
        ICP(source, reference, edge_points); 
        return 0;
    }
}