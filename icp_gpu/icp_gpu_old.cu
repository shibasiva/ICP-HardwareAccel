#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <Eigen/Dense>
#include <stdexcept>
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

#include "cuda_runtime.h"
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

tuple<vector<int>, VectorXf, int> RunCUDAOctreeSearch(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference);
vector<PointCloud<PointXYZ>::Ptr> DivideIntoChunks(PointCloud<PointXYZ>::Ptr cloud, int chunkSize);
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

vector<PointCloud<PointXYZ>::Ptr> DivideIntoChunks(PointCloud<PointXYZ>::Ptr cloud, int chunk_size){
    vector<PointCloud<PointXYZ>::Ptr> blocks;

    int division_begin = 0;
    int points_per_division = chunk_size;
    int data_divisions = 0;

    if(cloud->points.size() < chunk_size){ //if less than chunk_size points, just load onto GPU all at once
        data_divisions = 1;
    }
    else{
        data_divisions = (cloud->points.size() / chunk_size) + 1; //handle chunk_size points at a time on the
    }

    // cout<<"chunk size: " << chunk_size<<endl;
    // cout<<"divisions: " << data_divisions<<endl;
    // cout<<"points per division: " << points_per_division<<endl;
    for(int i = 0; i < data_divisions; i++){
        if(i == data_divisions - 1){
            points_per_division = cloud->points.size() - division_begin;
        }
           
        PointCloud<PointXYZ>::Ptr cloud_temp_block(new PointCloud<PointXYZ>);
        for(int j = 0; j < points_per_division; j++){
            cloud_temp_block->points.push_back(cloud->points[division_begin + j]);
        }
        blocks.push_back(cloud_temp_block);
        division_begin += points_per_division;
    }
    return blocks;

}

//following from this code: https://github.com/NVIDIA-AI-IOT/cuPCL/blob/main/cuOctree/main.cpp
tuple<vector<int>, VectorXf, int> RunCUDAOctreeSearch(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference){     
        cudaStream_t stream = NULL;
        cudaStreamCreate ( &stream );
        
        unsigned int nDstCount = source->points.size();
        float *outputData = (float *)source->points.data();

        unsigned int nCount = reference->points.size();
        float *inputData = (float *)reference->points.data();

        // cout<<"nDstCount: " <<nDstCount << endl;
        // cout<<"nCount: " <<nCount << endl;
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
        checkCudaErrors(cudaStreamAttachMemAsync(stream, selectedCount));
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

        int *pointIdxANSearch = index;
        float *pointANSquaredDistance = distance;
        int status = 0;
        *selectedCount = nDstCount;

        cudaDeviceSynchronize();

        status = treeTest.approxNearestSearch(output, pointIdxANSearch, pointANSquaredDistance, selectedCount);

        cudaDeviceSynchronize();

        if(status != 0){
            throw invalid_argument( "CUDA octree failed" );
        }

        vector<int> indices;
        VectorXf distances(*selectedCount);
        int count = *selectedCount;
        for(int i = 0; i < *selectedCount; i++){
            if(*(((unsigned int*)pointIdxANSearch) + i) > 10000000){
                cout<<"bad pointer?"<<endl;
                cout<<*(((unsigned int*)pointIdxANSearch) + i -1) << " " << *(((unsigned int*)pointIdxANSearch) + i) << endl;
                cout<<"distance: " << endl;
                cout << *(((unsigned int*)pointANSquaredDistance) + i) << endl;
                throw invalid_argument( "CUDA octree failed" );
            }
            indices.push_back(*(((unsigned int*)pointIdxANSearch) + i));
            distances(i) = *(((unsigned int*)pointANSquaredDistance) + i);
        }

        cudaFree(search);
        cudaFree(index);
        cudaFree(input);
        cudaFree(output);
        cudaFree(distance);
        cudaFree(selectedCount);
        cudaStreamDestroy(stream);

        return {indices, distances, count};
}

//map the source onto the reference
void ICP(PointCloud<PointXYZ>::Ptr source, PointCloud<PointXYZ>::Ptr reference)
{   
    cout<<"source points: " << source->points.size() << endl;
    cout<<"reference points: " << reference->points.size() <<endl;
    int max_iter = 100; // max iterations
    double convergence_criteria = 0.003;
    // float resolution = 128.0; 
    const int MAX_POINTS_PER_DIVISION = 24000;
    
    vector<PointCloud<PointXYZ>::Ptr> reference_blocks = DivideIntoChunks(reference, MAX_POINTS_PER_DIVISION);
    // for(auto a: reference_blocks){
    //     cout<<"size: " << a->points.size()<<endl;
    // }
    // Matrix3d total_rotation = Matrix3d::Identity();
    // Vector3d total_translation = Vector3d::Zero();

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> time_span =
    std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

    t1 = std::chrono::steady_clock::now();

    //following from this code: https://github.com/NVIDIA-AI-IOT/cuPCL/blob/main/cuOctree/main.cpp
    // cout<<"finished dividing the point clouds"<<endl;

    for (int iter = 0; iter < max_iter; iter++) // iterations
    { 
        cout<<"iter: "<<iter<<endl;
        // cout<<"source cloud size: "<< source->points.size()<<endl;
        MatrixXd source_cloud_matrix(3, source->points.size()); //X
        MatrixXd matched_cloud_matrix(3, source->points.size()); //P
        float rms = 0.0;
        vector<int*> best_indices_container;
        vector<PointCloud<PointXYZ>::Ptr> source_blocks = DivideIntoChunks(source, MAX_POINTS_PER_DIVISION);
        // cout<<"source blocks: " << source_blocks.size()<<endl;
        // cout<<"analyzing blocks"<<endl;
        for(int source_block_num = 0; source_block_num < source_blocks.size(); source_block_num++){
            bool first_pass = true;
            int* best_indices = new int[source_blocks[source_block_num]->points.size()];
            int* best_index_block = new int[source_blocks[source_block_num]->points.size()];
            VectorXf best_distances(source_blocks[source_block_num]->points.size());

            for(int reference_block_num = 0; reference_block_num < reference_blocks.size(); reference_block_num++){
                cout<<"calling cudaOctree"<<endl;
                auto [indices, distances, count] = RunCUDAOctreeSearch(source_blocks[source_block_num], reference_blocks[reference_block_num]);
                // cout<<"cudaOctree done"<<endl;
                for(int c = 0; c < count; c++){
                    // cout<<"got matched index and distance"<<endl;
                    if(first_pass){
                        best_indices[c] = indices[c];
                        best_distances(c) = distances(c);
                        best_index_block[c] = reference_block_num;
                    }
                    else{
                        if(distances(c) < best_distances(c)){
                            best_indices[c] = indices[c];
                            best_distances(c) = distances(c);
                            best_index_block[c] = reference_block_num;
                        }
                    }
                }
                first_pass = false;
                // cout<<"reference block analyzed: " << reference_block_num<<endl;
            }

            for(int index = 0; index < source_blocks[source_block_num]->points.size(); index++){
                // cout<<"block size: " << source_blocks[source_block_num]->points.size() <<  endl;
                // cout<<"matched point matrix size: " << matched_cloud_matrix.cols() << endl;
                // cout<<"starting at: " << source_block_num * points_per_division << endl;
                // cout<<"full_index at: " << source_block_num * points_per_division + index << endl;
                int full_index = source_block_num * MAX_POINTS_PER_DIVISION + index;
                Vector3d source_point (source->points[full_index].x, source->points[full_index].y, source->points[full_index].z);
                source_cloud_matrix.col(full_index) = source_point;

                cout<<"block: " << best_index_block[index] << endl;
                cout<<"index: " << index << " | " << best_indices[index] << endl;
                cout<<"matched index: " << best_index_block[index] * MAX_POINTS_PER_DIVISION + best_indices[index] << endl;
                int matched_index = best_index_block[index] * MAX_POINTS_PER_DIVISION + best_indices[index];
                Vector3d matched_point (reference->points[matched_index].x, reference->points[matched_index].y, reference->points[matched_index].z);
                // cout <<"matched index worked"<<endl;
                matched_cloud_matrix.col(full_index) = matched_point;
            }
            // cout<<"added to matrix"<<endl;
            rms += best_distances.sum()/1e9;
        }
        
        rms /= source->points.size();
        cout<<"rms: " << rms << endl;
        if(rms < convergence_criteria){
            cout<<"final rms: " <<rms<<endl;
            break;
        }
        
        // cout<<"regular ICP now"<<endl;
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
        cout<<"found U and V"<<endl;

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

    t2 = std::chrono::steady_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
    std::cout << "CUDA costs : " << time_span.count() << " ms."<< std::endl;
    // cudaFree(input);
    // cudaStreamDestroy(stream);
    //write result as pcd
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