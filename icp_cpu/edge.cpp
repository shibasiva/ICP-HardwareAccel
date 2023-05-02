#include "edge.h"

void edge_detection(pcl::PointCloud<pcl::PointXYZ>::Ptr reference, pcl::PointCloud<pcl::PointXYZ>::Ptr edgePoints, int k, double lambda)
{
    for(int i=0; i<reference->points.size(); i++){
        std::vector<int> indices(k);
        std::vector<float> sqrDistances(k);
        
        //calculate nearestKNeighbors
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(reference); 
        
        //|V_i| nearest neighbors
        kdtree.nearestKSearch(reference->points[i], k, indices, sqrDistances); 

        //Calculate centroid
        Eigen::Vector3d centroid = Eigen::Vector3d(reference->points[i].x, reference->points[i].y, reference->points[i].z);
        
        double resolution = sqrt(sqrDistances[k-1]);
        //summing closest neighbors to form centroid
        for(int j=0; j<k; j++){
            centroid += Eigen::Vector3d(reference->points[indices[j]].x, reference->points[indices[j]].y, reference->points[indices[j]].z);
        }

        //Centroid = s(1/|V_i|) 
        centroid = centroid/(k+1);
        
        //shift == ‖𝐶𝑖 − 𝑝𝑖‖
        double shift = (centroid - Eigen::Vector3d(reference->points[i].x, reference->points[i].y, reference->points[i].z)).norm();
        
        //if ‖𝐶𝑖 − 𝑝𝑖‖ > 𝜆 ∙ 𝑍𝑖 -> found edge
        if(shift > lambda * resolution){
            edgePoints->push_back(reference->points[i]);
        }
    }
}
