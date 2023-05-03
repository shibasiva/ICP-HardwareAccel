#include "edge.h"


void edge_detection(PointCloud<PointXYZ>::Ptr reference, PointCloud<PointXYZ>::Ptr edgePoints, int k, double lambda)
{      
    // cout<<"starting"<<endl;
    for(int i=0; i<reference->points.size(); i++){
        vector<int> indices(k);
        vector<float> sqrDistances(k);
        
        //calculate nearestKNeighbors
        KdTreeFLANN<PointXYZ> kdtree;
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
        // cout<<"pass done"<<endl;
    }
}

int main(int argc, char** argv){
    
    if(argc != 2){
        cout<<"Usage: ./find_edges [pcd reference]"<<endl;
        return 0;
    }
    else{
        PointCloud<PointXYZ>::Ptr edges (new PointCloud<PointXYZ>);
        PointCloud<PointXYZ>::Ptr reference (new PointCloud<PointXYZ>);

        if (io::loadPCDFile<PointXYZ> (argv[1], *reference) == -1){
            string s = argv[1];
            cout<< "Couldn't read file " + s + "\n" << endl;
            return (-1);
        }
        

        edge_detection(reference, edges, 10, 0.2);

        string name = argv[1];
        name = name.substr(0, name.find(".pcd"));
        string postfix =  "_edges.pcd";
        name = name + postfix;
        io::savePCDFileASCII (name, *edges);
        cout<<"saved ICP-CPU output to " << name <<endl;
        return 0;
    }
}