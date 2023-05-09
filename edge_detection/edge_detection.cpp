#include "edge.h"


void edge_detection(PointCloud<PointXYZ>::Ptr reference, PointCloud<PointXYZ>::Ptr edgePoints, int k, double lambda, string name)
{      
    vector<string> edge_points;
    // cout<<"starting"<<endl;
    for(int i=0; i<reference->points.size(); i++){
        cout<<"points left: " << reference->points.size() - i << endl;
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
            // edgePoints->push_back(reference->points[i]);
            edge_points.push_back(to_string(i));
        }
        // cout<<"pass done"<<endl;
    }

    //https://stackoverflow.com/questions/6406356/how-to-write-vector-values-to-a-file
    std::ofstream output_file(name);

    std::ostream_iterator<std::string> output_iterator(output_file, "\n");  
    std::copy(std::begin(edge_points), std::end(edge_points), output_iterator);
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
        
        string name = argv[1];
        name = name.substr(0, name.find(".pcd"));
        string postfix =  "_edges.txt";
        name = name + postfix;
        // io::savePCDFileASCII (name, *edges);

        edge_detection(reference, edges, reference->points.size()/40, 0.5, name);

        
        cout<<"saved edges output to result.txt " << name <<endl;
        return 0;
    }
}