#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/surface/convex_hull.h>
#include <iostream>
int main (int argc, char ** argv)
{
    
  if(argc!=2){
    std::cout<<"usage: ./convexhull [pcd reference]"<<std::endl;
    return 0;
  }
  //load reference file
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile (argv[1], *cloudPoints);

  //compute ConvexHull algorithm on reference points
  pcl::PointCloud<pcl::PointXYZ>::Ptr convexhull (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ConvexHull<pcl::PointXYZ> chull;
  chull.setInputCloud (cloudPoints);
  chull.reconstruct (*convexhull);

  
  pcl::io::savePCDFileASCII ("newref.pcd", *convexhull);

  return 0;
}





