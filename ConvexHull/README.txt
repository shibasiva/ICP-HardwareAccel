compile: clang++ convexhull.cpp -I ./pcl/include/ -L ./pcl/lib/ -lpcl_common -lpcl_io -lpcl_features -lpcl_filters -lpcl_registration -lpcl_segmentation -lpcl_surface -lpcl_kdtree -lpcl_octree -lpcl_search -lpcl_sample_consensus -o convexhull -std=c++14

usage: ./convexhull PCLfile.pcd