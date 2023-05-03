


import pandas as pd
import numpy as np
import sys

import open3d as o3d


def read_points(f):
    # reads Semantic3D .txt file f into a pandas dataframe
    col_names = ['x', 'y', 'z', 'i', 'r', 'g', 'b']
    col_dtype = {'x': np.float32, 'y': np.float32, 'z': np.float32, 'i': np.int32,
                  'r': np.uint8, 'g': np.uint8, 'b': np.uint8}
    try:
        return pd.read_csv(f, names=col_names, dtype=col_dtype, delim_whitespace=True)
    except:
        print("Could not find filename in directory")
        return
    

def read_labels(f):
    # reads Semantic3D .labels file f into a pandas dataframe
    return pd.read_csv(f, header=None)[0].values

def main ():
    if(len(sys.argv)!=2):
        print("usage: python parse.py semantic3D.txt")
        return

    
    file_name= sys.argv[1]
    print("Filename: " +file_name)
    
    panda_points=read_points(file_name)
    
    print("Loading panda dataframe")

    print(panda_points)

    points = panda_points[['x', 'y', 'z']].values
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    # save point cloud to pcd file 
    name = sys.argv[1].split(".txt")[0]
    o3d.io.write_point_cloud(name + ".pcd", pcd)


main()








