import open3d as o3d
import sys
if(len(sys.argv) < 2):
     print("Usage: python3 pcd_to_ply.py [a.pcd] [b.pcd] ...")
else:
    for i in range(1, len(sys.argv)):
        pcd = o3d.io.read_point_cloud(sys.argv[i])
        name = sys.argv[i].split(".pcd")[0]
        o3d.io.write_point_cloud(name + ".ply", pcd)