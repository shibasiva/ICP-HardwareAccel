#!/bin/sh
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 [name of source (ex: bunny1)] [name of ref (ex: bunny)] [folder location (test/point_clouds)]" >&2
  exit 1
fi

# https://stackoverflow.com/questions/16959337/usr-bin-time-format-output-elapsed-time-in-milliseconds
ts=$(date +%s%N)
cd ../../../
test/ply_test/build/pcd_to_ply "t" "$3/$1.pcd"  "$3/$2.pcd" 
Fast-Robust-ICP/build/FRICP "$3/$1.ply" "$3/$2.ply" "Fast-Robust-ICP/data/res/"
tt=$((($(date +%s%N) - $ts)/1000)) ; echo "Time taken: $tt microseconds"

Fast-Robust-ICP/compare_point_clouds/build/compare "$3/$1.pcd"  "$3/$2.pcd" "Fast-Robust-ICP/data/res/m3trans.txt"