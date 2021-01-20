#!/bin/bash

# update darknet submodule
git submodule init
git submodule update

cd darknet
# Patch our changes
git apply ../patches/darknet_detector.patch

# Build darknet
./build.sh 

# Download darknet YOLOv4 weights
python3 ../scripts/google_drive.py 1d3fJhDvEqSHu4tIqIpXFRYAAmZBVC_Cc "yolov4.weights"
