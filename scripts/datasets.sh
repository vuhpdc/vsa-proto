#!/bin/bash

: ${PROJECT_DIR:="./"}

[ -e "${PROJECT_DIR}/datasets" ] || mkdir -p "${PROJECT_DIR}/datasets"

# COCO 2017 val images
if [ ! -e "${PROJECT_DIR}/datasets/coco/val2017/images" ];
then 
  mkdir -p "${PROJECT_DIR}/datasets/coco/val2017/images"
  wget http://images.cocodataset.org/zips/val2017.zip
  unzip -j val2017.zip -d "${PROJECT_DIR}/datasets/coco/val2017/images"
  rm val2017.zip
fi

# COCO 2017 val annotations
if [ ! -e "${PROJECT_DIR}/datasets/coco/val2017/annotations" ];
then 
  mkdir -p  "${PROJECT_DIR}/datasets/coco/val2017/annotations"
  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -j annotations_trainval2017.zip -d "${PROJECT_DIR}/datasets/coco/val2017/annotations"
  rm annotations_trainval2017.zip
fi

# create image list
ls -d "${PROJECT_DIR}"/datasets/coco/val2017/images/* > "${PROJECT_DIR}"/datasets/coco/val2017/image_list.txt
  
# Download PKUMMD video
python3 ${PROJECT_DIR}/scripts/google_drive.py 1czaFDx_UVX4MNfXSQbMKLEOn2q_qjtxq "${PROJECT_DIR}/datasets/pkummd/0200-M.avi"