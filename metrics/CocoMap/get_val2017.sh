#!/bin/sh
wget -N http://images.cocodataset.org/zips/val2017.zip
unzip -o val2017.zip
rm val2017.zip
cd val2017
ls -d "$PWD"/* > image_list.txt
