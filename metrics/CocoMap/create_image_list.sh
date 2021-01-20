#!/bin/sh
if [ "$#" -ne 1 ]; then
    echo "provide image folder as argument"
else
  ls -d "$PWD"/$1* > $1/image_list.txt
fi

