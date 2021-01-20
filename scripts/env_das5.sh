#!/bin/bash

# REMOVE default cuda
module rm cuda90/toolkit/9.0.176
module rm nccl/cuda80/2.1.2
module rm gcc

# LOAD MODULES
module load cuda10.0
module load cuDNN/cuda10.0/7.6.4
module load cmake/3.15.4
module load gcc/6.3.0

# set ENV
export CC=gcc
export CXX=g++
