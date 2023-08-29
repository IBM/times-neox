#!/bin/bash

# Runs the "345M" parameter model

# asynio flags
export LDFLAGS="$LDFLAGS -L/usr/lib64/"
export CFLAGS="$CFLAGS -I/usr/include/"
# c++ libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vgurev/.conda/envs/GPT/x86_64-conda-linux-gnu/lib/
export PATH=/data/vgurev/.conda/envs/GPT/bin/:$PATH

#use mpirun, not pytorch luncher
export MPI=TRUE

GPUS_PER_NODE=2
NNODES=1
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

python ./deepy.py generate-times.py 49M.yml

