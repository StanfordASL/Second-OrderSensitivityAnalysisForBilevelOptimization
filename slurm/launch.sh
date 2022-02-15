#!/bin/bash

export CUSTOM_NB_NODES=10
export CUSTOM_CPUS_PER_NODE=8
export CUSTOM_MEM_PER_NODE=40G
export CUSTOM_NB_RUNS=90
export CUSTOM_USE_GPU=0

export CUSTOM_PROGRAM_FILE="../exps/auto_tuning/main.py"
export CUSTOM_PROGRAM_ARGS=""


if [ $CUSTOM_USE_GPU -eq 1 ]; then
  sbatch \
    -p gpu -G $CUSTOM_NB_NODES --gpus-per-node 1 -C "GPU_BRD:GEFORCE" \
    --mem=$CUSTOM_MEM_PER_NODE \
    -N $CUSTOM_NB_NODES -c $CUSTOM_CPUS_PER_NODE job.sh
else
  sbatch \
    --mem=$CUSTOM_MEM_PER_NODE \
    -N $CUSTOM_NB_NODES -c $CUSTOM_CPUS_PER_NODE job.sh
fi
