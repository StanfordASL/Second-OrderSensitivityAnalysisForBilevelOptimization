#!/bin/bash

export RDYRO_NB_NODES=10
export RDYRO_CPUS_PER_NODE=1

export RDYRO_NB_RUNS=47

export RDYRO_PROGRAM_FILE="../exps/hessian_error/mnist_vanilla.py"
export RDYRO_PROGRAM_ARGS="eval"

sbatch \
  -p gpu -G $RDYRO_NB_NODES --gpus-per-node 1 -C "GPU_BRD:GEFORCE" \
  -N $RDYRO_NB_NODES -c $RDYRO_CPUS_PER_NODE job.sh
