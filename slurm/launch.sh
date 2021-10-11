#!/bin/bash

export RDYRO_NB_NODES=20
export RDYRO_CPUS_PER_NODE=1
export RDYRO_MEM_PER_NODE=64G

export RDYRO_NB_RUNS=180

export RDYRO_PROGRAM_FILE="../exps/auto_tuning/main/py"
export RDYRO_PROGRAM_ARGS=""

sbatch \
  --mem=$RDYRO_MEM_PER_NODE \
  -N $RDYRO_NB_NODES -c $RDYRO_CPUS_PER_NODE job.sh

  #-p gpu -G $RDYRO_NB_NODES --gpus-per-node 1 -C "GPU_BRD:GEFORCE"
