#!/bin/bash

export RDYRO_NB_NODES=10
export RDYRO_CPUS_PER_NODE=8
export RDYRO_MEM_PER_NODE=40G

export RDYRO_NB_RUNS=90

export RDYRO_PROGRAM_FILE="../exps/auto_tuning/main.py"
export RDYRO_PROGRAM_ARGS=""

sbatch \
  --mem=$RDYRO_MEM_PER_NODE \
  -N $RDYRO_NB_NODES -c $RDYRO_CPUS_PER_NODE job.sh

  #-p gpu -G $RDYRO_NB_NODES --gpus-per-node 1 -C "GPU_BRD:GEFORCE"
