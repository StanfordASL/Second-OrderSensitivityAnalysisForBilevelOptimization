#!/bin/bash

#sbatch -p gpu -G 4 --gpus-per-node 1 -N 4 job.sh
sbatch -p gpu -G 1 --gpus-per-node 1 -N 1 -c 4 mnist.sh
