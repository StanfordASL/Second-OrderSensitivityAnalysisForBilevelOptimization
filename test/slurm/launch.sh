#!/bin/bash

nb_gpus=5

sbatch -p gpu -G $nb_gpus --gpus-per-node 1 -N $nb_gpus job.sh
