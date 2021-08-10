#!/bin/bash

nb_nodes=10

#sbatch -p gpu -G $nb_nodes --gpus-per-node 1 -N $nb_nodes job.sh
sbatch -N $nb_nodes --cpus-per-task 4 job.sh
