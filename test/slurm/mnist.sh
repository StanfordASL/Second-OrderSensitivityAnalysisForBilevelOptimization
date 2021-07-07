#!/bin/bash

#SBATCH --output=out_%j.txt
#SBATCH --time=0-10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rdyro@stanford.edu

source $VENV_HOME/bin/activate

program_file="../auto_tuning/mnist_vanilla.py"

date

wait # barrier #################################################################

srun -p gpu -N 1 -n 1 -c 4 -G 1 python3 $program_file train

wait # barrier #################################################################

date
