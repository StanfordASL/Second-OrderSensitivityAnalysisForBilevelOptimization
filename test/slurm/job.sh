#!/bin/bash
#SBATCH --output=out_%j.txt
#SBATCH --time=0-10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rdyro@stanford.edu
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=4
#SBATCH --cpus-per-task=2
##SBATCH --nodes=10


#source $VENV_HOME/bin/activate

program_file="../auto_tuning/mnist_vanilla.py"

date

nb_runs=40

wait # barrier #################################################################

for i in $(seq 0 $nb_runs); do
  srun -p gpu -N 1 -n 1 -c 1 -G 1 python3 $program_file $i &
done

wait # barrier #################################################################

date
