#!/bin/bash
#SBATCH --output=out_%j.txt
#SBATCH --time=0-10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rdyro@stanford.edu

source $VENV_HOME/bin/activate

program_file="../exps/auto_tuning/main.py"

date

nb_runs=150

wait # barrier #################################################################

for i in $(seq 0 $nb_runs); do
  #srun -p gpu -N 1 -n 1 -c 4 -G 1 python3 $program_file eval_slurm $i &
  srun -N 1 -n 1 -c 4 python3 $program_file $i &
done

wait # barrier #################################################################

date
