#!/bin/bash
#SBATCH --output=out_%j.txt
#SBATCH --time=0-10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rdyro@stanford.edu

source $VENV_HOME/bin/activate

date

wait # barrier #################################################################

for i in $(seq 0 $(( CUSTOM_NB_RUNS - 1 )) ); do
  if [ $CUSTOM_USE_GPU -eq 1 ]; then
    srun \
      -p gpu -G 1 -C "GPU_BRD:GEFORCE" \
      -N 1 -n 1 -c $CUSTOM_CPUS_PER_NODE \
      --mem=$CUSTOM_MEM_PER_NODE \
      python3 $CUSTOM_PROGRAM_FILE $CUSTOM_PROGRAM_ARGS $i &
  else
    srun \
      -N 1 -n 1 -c $CUSTOM_CPUS_PER_NODE \
      --mem=$CUSTOM_MEM_PER_NODE \
      python3 $CUSTOM_PROGRAM_FILE $CUSTOM_PROGRAM_ARGS $i &
  fi

done

wait # barrier #################################################################

date
