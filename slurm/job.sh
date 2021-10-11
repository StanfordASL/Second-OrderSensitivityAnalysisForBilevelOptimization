#!/bin/bash
#SBATCH --output=out_%j.txt
#SBATCH --time=0-10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rdyro@stanford.edu

source $VENV_HOME/bin/activate

date

wait # barrier #################################################################

for i in $(seq 0 $(( RDYRO_NB_RUNS - 1 )) ); do
  srun \
    -N 1 -n 1 -c $RDYRO_CPUS_PER_NODE \
    --mem=$RDYRO_MEM_PER_NODE \
    python3 $RDYRO_PROGRAM_FILE $RDYRO_PROGRAM_ARGS $i &

    #-p gpu -G 1 -C "GPU_BRD:GEFORCE" \
done

wait # barrier #################################################################

date
