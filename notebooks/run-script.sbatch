#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --job-name=postprocessing_law_school_race_batch1
#SBATCH --output=postprocessing_law_school_race_batch1.out

module purge
source $SCRATCH/Experiments-AAAI-24/.venv/bin/activate

python ./scripts/preprocessing_intervention_income_GA/StudentPerformance_Batch1.py