#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40GB
#SBATCH --time=48:00:00
#SBATCH --job-name=preprocessing_income_race_LFR_batch1
#SBATCH --output=preprocessing_income_race_LFR_batch1.out

module purge
source $SCRATCH/Experiments-AAAI-24/venv/bin/activate
pip install -r ./requirements.txt
pip install BlackBoxAuditing==0.1.54
pip install git+https://github.com/DataResponsibly/Virny.git@development
pip install --upgrade jupyter ipywidgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix

python ./scripts/preprocessing_intervention_income_GA/Folk_GA_2018_Batch1.py
