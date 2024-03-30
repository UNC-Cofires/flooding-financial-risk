#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=40g
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=borrower_sim
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=63,77,66,23,70,68,32,8,73,97,30,9,41,64,51,95,15,53,7,81,6,93,27,25,47,39,45,46,36,88,82
cd /proj/characklab/flooddata/NC/multiple_events/analysis
module purge
module load anaconda/2023.03
export PYTHONWARNINGS="ignore"
conda activate /proj/characklab/flooddata/NC/conda_environments/envs/flood_v1
python3.11 borrower_simulation.py
