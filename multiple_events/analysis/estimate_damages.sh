#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=140g
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=0-6

cd /proj/characklab/flooddata/NC/multiple_events/analysis
module purge
module load anaconda/2023.03
conda activate /proj/characklab/flooddata/NC/conda_environments/envs/flood_v1
python3.11 estimate_damages.py
