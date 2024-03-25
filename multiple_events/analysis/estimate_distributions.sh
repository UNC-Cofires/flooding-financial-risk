#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -t 4-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=kieranf@email.unc.edu
cd /proj/characklab/flooddata/NC/multiple_events/analysis
module purge
module load anaconda/2023.03
conda activate /proj/characklab/flooddata/NC/conda_environments/envs/flood_v1
python3.11 estimate_distributions.py