#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=250g
#SBATCH -t 11-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=pv_estimation
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=31-40
cd /proj/characklab/flooddata/NC/multiple_events/analysis
module purge
module load anaconda/2023.03
conda activate /proj/characklab/flooddata/NC/conda_environments/envs/flood_v1
python3.11 property_value_estimation.py
