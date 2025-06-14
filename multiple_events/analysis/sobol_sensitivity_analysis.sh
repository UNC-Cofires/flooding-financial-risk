#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32g
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=sobol_indicies
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=0-44%10
cd /proj/characklab/flooddata/NC/multiple_events/analysis
module purge
module load anaconda/2023.03
export PYTHONWARNINGS="ignore"
conda activate /proj/characklab/flooddata/NC/conda_environments/envs/flood_v1
python3.11 sobol_sensitivity_analysis.py
