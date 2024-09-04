#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=30g
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=all
#SBATCH --job-name=borrower_sim
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=0-99%50
cd /proj/characklab/flooddata/NC/multiple_events/analysis
module purge
module load anaconda/2023.03
export PYTHONWARNINGS="ignore"
conda activate /proj/characklab/flooddata/NC/conda_environments/envs/flood_v1

replicate_number=1
damage_cost_multiplier=1
property_value_multiplier=1
repair_rate_multiplier=1

python3.11 borrower_simulation.py $replicate_number $damage_cost_multiplier $property_value_multiplier $repair_rate_multiplier
