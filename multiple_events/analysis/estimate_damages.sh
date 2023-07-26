#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -t 24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=kieranf@email.unc.edu
#SBATCH --array=0-6

cd /proj/characklab/flooddata/NC/multiple_events/analysis
/proj/characklab/flooddata/NC/conda_environments/envs/flood_v1/bin/python estimate_damages.py
