#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=128g
#SBATCH -t 12:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=kieranf@email.unc.edu

cd /proj/characklab/flooddata/NC/data_joining
/proj/characklab/flooddata/NC/conda_environments/envs/flood_v1/bin/python join_data.py > $(date +%F)_output.log 
