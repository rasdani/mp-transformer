#!/bin/bash

#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=16
#SBATCH --gpus=0
#SBATCH --out=generate_toy_data.out
#SBATCH --mail-user=$EMAIL
#SBATCH --mail-type=END,FAIL

module purge 
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate $HOME/mambaforge/envs/mp-transformer

python mp_transformer/utils/generate_toy_data.py
