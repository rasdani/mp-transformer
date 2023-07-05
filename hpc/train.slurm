#!/bin/bash

#SBATCH --partition=short_gpu
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --out=train.out
#SBATCH --mail-user=aurasd@staff.uni-marburg.de
#SBATCH --mail-type=END,FAIL

module purge 
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate $HOME/mambaforge/envs/mp-transformer

python mp_transformer/train.py
#python mp_transformer/train.py --no-log
