#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --ntasks=1
#SBATCH --mem=50000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chakraba@mila.quebec
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
###########################

for lg in de es fr it pt en
do
    python -u preprocess.py --pivot-lang $lg
done
