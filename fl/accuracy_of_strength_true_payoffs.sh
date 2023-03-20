#!/bin/bash
#SBATCH --partition=cpu

source ~/virtualenvs/lcs/bin/activate
python3 accuracy_of_strength_true_payoffs.py $1 $2
