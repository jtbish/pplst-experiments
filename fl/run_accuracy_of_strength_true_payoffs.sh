#!/bin/bash

gss=( 4 8 12 )
sps=( 0 0.1 0.3 0.5 )

for gs in "${gss[@]}"; do
    for sp in "${sps[@]}"; do
        echo sbatch accuracy_of_strength_true_payoffs.sh "$gs" "$sp"
    done
done
