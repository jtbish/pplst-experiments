#!/bin/bash
# variable params
fl_grid_size=8
fl_slip_prob=0.3
fl_iod_strat_base_train="frozen"
fl_iod_strat_base_test="frozen"

# static / calced params
declare -A pplst_indiv_sizes=( [4]=7 [8]=21 [12]=42 )
pplst_indiv_size="${pplst_indiv_sizes[$fl_grid_size]}"
# 16x indiv sizes
declare -A pplst_pop_sizes=( [4]=112 [8]=336 [12]=672 )
pplst_pop_size="${pplst_pop_sizes[$fl_grid_size]}"
pplst_num_gens=250
pplst_tourn_size=3
pplst_p_cross=0.7
pplst_p_cross_swap=0.5
pplst_p_mut=0.01
pplst_num_reinf_rollouts=50
pplst_weight_I_min=0
pplst_weight_I_max=0
pplst_x_nought=10
pplst_eta=0.1
gamma=0.95

for pplst_seed in {60..89}; do
   echo sbatch pplst_frozen_lake.sh \
        "$fl_grid_size" \
        "$fl_slip_prob" \
        "$fl_iod_strat_base_train" \
        "$fl_iod_strat_base_test" \
        "$pplst_num_gens" \
        "$pplst_seed" \
        "$pplst_pop_size" \
        "$pplst_indiv_size" \
        "$pplst_tourn_size" \
        "$pplst_p_cross" \
        "$pplst_p_cross_swap" \
        "$pplst_p_mut" \
        "$pplst_num_reinf_rollouts" \
        "$pplst_weight_I_min" \
        "$pplst_weight_I_max" \
        "$pplst_x_nought" \
        "$pplst_eta" \
        "$gamma"
done
