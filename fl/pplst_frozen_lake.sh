#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4

source ~/virtualenvs/ppl/bin/activate
python3 pplst_frozen_lake.py \
    --experiment-name="$SLURM_JOB_ID" \
    --fl-grid-size="$1" \
    --fl-slip-prob="$2" \
    --fl-iod-strat-base-train="$3" \
    --fl-iod-strat-base-test="$4" \
    --pplst-num-gens="$5" \
    --pplst-seed="$6" \
    --pplst-pop-size="$7" \
    --pplst-indiv-size="$8" \
    --pplst-tourn-size="$9" \
    --pplst-p-cross="${10}" \
    --pplst-p-cross-swap="${11}" \
    --pplst-p-mut="${12}" \
    --pplst-num-reinf-rollouts="${13}" \
    --pplst-weight-i-min="${14}" \
    --pplst-weight-i-max="${15}" \
    --pplst-x-nought="${16}" \
    --pplst-eta="${17}" \
    --gamma="${18}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
#mv "${SLURM_JOB_ID}.prof" "${SLURM_JOB_ID}/"
