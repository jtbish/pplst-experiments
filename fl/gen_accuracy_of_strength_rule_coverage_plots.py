import functools
import glob
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pplst.util import augment_obs
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

_EXPECTED_NUM_EXP_DIRS = 30
# both not important
_FL_IOD_STRAT = "top_left"
_FL_SEED = 0

_NUM_GENS = 250

_NUM_OBS_DIMS = 2

_MIN_ERROR = 0

_RULESET_SIZES = {4: 7, 8: 21, 12: 42}


def main():
    gs = int(sys.argv[1])
    sp = float(sys.argv[2])

    env = make_fl(grid_size=gs,
                  slip_prob=sp,
                  iod_strat=_FL_IOD_STRAT,
                  seed=_FL_SEED)
    nonterm_obs_tuples = [tuple(obs) for obs in env.nonterminal_states]

    if sp == 0:
        glob_expr = f"./frozen_redux/detrm/gs_{gs}/6*"
    else:
        glob_expr = f"./frozen_redux/stoca/gs_{gs}_sp_{sp}/6*"
    exp_dirs = glob.glob(glob_expr)
    assert len(exp_dirs) == _EXPECTED_NUM_EXP_DIRS

    # first, gen PMF of rule coverage sizes
    transdir_rule_coverage_sizes = []
    for exp_dir in exp_dirs:
        with open(f"{exp_dir}/best_indiv_history.pkl", "rb") as fp:
            hist = pickle.load(fp)
        indiv = hist[_NUM_GENS]

        for rule in indiv.rules:
            nonterm_covered_obs_tuples = _find_nonterm_covered_obs_tuples(
                rule.condition, nonterm_obs_tuples)
            coverage_size = len(nonterm_covered_obs_tuples)
            transdir_rule_coverage_sizes.append(coverage_size)

    assert len(transdir_rule_coverage_sizes) == (_EXPECTED_NUM_EXP_DIRS *
                                                 _RULESET_SIZES[gs])

    print(transdir_rule_coverage_sizes.count(0))
    print(len(transdir_rule_coverage_sizes))
    plt.figure()
    sns.histplot(transdir_rule_coverage_sizes, stat="percent", discrete=True)
    plt.savefig(
        f"accuracy_of_strength_gs_{gs}_sp_{sp}_rule_coverage_pmf_plot.pdf",
        bbox_inches="tight")


    # second gen a scatterplot of rule errors by coverage size


def _find_nonterm_covered_obs_tuples(condition, nonterm_obs_tuples):
    assert len(condition.phenotype) == _NUM_OBS_DIMS
    (x_interval, y_interval) = condition.phenotype
    xs = list(range(x_interval.lower, (x_interval.upper + 1), 1))
    ys = list(range(y_interval.lower, (y_interval.upper + 1), 1))

    covered_obs_tuples = [(x, y) for x in xs for y in ys]
    nonterm_covered_obs_tuples = [
        obs for obs in covered_obs_tuples if obs in nonterm_obs_tuples
    ]
    return nonterm_covered_obs_tuples


if __name__ == "__main__":
    main()
