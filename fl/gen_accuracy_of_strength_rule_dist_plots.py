import functools
import glob
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
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

_KDE_VLINE_STYLE = "dashed"
_KDE_VLINE_COLOR = "black"
_KDE_VLINE_WIDTH = 0.5

_SCATTER_MARKERSIZE = 8
_SCATTER_ALPHA = 0.25

_RULE_ERR_TYPES = ("RMSE", "MNAE")
_RULE_ERR_TYPE = "MNAE"
assert _RULE_ERR_TYPE in _RULE_ERR_TYPES

_MNAE_EPSILON = 1e-5

n = np.nan
_SHORTEST_DISTS_MATS = {
    4: (np.asarray([[6, 5, 4, 5], [5, n, 3, n], [4, 3, 2, n], [n, 2, 1,
                                                               n]]).T),
    8: (np.asarray([[14, 13, 12, 11, 10, 9, 8, 7],
                    [13, 12, 11, 10, 9, 8, 7, 6], [12, 11, 10, n, 8, 7, 6, 5],
                    [11, 10, 9, 8, 7, n, 5, 4], [12, 11, 10, n, 6, 5, 4, 3],
                    [13, n, n, 6, 5, 4, n, 2], [12, n, 8, 7, n, 3, n, 1],
                    [11, 10, 9, n, 3, 2, 1, n]]).T),
    12: (np.asarray([[22, 21, 22, n, 18, 17, 16, 15, 14, n, 12, 11],
                     [n, 20, 21, 22, n, n, 15, 14, 13, n, 11, 10],
                     [20, 19, n, 23, n, 15, 14, 13, 12, 11, 10, 9],
                     [19, 18, 17, n, 15, n, 13, 12, n, 10, 9, 8],
                     [18, n, 16, 15, 14, 13, n, 11, n, 9, 8, 7],
                     [17, 16, 15, 14, 13, 12, 11, 10, 9, n, 7, 6],
                     [16, n, 14, 13, 12, n, 10, 9, 8, n, 6, 5],
                     [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, n, 4],
                     [16, n, 14, 13, 12, n, 8, 7, 6, 5, 4, 3],
                     [17, 16, 15, 14, 13, n, 9, 8, 7, n, 3, 2],
                     [18, 17, 16, 15, n, n, 10, 9, n, n, 2, 1],
                     [19, 18, 17, n, 13, 12, 11, 10, n, 2, 1, n]]).T)
}

assert _SHORTEST_DISTS_MATS[4].shape == (4, 4)
assert _SHORTEST_DISTS_MATS[8].shape == (8, 8)
assert _SHORTEST_DISTS_MATS[12].shape == (12, 12)


def main():
    gs = int(sys.argv[1])
    sp = float(sys.argv[2])

    env = make_fl(grid_size=gs,
                  slip_prob=sp,
                  iod_strat=_FL_IOD_STRAT,
                  seed=_FL_SEED)
    nonterm_obs_tuples = [tuple(obs) for obs in env.nonterminal_states]

    shortest_dists_mat = _SHORTEST_DISTS_MATS[gs]
    for x in range(0, gs):
        for y in range(0, gs):
            val = shortest_dists_mat[x][y]
            if (x, y) in nonterm_obs_tuples:
                assert not np.isnan(val)
            else:
                assert np.isnan(val)

    if sp == 0:
        glob_expr = f"./frozen_redux/detrm/gs_{gs}/6*"
    else:
        glob_expr = f"./frozen_redux/stoca/gs_{gs}_sp_{sp}/6*"
    exp_dirs = glob.glob(glob_expr)
    assert len(exp_dirs) == _EXPECTED_NUM_EXP_DIRS

    # first, gen KDE of rule dists
    transdir_rule_dists = []
    for exp_dir in exp_dirs:
        with open(f"{exp_dir}/best_indiv_history.pkl", "rb") as fp:
            hist = pickle.load(fp)
        indiv = hist[_NUM_GENS]

        for rule in indiv.rules:
            nonterm_covered_obs_tuples = _find_nonterm_covered_obs_tuples(
                rule.condition, nonterm_obs_tuples)
            rule_dist = _calc_rule_dist(nonterm_covered_obs_tuples,
                                        shortest_dists_mat)
            if rule_dist is not None:
                transdir_rule_dists.append(rule_dist)

    plt.figure()
    sns.kdeplot(transdir_rule_dists)
    plt.xlabel("Rule goal distance")

    plt.axvline(x=0,
                linestyle=_KDE_VLINE_STYLE,
                color=_KDE_VLINE_COLOR,
                linewidth=_KDE_VLINE_WIDTH)
    non_nan_shortest_dists = np.asarray(
        [e for e in shortest_dists_mat.flatten() if not np.isnan(e)])
    max_shortest_dist = np.max(non_nan_shortest_dists)
    plt.axvline(x=max_shortest_dist,
                linestyle=_KDE_VLINE_STYLE,
                color=_KDE_VLINE_COLOR,
                linewidth=_KDE_VLINE_WIDTH)

    plt.savefig(f"accuracy_of_strength_gs_{gs}_sp_{sp}_rule_dist_kde_plot.pdf",
                bbox_inches="tight")

    # second gen a scatterplot of rule w_rmses by dist
    transdir_rule_w_errors = []
    for exp_dir in exp_dirs:
        with open(f"{exp_dir}/best_indiv_history.pkl", "rb") as fp:
            hist = pickle.load(fp)
        indiv = hist[_NUM_GENS]

        with open(f"{exp_dir}/best_final_indiv_true_payoffs.pkl", "rb") as fp:
            true_payoffs = pickle.load(fp)

        for (k, v) in true_payoffs.items():
            print(k, v)

        x_nought = indiv.x_nought
        rule_errors = [
            _calc_rule_errors(rule, true_payoffs, nonterm_obs_tuples, x_nought)
            for rule in indiv.rules
        ]
        for e in rule_errors:
            if e is not None:
                w_error = e[0]
                transdir_rule_w_errors.append(w_error)

    assert len(transdir_rule_w_errors) == len(transdir_rule_dists)

    print(len(transdir_rule_w_errors))
    print(len([e for e in transdir_rule_w_errors if e > 1]))

    plt.figure()
    plt.scatter(x=transdir_rule_dists,
                y=transdir_rule_w_errors,
                s=_SCATTER_MARKERSIZE,
                alpha=_SCATTER_ALPHA)
    plt.xlabel("Rule goal dist")
    plt.ylabel(f"Rule w {_RULE_ERR_TYPE}")

    plt.ylim(bottom=_MIN_ERROR, top=2)

    plt.savefig(
        f"accuracy_of_strength_gs_{gs}_sp_{sp}_rule_dist_w_{_RULE_ERR_TYPE}_scatter_plot.pdf",
        bbox_inches="tight")


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


def _calc_rule_dist(nonterm_covered_obs_tuples, shortest_dists_mat):
    if len(nonterm_covered_obs_tuples) == 0:
        return None
    else:
        return np.mean([
            shortest_dists_mat[x][y] for (x, y) in nonterm_covered_obs_tuples
        ])


def _calc_rule_errors(rule, true_payoffs, nonterm_obs_tuples, x_nought):
    nonterm_covered_obs_tuples = _find_nonterm_covered_obs_tuples(
        rule.condition, nonterm_obs_tuples)

    if len(nonterm_covered_obs_tuples) == 0:

        # it is possible that the rule only covers terminal state(s), so can't
        # calc anything for it
        return None

    else:

        action = rule.action
        aug_obs = functools.partial(augment_obs, x_nought=x_nought)

        # TODO clean up
        if _RULE_ERR_TYPE == "RMSE":

            squared_diffs = []
            for obs in nonterm_covered_obs_tuples:
                true_payoff = true_payoffs[(obs, action)]
                estimated_payoff = rule.prediction(aug_obs(obs))

                squared_diff = (true_payoff - estimated_payoff)**2
                squared_diffs.append(squared_diff)

            assert len(squared_diffs) > 0

            w_rmse = np.sqrt(np.mean(squared_diffs))
            stdev_err = np.abs(w_rmse - rule.payoff_stdev)

            return (w_rmse, stdev_err)

        elif _RULE_ERR_TYPE == "MNAE":

            normed_abs_diffs = []
            for obs in nonterm_covered_obs_tuples:
                true_payoff = true_payoffs[(obs, action)]
                estimated_payoff = rule.prediction(aug_obs(obs))

                abs_diff = np.abs(true_payoff - estimated_payoff)
                norm = np.abs(true_payoff)
                if norm == 0:
                    norm += _MNAE_EPSILON
                normed_abs_diff = (abs_diff / norm)

                normed_abs_diffs.append(normed_abs_diff)

            assert len(normed_abs_diffs) > 0

            w_mnae = np.mean(normed_abs_diffs)
            dummy = 420
            stdev_err = dummy

            return (w_mnae, stdev_err)


if __name__ == "__main__":
    main()
