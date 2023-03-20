import glob
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pplst.util import augment_obs
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

_EXPECTED_NUM_EXP_DIRS = 30
# both not important
_FL_IOD_STRAT = "top_left"
_FL_SEED = 0

_NUM_GENS = 250

_NUM_OBS_DIMS = 2

_MIN_ERROR = 0

_SCATTER_MARKERSIZE = 10
_SCATTER_ALPHA = 0.66
_SCATTER_CMAP = "plasma"

_JITTER_SPREAD = 0.33

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

_ERR_TYPES = ("abs", "rel")
_Q_REF_TYPES = ("q_pi", "q_star")

_REL_ERR_EPSILON = 1e-6

_FIGSIZE = (12, 6)


def main():
    gs = int(sys.argv[1])
    sp = float(sys.argv[2])
    err_type = sys.argv[3]
    assert err_type in _ERR_TYPES

    env = make_fl(grid_size=gs,
                  slip_prob=sp,
                  iod_strat=_FL_IOD_STRAT,
                  seed=_FL_SEED)
    nonterm_obs_tuples = [tuple(obs) for obs in env.nonterminal_states]

    # validate structure of shortest dists mat
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

    # setup global df
    df = pd.DataFrame(columns=[
        "goal_dist", "generality", "q_pi:strength_error", "q_pi:w_error",
        "q_pi:stdev_error", "q_star:strength_error", "q_star:w_error",
        "q_star:stdev_error"
    ])

    # first, compute rule dists and generalities across all exp dirs
    # this is the same for both q_ref types
    transdir_rule_dists = []
    transdir_rule_generalities = []
    for exp_dir in exp_dirs:
        with open(f"{exp_dir}/best_indiv_history.pkl", "rb") as fp:
            hist = pickle.load(fp)
        indiv = hist[_NUM_GENS]

        for rule in indiv.rules:
            nonterm_covered_obs_tuples = _find_nonterm_covered_obs_tuples(
                rule.condition, nonterm_obs_tuples)
            rule_dist = _calc_rule_dist(nonterm_covered_obs_tuples,
                                        shortest_dists_mat)
            rule_generality = len(nonterm_covered_obs_tuples)

            if rule_dist is not None:
                transdir_rule_dists.append(rule_dist)
            if rule_generality > 0:
                transdir_rule_generalities.append(rule_generality)

    assert len(transdir_rule_dists) == len(transdir_rule_generalities)
    df["goal_dist"] = transdir_rule_dists
    df["generality"] = transdir_rule_generalities

    # second, compute all 3 error metrics for both q_ref types and store in df
    for q_ref_type in _Q_REF_TYPES:

        # calc q_dicts to refer against when calcing rule errors for each
        # exp dir
        q_dicts = None
        if q_ref_type == "q_pi":

            # each exp dir has own Q^pi to compare to
            q_dicts = []
            for exp_dir in exp_dirs:
                with open(f"{exp_dir}/best_final_indiv_true_payoffs.pkl",
                          "rb") as fp:
                    true_payoffs = pickle.load(fp)
                q_dict = true_payoffs
                q_dicts.append(q_dict)

        elif q_ref_type == "q_star":

            # just copy Q^* dict needed number of times since same for all
            # exp dirs
            q_npy_path = \
                f"/home/Staff/uqjbish3/value-iteration/FrozenLake{gs}x{gs}-v0_gamma_0.95_slip_prob_{sp:.2f}_Qstar.npy"
            q_npy = np.load(q_npy_path)

            q_dict = {}
            for x in range(0, gs):
                for y in range(0, gs):
                    for a in env.action_space:
                        # remember that x is col idx, y is row idx for q_npy
                        q_dict[((x, y), a)] = q_npy[y, x, a]

            q_dicts = [q_dict for _ in range(len(exp_dirs))]

        else:
            assert False

        assert q_dicts is not None
        assert len(q_dicts) == len(exp_dirs)

        # Compute errors of rule components: i) strength, ii) w vec, iii)
        # stdev
        transdir_rule_strength_errors = []
        transdir_rule_w_errors = []
        transdir_rule_stdev_errors = []

        for (exp_dir, q_dict) in zip(exp_dirs, q_dicts):

            print(q_ref_type, exp_dir)

            with open(f"{exp_dir}/best_indiv_history.pkl", "rb") as fp:
                hist = pickle.load(fp)
            indiv = hist[_NUM_GENS]

            x_nought = indiv.x_nought

            for rule in indiv.rules:
                nonterm_covered_obs_tuples = _find_nonterm_covered_obs_tuples(
                    rule.condition, nonterm_obs_tuples)

                can_calc_errors = (len(nonterm_covered_obs_tuples) > 0)
                if can_calc_errors:

                    rule_strength_error = _calc_rule_strength_error(
                        rule, nonterm_covered_obs_tuples, q_dict, x_nought,
                        err_type)
                    transdir_rule_strength_errors.append(rule_strength_error)

                    rule_w_error = \
                        _calc_rule_w_error(rule, nonterm_covered_obs_tuples,
                                           q_dict, x_nought, err_type)
                    transdir_rule_w_errors.append(rule_w_error)

                    rule_stdev_error = _calc_rule_stdev_error(
                        rule, nonterm_covered_obs_tuples, q_dict, x_nought,
                        err_type)
                    transdir_rule_stdev_errors.append(rule_stdev_error)

        assert len(transdir_rule_strength_errors) == \
            len(transdir_rule_generalities)
        assert len(transdir_rule_strength_errors) == \
            len(transdir_rule_w_errors)
        assert len(transdir_rule_w_errors) == len(transdir_rule_stdev_errors)

        df[f"{q_ref_type}:strength_error"] = transdir_rule_strength_errors
        df[f"{q_ref_type}:w_error"] = transdir_rule_w_errors
        df[f"{q_ref_type}:stdev_error"] = transdir_rule_stdev_errors

    print(df)

    # now do the plot

    # share the generality axis jitter noise + make reproducible
    np.random.seed(int(gs + 10 * sp))
    generality_jitter_noise = np.random.uniform(
        low=(-1 * _JITTER_SPREAD),
        high=_JITTER_SPREAD,
        size=len(transdir_rule_generalities))

    # setup global fig.
    nrows = 2
    ncols = 3
    fig, axs = plt.subplots(nrows,
                            ncols,
                            sharex=True,
                            sharey=True,
                            figsize=_FIGSIZE)

    all_errors_flat = list(((df.iloc[:, 2:]).to_numpy()).flatten())
    sorted_all_errors_flat = sorted(all_errors_flat)

    found_non_zero_min_error = False
    non_zero_min_error = None
    idx = 0
    while not found_non_zero_min_error:
        elem = sorted_all_errors_flat[idx]
        if elem != 0:
            non_zero_min_error = elem
            found_non_zero_min_error = True
        else:
            idx += 1

    assert non_zero_min_error is not None
    min_error = non_zero_min_error

    max_error = sorted_all_errors_flat[-1]

    print(f"Min error = {min_error}")
    print(f"Max error = {max_error}")
    norm = matplotlib.colors.LogNorm(vmin=min_error, vmax=max_error)

    for (row_idx, q_ref_type) in enumerate(_Q_REF_TYPES):
        for (col_idx, attr) in enumerate(["strength", "w", "stdev"]):
            ax = axs[row_idx, col_idx]

            sm = ax.scatter(x=df["goal_dist"],
                            y=(df["generality"] + generality_jitter_noise),
                            c=df[f"{q_ref_type}:{attr}_error"],
                            cmap=_SCATTER_CMAP,
                            norm=norm,
                            s=_SCATTER_MARKERSIZE,
                            alpha=_SCATTER_ALPHA)
            ax.set_title(f"{q_ref_type}:{attr}")

            if (row_idx == (nrows - 1)):
                ax.set_xlabel("Rule goal dist.")

            if (col_idx == 0):
                ax.set_ylabel("Rule generality")

    cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    fig.colorbar(sm, cax=cbar_ax, alpha=1)
    plt.savefig(
        f"accuracy_of_strength_gs_{gs}_sp_{sp}_rulewise_error_plot_{err_type}.pdf",
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


def _calc_rule_strength_error(rule, nonterm_covered_obs_tuples, q_dict,
                              x_nought, err_type):
    errors = []
    for obs in nonterm_covered_obs_tuples:
        actual_q_val = q_dict[(obs, rule.action)]

        aug_obs = augment_obs(obs, x_nought)
        # FULL strength eqn: w - sqrt(var)
        pred_strength = rule.strength(aug_obs)

        errors.append(
            _calc_error(err_type,
                        actual_val=actual_q_val,
                        pred_val=pred_strength))

    strength_error = np.mean(errors)
    return strength_error


def _calc_rule_w_error(rule, nonterm_covered_obs_tuples, q_dict, x_nought,
                       err_type):
    errors = []
    for obs in nonterm_covered_obs_tuples:
        actual_q_val = q_dict[(obs, rule.action)]

        aug_obs = augment_obs(obs, x_nought)
        # NOT full strength eqn: just w part
        pred_q_val = rule.prediction(aug_obs)

        errors.append(
            _calc_error(err_type, actual_val=actual_q_val,
                        pred_val=pred_q_val))

    w_error = np.mean(errors)
    return w_error


def _calc_rule_stdev_error(rule, nonterm_covered_obs_tuples, q_dict, x_nought,
                           err_type):
    squared_diffs = []
    for obs in nonterm_covered_obs_tuples:
        actual_q_val = q_dict[(obs, rule.action)]

        aug_obs = augment_obs(obs, x_nought)
        # NOT full strength eqn: just w part
        pred_q_val = rule.prediction(aug_obs)

        squared_diff = ((actual_q_val - pred_q_val)**2)
        squared_diffs.append(squared_diff)

    actual_stdev = np.sqrt(np.mean(squared_diffs))  # RMSE a.k.a. stdev
    pred_stdev = rule.payoff_stdev

    stdev_error = _calc_error(err_type,
                              actual_val=actual_stdev,
                              pred_val=pred_stdev)
    return stdev_error


def _calc_error(err_type, actual_val, pred_val):
    if err_type == "abs":

        return np.abs(actual_val - pred_val)

    elif err_type == "rel":

        abs_diff = np.abs(actual_val - pred_val)
        norm = np.max([np.abs(actual_val), np.abs(pred_val)])
        if norm == 0:
            norm += _REL_ERR_EPSILON
        return (abs_diff / norm)

    else:
        assert False


if __name__ == "__main__":
    main()
