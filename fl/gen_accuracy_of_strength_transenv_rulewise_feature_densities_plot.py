import glob
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from pplst.util import augment_obs
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

_GSS_ZIP = (8, 12)
_SPS_ZIP = (0.1, 0.5)
assert len(_GSS_ZIP) == len(_SPS_ZIP)

_COLORS = ("tab:blue", "tab:red")
assert len(_COLORS) == len(_SPS_ZIP)

_FIGSIZE = (10, 6)

_EXPECTED_NUM_EXP_DIRS = 30
# both not important
_FL_IOD_STRAT = "top_left"
_FL_SEED = 0

_NUM_GENS = 250

_NUM_OBS_DIMS = 2

_MIN_GENERALITY = 1
_MIN_GOAL_DIST = 1
_MIN_ERROR = 0

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

_REL_ERR_EPSILON = 1e-9

_DF_COLS = ("generality", "goal_dist", "strength_error", "w_error",
            "stdev_error")

_HIST_ALPHA = 0.25

_MAX_GENERALITIES = {4: 11, 8: 53, 12: 114}
_MAX_GOAL_DISTS = {4: 6, 8: 14, 12: 23}

_XLABEL_NON_BRACKETS = {
    "generality": "Generality",
    "goal_dist": "Goal dist.",
    "strength_error": "Strength error",
    "w_error": (r"$\vec{w}$" + " error"),
    "stdev_error": "Stdev. error"
}

_XLABEL_BRACKETS = {
    ("abs", "q_pi", "strength_error"): r"$Q^\pi\ \mathrm{MAE}$",
    ("abs", "q_star", "strength_error"): r"$Q^*\ \mathrm{MAE}$",
    ("rel", "q_pi", "strength_error"): r"$Q^\pi\ \mathrm{MRE}$",
    ("rel", "q_star", "strength_error"): r"$Q^*\ \mathrm{MRE}$",
    ("abs", "q_pi", "stdev_error"): r"$Q^\pi\ \mathrm{abs.}$",
    ("abs", "q_star", "stdev_error"): r"$Q^*\ \mathrm{abs.}$",
    ("rel", "q_pi", "stdev_error"): r"$Q^\pi\ \mathrm{rel.}$",
    ("rel", "q_star", "stdev_error"): r"$Q^*\ \mathrm{rel.}$"
}
_XLABEL_BRACKETS[("abs", "q_pi",
                  "w_error")] = _XLABEL_BRACKETS[("abs", "q_pi",
                                                  "strength_error")]
_XLABEL_BRACKETS[("abs", "q_star",
                  "w_error")] = _XLABEL_BRACKETS[("abs", "q_star",
                                                  "strength_error")]
_XLABEL_BRACKETS[("rel", "q_pi",
                  "w_error")] = _XLABEL_BRACKETS[("rel", "q_pi",
                                                  "strength_error")]
_XLABEL_BRACKETS[("rel", "q_star",
                  "w_error")] = _XLABEL_BRACKETS[("rel", "q_star",
                                                  "strength_error")]

_BOUND_LINEWIDTH = 1
_BOUND_LINESTYLE = "dashed"
_BOUND_LINECOLOR = "black"


def main():
    err_type = sys.argv[1]
    assert err_type in _ERR_TYPES
    q_ref_type = sys.argv[2]
    assert q_ref_type in _Q_REF_TYPES

    # calc the feature matrices for all envs specified
    dfs = {}
    for (gs, sp) in zip(_GSS_ZIP, _SPS_ZIP):
        print(gs, sp)

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

        # first, compute rule dists and generalities across all exp dirs
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
                if rule_generality >= _MIN_GENERALITY:
                    transdir_rule_generalities.append(rule_generality)

        assert len(transdir_rule_dists) == len(transdir_rule_generalities)

        # step 1.5: calc q_dicts to refer against when calcing rule errors for
        # each exp dir
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

            # just copy Q^* dict needed number of times since same for all exp
            # dirs
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

        # Step 2: compute errors of rule components: i) strength, ii) w vec,
        # iii) stdev
        transdir_rule_strength_errors = []
        transdir_rule_w_errors = []
        transdir_rule_stdev_errors = []

        for (exp_dir, q_dict) in zip(exp_dirs, q_dicts):

            print(exp_dir)

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

                    rule_w_error = _calc_rule_w_error(
                        rule, nonterm_covered_obs_tuples, q_dict, x_nought,
                        err_type)
                    transdir_rule_w_errors.append(rule_w_error)

                    rule_stdev_error = _calc_rule_stdev_error(
                        rule, nonterm_covered_obs_tuples, q_dict, x_nought,
                        err_type)
                    transdir_rule_stdev_errors.append(rule_stdev_error)

        assert len(transdir_rule_strength_errors) == \
            len(transdir_rule_generalities)
        assert len(transdir_rule_strength_errors) == len(
            transdir_rule_w_errors)
        assert len(transdir_rule_w_errors) == len(transdir_rule_stdev_errors)

        df = pd.DataFrame(columns=_DF_COLS)
        df["generality"] = transdir_rule_generalities
        df["goal_dist"] = transdir_rule_dists
        df["strength_error"] = transdir_rule_strength_errors
        df["w_error"] = transdir_rule_w_errors
        df["stdev_error"] = transdir_rule_stdev_errors

        print(df)

        dfs[(gs, sp)] = df

    # now do the density plots
    # thanks: https://stackoverflow.com/questions/48744165/uneven-subplot-in-python
    # (accepted answer)
    fig = plt.figure(figsize=_FIGSIZE)

    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=3)
    ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
    ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)

    axs = (ax1, ax2, ax3, ax4, ax5)

    for (col, ax) in zip(_DF_COLS, axs):
        for (env_idx, (gs, sp,
                       color)) in enumerate(zip(_GSS_ZIP, _SPS_ZIP, _COLORS)):
            data = dfs[(gs, sp)][col]
            print(col, gs, sp, len(data))
            is_last_env = (env_idx == (len(_GSS_ZIP) - 1))

            if col == "generality":
                # hist
                sns.histplot(data=data,
                             stat="percent",
                             element="step",
                             alpha=_HIST_ALPHA,
                             ax=ax,
                             color=color,
                             legend=None)

                if is_last_env:
                    # put legend in first ax
                    legend_handles = [
                        Line2D(xdata=[0, 1],
                               ydata=[0, 0],
                               linewidth=1.5,
                               color=color) for color in _COLORS
                    ]
                    legend_labels = [
                        f"({gs}, {sp})"
                        for (gs, sp) in zip(_GSS_ZIP, _SPS_ZIP)
                    ]
                    ax.legend(legend_handles,
                              legend_labels,
                              loc="upper center")

                    xticks = [
                        _MIN_GENERALITY, 20, 40, 60, 80, 100,
                        max(_MAX_GENERALITIES.values())
                    ]
                    ax.set_xticks(xticks)

                    # add boundary lines
                    ax.axvline(_MIN_GENERALITY,
                               linewidth=_BOUND_LINEWIDTH,
                               linestyle=_BOUND_LINESTYLE,
                               color=_BOUND_LINECOLOR)
                    ax.axvline(max(_MAX_GENERALITIES.values()),
                               linewidth=_BOUND_LINEWIDTH,
                               linestyle=_BOUND_LINESTYLE,
                               color=_BOUND_LINECOLOR)

            else:
                # kde
                sns.kdeplot(data=data, ax=ax, color=color, legend=None)

                if is_last_env:
                    # add boundary lines
                    if col == "goal_dist":
                        ax.axvline(_MIN_GOAL_DIST,
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)
                        ax.axvline(max(_MAX_GOAL_DISTS.values()),
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)

                        xticks = [
                            _MIN_GOAL_DIST, 5, 10, 15, 20,
                            max(_MAX_GOAL_DISTS.values())
                        ]
                        ax.set_xticks(xticks)
                    else:
                        ax.axvline(_MIN_ERROR,
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)

            if is_last_env:
                if col in ("generality", "goal_dist"):
                    xlabel = _XLABEL_NON_BRACKETS[col]
                else:
                    xlabel_non_brackets = _XLABEL_NON_BRACKETS[col]
                    #xlabel_brackets = _XLABEL_BRACKETS[(err_type, q_ref_type,
                    #                                    col)]
                    xlabel_brackets = f"{err_type}."
                    xlabel = f"{xlabel_non_brackets} ({xlabel_brackets})"
                ax.set_xlabel(xlabel)

    fig.tight_layout()

    plt.savefig(
        f"accuracy_of_strength_transenv_rulewise_feature_densities_{err_type}_{q_ref_type}_plot.pdf"
    )

    sys.exit(1)

    # bonus plots: x, y scatterplots
    for (gs, sp) in zip(_GSS_ZIP, _SPS_ZIP):
        fig, axs = plt.subplots(nrows=3, ncols=2)

        for (row_idx,
             y_col) in enumerate(["strength_error", "w_error", "stdev_error"]):
            for (col_idx, x_col) in enumerate(["generality", "goal_dist"]):
                x = dfs[(gs, sp)][x_col]
                y = dfs[(gs, sp)][y_col]
                ax = axs[row_idx][col_idx]

                ax.scatter(x, y, s=4)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)

        plt.savefig(
            f"accuracy_of_strength_gs_{gs}_sp_{sp}_rulewise_xy_scatters_{err_type}_{q_ref_type}_plot.pdf",
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
