import glob
import math
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from pplst.util import augment_obs
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

_EXPECTED_NUM_EXP_DIRS = 30
# both not important
_FL_IOD_STRAT = "top_left"
_FL_SEED = 0

_NUM_GENS = 250

_NUM_OBS_DIMS = 2

_MIN_GENERALITY = 1
_MIN_GOAL_DIST = 1
_MAX_GENERALITIES = {4: 11, 8: 53, 12: 114}
_MAX_GOAL_DISTS = {4: 6, 8: 14, 12: 23}

_MIN_ERROR = 0

_SCATTER_MARKERSIZE = 15
_SCATTER_ALPHA = 0.66
_SCATTER_CMAP = "plasma"

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

_CBAR_LABEL_BRACKETS = {
    ("abs", "q_pi", "strength"): r"$Q^\pi\ \mathrm{MAE}$",
    ("abs", "q_star", "strength"): r"$Q^*\ \mathrm{MAE}$",
    ("rel", "q_pi", "strength"): r"$Q^\pi\ \mathrm{MRE}$",
    ("rel", "q_star", "strength"): r"$Q^*\ \mathrm{MRE}$",
    ("abs", "q_pi", "stdev"): r"$Q^\pi\ \mathrm{abs.}$",
    ("abs", "q_star", "stdev"): r"$Q^*\ \mathrm{abs.}$",
    ("rel", "q_pi", "stdev"): r"$Q^\pi\ \mathrm{rel.}$",
    ("rel", "q_star", "stdev"): r"$Q^*\ \mathrm{rel.}$"
}
_CBAR_LABEL_BRACKETS[("abs", "q_pi",
                      "w")] = _CBAR_LABEL_BRACKETS[("abs", "q_pi", "strength")]
_CBAR_LABEL_BRACKETS[("abs", "q_star",
                      "w")] = _CBAR_LABEL_BRACKETS[("abs", "q_star",
                                                    "strength")]
_CBAR_LABEL_BRACKETS[("rel", "q_pi",
                      "w")] = _CBAR_LABEL_BRACKETS[("rel", "q_pi", "strength")]
_CBAR_LABEL_BRACKETS[("rel", "q_star",
                      "w")] = _CBAR_LABEL_BRACKETS[("rel", "q_star",
                                                    "strength")]

_HIST_ALPHA = 0.25

_ZOOM_PLOT_FIGSIZE = (9, 3)
_ZOOM_PLOT_SAMPLE_COLOR = "tab:green"
_ZOOM_PLOT_RECT_LINEWIDTH = 1.25

_BOUND_LINEWIDTH = 1
_BOUND_LINESTYLE = "dashed"
_BOUND_LINECOLOR = "black"


def main():
    gs = int(sys.argv[1])
    sp = float(sys.argv[2])
    err_type = sys.argv[3]
    assert err_type in _ERR_TYPES
    q_ref_type = sys.argv[4]
    assert q_ref_type in _Q_REF_TYPES

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

    # step 1.5: calc q_dicts to refer against when calcing rule errors for each
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

        # just copy Q^* dict needed number of times since same for all exp dirs
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

    # Step 2: compute errors of rule components: i) strength, ii) w vec, iii)
    # stdev
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

                rule_w_error = _calc_rule_w_error(rule,
                                                  nonterm_covered_obs_tuples,
                                                  q_dict, x_nought, err_type)
                transdir_rule_w_errors.append(rule_w_error)

                rule_stdev_error = _calc_rule_stdev_error(
                    rule, nonterm_covered_obs_tuples, q_dict, x_nought,
                    err_type)
                transdir_rule_stdev_errors.append(rule_stdev_error)

    assert len(transdir_rule_strength_errors) == \
        len(transdir_rule_generalities)
    assert len(transdir_rule_strength_errors) == len(transdir_rule_w_errors)
    assert len(transdir_rule_w_errors) == len(transdir_rule_stdev_errors)

    df = pd.DataFrame(columns=[
        "goal_dist", "generality", "strength_error", "w_error", "stdev_error"
    ])
    df["goal_dist"] = transdir_rule_dists
    df["generality"] = transdir_rule_generalities
    df["strength_error"] = transdir_rule_strength_errors
    df["w_error"] = transdir_rule_w_errors
    df["stdev_error"] = transdir_rule_stdev_errors

    print(df)

    gd_gen_merged_df = _merge_df_on_cols(df, "goal_dist", "generality")
    print(gd_gen_merged_df)

    # now do the plots

    gen_ticks = _make_gen_ticks(gs)
    gd_ticks = _make_gd_ticks(gs)

    gen_xlim_right = (gen_ticks[-1] + 3)
    gd_ylim_top = (gd_ticks[-1] + 1)

    # i) strength error
    plt.figure()
    plt.scatter(x=gd_gen_merged_df["generality"],
                y=gd_gen_merged_df["goal_dist"],
                c=gd_gen_merged_df["strength_error"],
                cmap=_SCATTER_CMAP,
                norm=matplotlib.colors.LogNorm(),
                s=_SCATTER_MARKERSIZE,
                alpha=_SCATTER_ALPHA)
    #cbar_label_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type,
    #                                            "strength")]
    cbar_label_brackets = f"{err_type}."
    cbar_label = f"Rule strength error ({cbar_label_brackets})"
    plt.colorbar(label=cbar_label)
    plt.xlabel("Rule generality")
    plt.ylabel("Rule goal dist.")
    plt.xticks(gen_ticks)
    plt.yticks(gd_ticks)
    plt.xlim(right=gen_xlim_right)
    plt.ylim(top=gd_ylim_top)
    plt.savefig(
        f"accuracy_of_strength_gs_{gs}_sp_{sp}_rulewise_strength_error_plot_{err_type}_{q_ref_type}.pdf",
        bbox_inches="tight")

    # ii) w error
    plt.figure()
    plt.scatter(x=gd_gen_merged_df["generality"],
                y=gd_gen_merged_df["goal_dist"],
                c=gd_gen_merged_df["w_error"],
                cmap=_SCATTER_CMAP,
                norm=matplotlib.colors.LogNorm(),
                s=_SCATTER_MARKERSIZE,
                alpha=_SCATTER_ALPHA)
    #cbar_label_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type, "w")]
    cbar_label_brackets = f"{err_type}."
    cbar_label = ("Rule " + r"$\vec{w}$" + f" error ({cbar_label_brackets})")
    plt.colorbar(label=cbar_label)
    plt.xlabel("Rule generality")
    plt.ylabel("Rule goal dist.")
    plt.xticks(gen_ticks)
    plt.yticks(gd_ticks)
    plt.xlim(right=gen_xlim_right)
    plt.ylim(top=gd_ylim_top)
    plt.savefig(
        f"accuracy_of_strength_gs_{gs}_sp_{sp}_rulewise_w_error_plot_{err_type}_{q_ref_type}.pdf",
        bbox_inches="tight")

    # iii) stdev error
    plt.figure()
    plt.scatter(x=gd_gen_merged_df["generality"],
                y=gd_gen_merged_df["goal_dist"],
                c=gd_gen_merged_df["stdev_error"],
                cmap=_SCATTER_CMAP,
                norm=matplotlib.colors.LogNorm(),
                s=_SCATTER_MARKERSIZE,
                alpha=_SCATTER_ALPHA)
    #cbar_label_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type, "stdev")]
    cbar_label_brackets = f"{err_type}."
    cbar_label = f"Rule stdev. error ({cbar_label_brackets})"
    plt.colorbar(label=cbar_label)
    plt.xlabel("Rule generality")
    plt.ylabel("Rule goal dist.")
    plt.xticks(gen_ticks)
    plt.yticks(gd_ticks)
    plt.xlim(right=gen_xlim_right)
    plt.ylim(top=gd_ylim_top)
    plt.savefig(
        f"accuracy_of_strength_gs_{gs}_sp_{sp}_rulewise_stdev_error_plot_{err_type}_{q_ref_type}.pdf",
        bbox_inches="tight")

    w_err_stdev_err_merged_df = _merge_df_on_cols(df, "w_error", "stdev_error")
    print(w_err_stdev_err_merged_df)

    # bonus plot:
    # strength error vs. (w error, stdev_error)

    plt.figure()
    plt.scatter(x=w_err_stdev_err_merged_df["w_error"],
                y=w_err_stdev_err_merged_df["stdev_error"],
                c=w_err_stdev_err_merged_df["strength_error"],
                cmap=_SCATTER_CMAP,
                norm=matplotlib.colors.LogNorm(),
                s=_SCATTER_MARKERSIZE,
                alpha=_SCATTER_ALPHA)
    #cbar_label_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type,
    #                                            "strength")]
    cbar_label_brackets = f"{err_type}."
    cbar_label = f"Rule strength error ({cbar_label_brackets})"
    plt.colorbar(label=cbar_label)
    #xlabel_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type, "w")]
    xlabel_brackets = cbar_label_brackets
    plt.xlabel("Rule " + r"$\vec{w}$" + " error " + f"({xlabel_brackets})")
    #ylabel_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type, "stdev")]
    ylabel_brackets = cbar_label_brackets
    plt.ylabel("Rule stdev. error " + f"({ylabel_brackets})")

    plt.savefig(
        f"accuracy_of_strength_gs_{gs}_sp_{sp}_rulewise_tri_error_plot_{err_type}_{q_ref_type}.pdf",
        bbox_inches="tight")
    sys.exit(1)

    if (gs == 8 and sp == 0.1 and err_type == "rel"
            and q_ref_type == "q_star"):
        # xy == bottom left corner
        w_lb = -0.05
        w_ub = 0.2
        stdev_lb = 0.65
        stdev_ub = 1.05
        xy = (w_lb, stdev_lb)
        width = (w_ub - w_lb)
        height = (stdev_ub - stdev_lb)
        plt.gca().add_patch(
            Rectangle(xy=xy,
                      width=width,
                      height=height,
                      edgecolor=_ZOOM_PLOT_SAMPLE_COLOR,
                      facecolor="none",
                      lw=_ZOOM_PLOT_RECT_LINEWIDTH))
        plt.savefig(
            f"accuracy_of_strength_gs_{gs}_sp_{sp}_rulewise_tri_error_plot_{err_type}_{q_ref_type}.pdf",
            bbox_inches="tight")

        # calc zoomed df on original df so that when comparing marginals it is
        # fair
        zoomed_df = df.loc[
            (df["w_error"].between(w_lb, w_ub, inclusive="both"))
            &
            (df["stdev_error"].between(stdev_lb, stdev_ub, inclusive="both"))]
        print(zoomed_df)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=_ZOOM_PLOT_FIGSIZE)
        for (ax, col) in zip(axs,
                             ["strength_error", "generality", "goal_dist"]):
            for (sample_idx, (sample_df, color)) in enumerate(
                    zip([df, zoomed_df], ["tab:blue", "tab:green"])):
                data = sample_df[col]
                print(color, len(data))

                if col == "generality":
                    sns.histplot(data=data,
                                 stat="percent",
                                 element="step",
                                 alpha=_HIST_ALPHA,
                                 color=color,
                                 ax=ax)
                else:
                    sns.kdeplot(data=data, color=color, ax=ax)

                # embellishments
                if (sample_idx == 1):
                    if col == "strength_error":
                        ax.axvline(_MIN_ERROR,
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)
                        ax.set_xlabel("Strength error (" +
                                      r"$Q^*\ \mathrm{MRE}$" + ")")

                    elif col == "generality":
                        xticks = _make_gen_ticks(gs, step=10, start=10)
                        ax.set_xticks(xticks)
                        ax.set_xlabel("Generality")

                        ax.axvline(_MIN_GENERALITY,
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)
                        ax.axvline(_MAX_GENERALITIES[gs],
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)

                    elif col == "goal_dist":
                        xticks = _make_gd_ticks(gs, step=3, start=4)
                        ax.set_xticks(xticks)
                        ax.set_xlabel("Goal dist.")

                        ax.axvline(_MIN_GOAL_DIST,
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)
                        ax.axvline(_MAX_GOAL_DISTS[gs],
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)

                    else:
                        assert False

        fig.tight_layout()

        plt.savefig(
            f"accuracy_of_strength_gs_{gs}_sp_{sp}_rulewise_tri_error_zoom_densities_plot_{err_type}_{q_ref_type}.pdf",
            bbox_inches="tight")

    elif (gs == 12 and sp == 0.5 and err_type == "rel"
          and q_ref_type == "q_star"):
        # xy == bottom left corner
        w_lb = 0.8
        w_ub = 1.2
        stdev_lb = 0.8
        stdev_ub = 1.05
        xy = (w_lb, stdev_lb)
        width = (w_ub - w_lb)
        height = (stdev_ub - stdev_lb)
        plt.gca().add_patch(
            Rectangle(xy=xy,
                      width=width,
                      height=height,
                      edgecolor=_ZOOM_PLOT_SAMPLE_COLOR,
                      facecolor="none",
                      lw=_ZOOM_PLOT_RECT_LINEWIDTH))
        plt.savefig(
            f"accuracy_of_strength_gs_{gs}_sp_{sp}_rulewise_tri_error_plot_{err_type}_{q_ref_type}.pdf",
            bbox_inches="tight")

        zoomed_df = df.loc[
            (df["w_error"].between(w_lb, w_ub, inclusive="both"))
            &
            (df["stdev_error"].between(stdev_lb, stdev_ub, inclusive="both"))]
        print(zoomed_df)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=_ZOOM_PLOT_FIGSIZE)
        for (ax, col) in zip(axs,
                             ["strength_error", "generality", "goal_dist"]):
            for (sample_idx, (sample_df, color)) in enumerate(
                    zip([df, zoomed_df], ["tab:red", "tab:green"])):
                data = sample_df[col]
                print(color, len(data))

                if col == "generality":
                    sns.histplot(data=data,
                                 stat="percent",
                                 element="step",
                                 alpha=_HIST_ALPHA,
                                 color=color,
                                 ax=ax)
                else:
                    sns.kdeplot(data=data, color=color, ax=ax)

                # embellishments
                if (sample_idx == 1):
                    if col == "strength_error":
                        ax.axvline(_MIN_ERROR,
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)
                        ax.set_xlabel("Strength error (" +
                                      r"$Q^*\ \mathrm{MRE}$" + ")")

                    elif col == "generality":
                        xticks = _make_gen_ticks(gs,
                                                 step=20,
                                                 start=25)
                        ax.set_xticks(xticks)
                        ax.set_xlabel("Generality")

                        ax.axvline(_MIN_GENERALITY,
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)
                        ax.axvline(_MAX_GENERALITIES[gs],
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)

                    elif col == "goal_dist":
                        xticks = _make_gd_ticks(gs,
                                                step=4,
                                                start=5)
                        ax.set_xticks(xticks)
                        ax.set_xlabel("Goal dist.")

                        ax.axvline(_MIN_GOAL_DIST,
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)
                        ax.axvline(_MAX_GOAL_DISTS[gs],
                                   linewidth=_BOUND_LINEWIDTH,
                                   linestyle=_BOUND_LINESTYLE,
                                   color=_BOUND_LINECOLOR)

                    else:
                        assert False

        fig.tight_layout()

        plt.savefig(
            f"accuracy_of_strength_gs_{gs}_sp_{sp}_rulewise_tri_error_zoom_densities_plot_{err_type}_{q_ref_type}.pdf",
            bbox_inches="tight")

    else:
        plt.savefig(
            f"accuracy_of_strength_gs_{gs}_sp_{sp}_rulewise_tri_error_plot_{err_type}_{q_ref_type}.pdf",
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


def _merge_df_on_cols(df, col1, col2):
    assert col1 in df.columns
    assert col2 in df.columns

    col1_col2_pair_counts = {}
    for (e1, e2) in zip(df[col1], df[col2]):
        try:
            col1_col2_pair_counts[(e1, e2)] += 1
        except KeyError:
            col1_col2_pair_counts[(e1, e2)] = 1

    assert len(df[col1]) == len(df[col2])
    assert sum(col1_col2_pair_counts.values()) == len(df[col1])

    other_cols = [col for col in df.columns if (col != col1 and col != col2)]
    assert len(other_cols) == (len(df.columns) - 2)

    # put the merged columns at the front
    merged_df = pd.DataFrame(columns=([col1, col2] + other_cols))

    for ((e1, e2), count) in col1_col2_pair_counts.items():

        len_merged_df_before = len(merged_df)

        sub_df = df.loc[(df[col1] == e1) & (df[col2] == e2)]
        assert len(sub_df) == count

        if count == 1:
            # just add the sub_df to the merged_df
            merged_df = merged_df.append(sub_df, ignore_index=True)
        elif count > 1:
            # calc the mean of all the other columns then add it to the merged
            # df for this (e1, e2) pair
            means = []
            for col in other_cols:
                means.append(np.mean(sub_df[col]))
            merged_df.loc[len(merged_df)] = ([e1, e2] + means)
        else:
            assert False

        len_merged_df_after = len(merged_df)
        assert len_merged_df_after == (len_merged_df_before + 1)

    return merged_df


def _make_gen_ticks(gs, step=None, start=None, snip_end=True):
    min_gen = _MIN_GENERALITY
    max_gen = _MAX_GENERALITIES[gs]

    if (step is None and start is None):
        if gs == 8:
            step = 5
            start = 5
        elif gs == 12:
            step = 10
            start = 10
        else:
            raise NotImplementedError

    ticks = ([min_gen] + list(range(start, max_gen, step)) + [max_gen])
    if snip_end:
        if ((ticks[-1] - ticks[-2]) < step):
            del ticks[-2]

    return ticks


def _make_gd_ticks(gs, step=None, start=None, snip_end=True):
    min_gd = _MIN_GOAL_DIST
    max_gd = _MAX_GOAL_DISTS[gs]

    if (step is None and start is None):
        if gs == 8:
            step = 2
            start = 3
        elif gs == 12:
            step = 2
            start = 3
        else:
            raise NotImplementedError

    ticks = ([min_gd] + list(range(start, max_gd, step)) + [max_gd])
    if snip_end:
        if ((ticks[-1] - ticks[-2]) < step):
            del ticks[-2]

    return ticks


if __name__ == "__main__":
    main()
