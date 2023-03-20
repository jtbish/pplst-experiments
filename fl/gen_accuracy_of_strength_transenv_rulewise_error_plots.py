import glob
import pickle
import sys

import matplotlib
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
assert len(_COLORS) == len(_GSS_ZIP)

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

_BOUND_LINEWIDTH = 1
_BOUND_LINESTYLE = "dashed"
_BOUND_LINECOLOR = "black"

_PLOT_TRIPLES = [("generality", "goal_dist", "strength_error"),
                 ("generality", "goal_dist", "w_error"),
                 ("generality", "goal_dist", "stdev_error"),
                 ("w_error", "stdev_error", "strength_error")]

_SCATTER_FIGSIZE = (9, 4)
_KDE_FIGSIZE = (10.5, 3.5)

# For (8, 0.1) env
_GEN_BOUND = 46
_GD_BOUND = 13.5


def main():
    err_type = sys.argv[1]
    assert err_type in _ERR_TYPES
    q_ref_type = sys.argv[2]
    assert q_ref_type in _Q_REF_TYPES

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

        df = pd.DataFrame(columns=[
            "goal_dist", "generality", "strength_error", "w_error",
            "stdev_error"
        ])
        df["goal_dist"] = transdir_rule_dists
        df["generality"] = transdir_rule_generalities
        df["strength_error"] = transdir_rule_strength_errors
        df["w_error"] = transdir_rule_w_errors
        df["stdev_error"] = transdir_rule_stdev_errors

        print(df)

        dfs[(gs, sp)] = df

    for (idx, (x_col, y_col, c_col)) in enumerate(_PLOT_TRIPLES):
        nrows = 1
        ncols = len(_GSS_ZIP)
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=(ncols + 1),
            figsize=_SCATTER_FIGSIZE,
            gridspec_kw={"width_ratios": ([1] * ncols + [0.05])})

        cax = axs[-1]
        axs = axs[0:-1]

        # figure out vmin and vmax for colorbar
        cmin = None
        cmax = None
        for (gs, sp) in zip(_GSS_ZIP, _SPS_ZIP):
            env_df = dfs[(gs, sp)]
            env_merged_df = _merge_df_on_cols(env_df, x_col, y_col)

            min_ = env_merged_df[c_col].min()
            if cmin is None or min_ < cmin:
                cmin = min_
            max_ = env_merged_df[c_col].max()
            if cmax is None or max_ > cmax:
                cmax = max_

        assert cmin is not None
        assert cmax is not None

        # do the plots with shared cbar
        norm = matplotlib.colors.LogNorm(vmin=cmin, vmax=cmax)

        for (ax_idx, (gs, sp)) in enumerate(zip(_GSS_ZIP, _SPS_ZIP)):
            ax = axs[ax_idx]

            env_df = dfs[(gs, sp)]
            env_merged_df = _merge_df_on_cols(env_df, x_col, y_col)

            x = env_merged_df[x_col]
            y = env_merged_df[y_col]
            c = env_merged_df[c_col]
            sm = ax.scatter(x=x,
                            y=y,
                            c=c,
                            cmap=_SCATTER_CMAP,
                            norm=norm,
                            s=_SCATTER_MARKERSIZE,
                            alpha=_SCATTER_ALPHA)

            ax.set_title(f"({gs}, {sp})")

        # after plotting the data for all envs, figure out the axes limits
        xmin = min([ax.get_xlim()[0] for ax in axs])
        xmax = max([ax.get_xlim()[1] for ax in axs])
        ymin = min([ax.get_ylim()[0] for ax in axs])
        ymax = max([ax.get_ylim()[1] for ax in axs])

        # sync all the axes limits
        for ax in axs:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            #ax.set_aspect((xmax - xmin) / (ymax - ymin))

        # embellishments: ax labels and boundary lines
        if (x_col == "generality" and y_col == "goal_dist"):
            max_gs = max(_GSS_ZIP)
            gen_ticks = _make_gen_ticks(max_gs)
            gd_ticks = _make_gd_ticks(max_gs)

            for ax in axs:
                ax.set_xticks(gen_ticks)
                ax.set_yticks(gd_ticks)

                # MAGIC NUMBERS
                gen_bound = _GEN_BOUND
                gd_bound = _GD_BOUND

                ax.plot((gen_bound, gen_bound), (ymin, gd_bound),
                        linestyle=_BOUND_LINESTYLE,
                        linewidth=_BOUND_LINEWIDTH,
                        color=_BOUND_LINECOLOR)
                ax.plot((xmin, gen_bound), (gd_bound, gd_bound),
                        linestyle=_BOUND_LINESTYLE,
                        linewidth=_BOUND_LINEWIDTH,
                        color=_BOUND_LINECOLOR)

                ax.set_xlabel("Rule generality")
            axs[0].set_ylabel("Rule goal dist.")

        elif (x_col == "w_error" and y_col == "stdev_error"):

            for ax in axs:
                xlabel_no_brackets = "Rule " + r"$\vec{w}$" + " error"
                #xlabel_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type,
                #                                        "w")]
                xlabel_brackets = f"{err_type}."
                xlabel = f"{xlabel_no_brackets} ({xlabel_brackets})"
                ax.set_xlabel(xlabel)

            ylabel_no_brackets = "Rule stdev. error"
            ylabel_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type,
                                                    "stdev")]
            ylabel_brackets = f"{err_type}."
            ylabel = f"{ylabel_no_brackets} ({ylabel_brackets})"
            axs[0].set_ylabel(ylabel)

        else:
            assert False

        # since no sharey manually kill the ticklabels
        axs[-1].set_yticklabels([])

        if c_col == "strength_error":
            cbar_label_no_brackets = "Rule strength error"
            #cbar_label_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type,
            #                                            "strength")]
        elif c_col == "w_error":
            cbar_label_no_brackets = "Rule " + r"$\vec{w}$" + " error"
            #cbar_label_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type,
            #                                            "w")]
        elif c_col == "stdev_error":
            cbar_label_no_brackets = "Rule stdev. error"
            #cbar_label_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type,
            #                                            "stdev")]
        cbar_label_brackets = f"{err_type}."
        cbar_label = f"{cbar_label_no_brackets} ({cbar_label_brackets})"

        plt.colorbar(sm, label=cbar_label, cax=cax)
        plt.tight_layout()
        plt.savefig(
            f"accuracy_of_strength_transenv_rulewise_{x_col}_{y_col}_{c_col}_{err_type}_{q_ref_type}_plot.pdf",
            bbox_inches="tight")

    # bonus plot: kdes
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=_KDE_FIGSIZE)
    for (col, ax) in zip(["strength_error", "w_error", "stdev_error"], axs):
        for (gs, sp, color) in zip(_GSS_ZIP, _SPS_ZIP, _COLORS):
            env_df = dfs[(gs, sp)]

            if (gs, sp) == (8, 0.1):

                # whole dataset
                sns.kdeplot(data=env_df[col], color=color, ax=ax)

            elif (gs, sp) == (12, 0.5):

                # two kdes
                low_df = env_df.loc[(env_df["generality"] <= _GEN_BOUND)
                                    & (env_df["goal_dist"] <= _GD_BOUND)]
                high_df = env_df.loc[(env_df["generality"] > _GEN_BOUND)
                                     | (env_df["goal_dist"] > _GD_BOUND)]
                print(len(low_df))
                print(len(high_df))
                assert (len(low_df) + len(high_df)) == len(env_df)

                sns.kdeplot(data=low_df[col],
                            color=color,
                            linestyle="dashed",
                            ax=ax)
                sns.kdeplot(data=high_df[col],
                            color=color,
                            linestyle="dotted",
                            ax=ax)

                # add legend on first subplot
                if col == "strength_error":

                    linestyles = ["solid", "dashed", "dotted"]
                    colors = ["tab:blue", "tab:red", "tab:red"]

                    legend_handles = [
                        Line2D(xdata=[0, 0.66],
                               ydata=[0, 0],
                               linewidth=1.5,
                               linestyle=linestyle,
                               color=color)
                        for (linestyle, color) in zip(linestyles, colors)
                    ]

                    low_frac = (len(low_df) / len(env_df))
                    high_frac = (len(high_df) / len(env_df))
                    low_frac_str = (f"{low_frac:.2f}"[2:] + "%")
                    high_frac_str = (f"{high_frac:.2f}"[2:] + "%")
                    legend_labels = [
                        "(8, 0.1)", f"(12, 0.5) in: {low_frac_str}",
                        f"(12, 0.5) out: {high_frac_str}"
                    ]

                    ax.legend(legend_handles,
                              legend_labels,
                              loc="upper right",
                              fontsize=6.5)

            else:
                assert False

        # embellishments
        ax.axvline(_MIN_ERROR,
                   linestyle=_BOUND_LINESTYLE,
                   linewidth=_BOUND_LINEWIDTH,
                   color=_BOUND_LINECOLOR)

        if col == "strength_error":
            xlabel_no_brackets = "Strength error"
            #xlabel_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type,
            #                                        "strength")]
        elif col == "w_error":
            xlabel_no_brackets = r"$\vec{w}$" + " error"
            #xlabel_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type, "w")]
        elif col == "stdev_error":
            xlabel_no_brackets = "Stdev. error"
            #xlabel_brackets = _CBAR_LABEL_BRACKETS[(err_type, q_ref_type,
            #                                        "stdev")]
        else:
            assert False
        xlabel_brackets = f"{err_type}."
        xlabel = f"{xlabel_no_brackets} ({xlabel_brackets})"
        ax.set_xlabel(xlabel)

        if col == "stdev_error":
            # remove negative tick
            xticks = ax.get_xticks()
            ax.set_xticks(xticks[1:])

    fig.tight_layout()
    plt.savefig(
        f"accuracy_of_strength_transenv_rulewise_error_densities_{err_type}_{q_ref_type}_plot.pdf",
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
