import glob
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pplst.util import augment_obs
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

_GSS = (4, 8, 12)
_SPS = (0, 0.1, 0.3, 0.5)

_EXPECTED_NUM_EXP_DIRS = 30
# both not important
_FL_IOD_STRAT = "top_left"
_FL_SEED = 0

_NUM_GENS = 250

_NUM_OBS_DIMS = 2

_FIGSIZE = (8, 14)

_MIN_ERROR = 0

_ERR_TYPES = ("abs", "rel")
_Q_REF_TYPES = ("q_pi", "q_star")

_REL_ERR_EPSILON = 1e-9

_YLABEL_BRACKETS = {"abs": "MAE", "rel": "MRE"}


def main():
    err_type = sys.argv[1]
    assert err_type in _ERR_TYPES

    # first just calc all the errs across all envs for both q_ref types
    q_pi_errors_df = pd.DataFrame(
        columns=[f"({gs}, {sp})" for gs in _GSS for sp in _SPS])
    q_star_errors_df = pd.DataFrame(
        columns=[f"({gs}, {sp})" for gs in _GSS for sp in _SPS])

    for gs in _GSS:
        for sp in _SPS:
            for q_ref_type in _Q_REF_TYPES:
                ruleset_errors = _calc_ruleset_errors(gs, sp, err_type,
                                                      q_ref_type)
                if q_ref_type == "q_pi":
                    q_pi_errors_df[f"({gs}, {sp})"] = ruleset_errors
                elif q_ref_type == "q_star":
                    q_star_errors_df[f"({gs}, {sp})"] = ruleset_errors
                else:
                    assert False

    # thanks: https://stackoverflow.com/questions/44552489/plotting-multiple-boxplots-in-seaborn
    q_pi_errors_df = q_pi_errors_df.assign(P=r"$Q^\pi$")
    q_star_errors_df = q_star_errors_df.assign(P=r"$Q^*$")

    print(q_pi_errors_df)
    print(q_star_errors_df)

    all_errors_df = pd.concat([q_pi_errors_df, q_star_errors_df])
    print(all_errors_df)

    # now do plot of errors
    fig, axs = plt.subplots(nrows=len(_GSS),
                            ncols=1,
                            sharex=True,
                            figsize=_FIGSIZE)

    for gs, ax in zip(_GSS, axs):

        cols_to_select = ([f"({gs}, {sp})" for sp in _SPS] + ["P"])
        this_gs_errors_df = all_errors_df.loc[:, cols_to_select]
        this_gs_errors_df.rename(
            columns={f"({gs}, {sp})": f"{sp}"
                     for sp in _SPS}, inplace=True)
        print(this_gs_errors_df)

        this_gs_errors_df = pd.melt(this_gs_errors_df,
                                    id_vars=["P"],
                                    var_name="sp",
                                    value_name="error")
        print(this_gs_errors_df)

        sns.boxplot(x="sp",
                    y="error",
                    hue="P",
                    palette=["tab:blue", "tab:green"],
                    data=this_gs_errors_df,
                    ax=ax)

        show_legend = (gs == _GSS[0])
        if not show_legend:
            ax.legend_.remove()

        ax.set_ylim(bottom=_MIN_ERROR)

        ax.set_title(r"$M = $" + f"{gs}")
        if (gs == _GSS[-1]):
            ax.set_xlabel(r"$p_{\mathrm{slip}}$")
        else:
            ax.set(xlabel=None)

        ylabel_brackets = _YLABEL_BRACKETS[err_type]
        ylabel = f"Ruleset error ({ylabel_brackets})"
        ax.set_ylabel(ylabel)

    # sync axs top ylims
    max_ylim_top = max([ax.get_ylim()[1] for ax in axs])
    for ax in axs:
        ax.set_ylim(bottom=_MIN_ERROR, top=max_ylim_top)

    plt.tight_layout()
    plt.savefig(
        f"accuracy_of_strength_transenv_ruleset_error_{err_type}_plot.pdf",
        bbox_inches="tight")


def _calc_ruleset_errors(gs, sp, err_type, q_ref_type):

    print(q_ref_type, gs, sp)

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

    ruleset_errors = []
    for (exp_dir, q_dict) in zip(exp_dirs, q_dicts):
        ruleset_errors.append(
            _calc_ruleset_error(exp_dir, nonterm_obs_tuples, err_type, q_dict))
    return ruleset_errors


def _calc_ruleset_error(exp_dir, nonterm_obs_tuples, err_type, q_dict):

    with open(f"{exp_dir}/best_indiv_history.pkl", "rb") as fp:
        hist = pickle.load(fp)
    indiv = hist[_NUM_GENS]

    nonterm_covered_obs_action_pairs = _find_nonterm_covered_obs_action_pairs(
        indiv, nonterm_obs_tuples)

    errors = []
    for obs_action_pair in nonterm_covered_obs_action_pairs:
        (obs, action) = obs_action_pair

        q_actual = q_dict[obs_action_pair]
        q_pred = _gen_q_pred(indiv, obs, action)

        if err_type == "abs":

            error = np.abs(q_actual - q_pred)

        elif err_type == "rel":

            abs_diff = np.abs(q_actual - q_pred)
            norm = np.max([np.abs(q_actual), np.abs(q_pred)])
            if norm == 0:
                norm += _REL_ERR_EPSILON
            error = (abs_diff / norm)

        else:
            assert False

        errors.append(error)

    return np.mean(errors)


def _find_nonterm_covered_obs_action_pairs(indiv, nonterm_obs_tuples):
    covered = set()

    for rule in indiv.rules:
        nonterm_covered_obs_tuples = _find_nonterm_covered_obs_tuples(
            rule.condition, nonterm_obs_tuples)
        for obs_tuple in nonterm_covered_obs_tuples:
            obs_action_pair = ((obs_tuple), rule.action)
            covered.add(obs_action_pair)

    return list(covered)


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


def _gen_q_pred(indiv, obs, action):
    match_set = [rule for rule in indiv.rules if rule.does_match(obs)]
    action_set = [rule for rule in match_set if rule.action == action]
    assert len(action_set) > 0

    aug_obs = augment_obs(obs, x_nought=indiv.x_nought)
    max_strength = np.max([rule.strength(aug_obs) for rule in action_set])

    return max_strength


if __name__ == "__main__":
    main()
