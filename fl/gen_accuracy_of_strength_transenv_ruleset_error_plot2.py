import glob
import itertools
import math
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

#_YLABEL_BRACKETS = {
#    ("abs", "q_pi"): r"$Q^\pi\ \mathrm{MAE}$",
#    ("abs", "q_star"): r"$Q^*\ \mathrm{MAE}$",
#    ("rel", "q_pi"): r"$Q^\pi\ \mathrm{MRE}$",
#    ("rel", "q_star"): r"$Q^*\ \mathrm{MRE}$"
#}
_YLABEL_BRACKETS = {
    ("abs", "q_pi"): "abs.",
    ("abs", "q_star"): "abs.",
    ("rel", "q_pi"): "rel.",
    ("rel", "q_star"): "rel."
}


def main():
    err_type = sys.argv[1]
    assert err_type in _ERR_TYPES
    q_ref_type = sys.argv[2]
    assert q_ref_type in _Q_REF_TYPES

    # first just calc all the errs across all envs
    errors_df = pd.DataFrame(
        columns=[f"({gs}, {sp})" for gs in _GSS for sp in _SPS])

    for gs in _GSS:
        for sp in _SPS:
            ruleset_errors = _calc_ruleset_errors(gs, sp, err_type, q_ref_type)
            errors_df[f"({gs}, {sp})"] = ruleset_errors

    print(errors_df)

    # now do plot of errors
    fig, axs = plt.subplots(nrows=len(_GSS),
                            ncols=1,
                            sharex=True,
                            figsize=_FIGSIZE)

    ylabel_brackets = _YLABEL_BRACKETS[(err_type, q_ref_type)]
    ylabel = f"Ruleset error ({ylabel_brackets})"

    for gs, ax in zip(_GSS, axs):

        ax.set_title(r"$M = $" + f"{gs}")
        if gs == _GSS[-1]:
            ax.set_xlabel(r"$p_{\mathrm{slip}}$")
        ax.set_ylabel("Ruleset error")

        df = pd.DataFrame(columns=[str(e) for e in _SPS])

        for sp in _SPS:
            df[str(sp)] = errors_df[f"({gs}, {sp})"]

        sns.boxplot(data=df,
                    ax=ax,
                    palette=["tab:green", "tab:blue", "tab:olive", "tab:red"])
        ax.set_ylim(bottom=_MIN_ERROR)
        ax.set_ylabel(ylabel)

    # sync axs top ylims
    max_ylim_top = max([ax.get_ylim()[1] for ax in axs])

    max_ylim_top = (0.6 if err_type == "abs" else 1.5)
    ytick_step_size = (0.05 if err_type == "abs" else 0.1)

    yticks = [
        i * ytick_step_size
        for i in range(0,
                       math.ceil(max_ylim_top / ytick_step_size) + 1)
    ]
    for ax in axs:
        ax.set_ylim(bottom=_MIN_ERROR, top=max_ylim_top)
        ax.set_yticks(yticks)

    plt.tight_layout()
    plt.savefig(
        f"accuracy_of_strength_transenv_ruleset_error_{err_type}_{q_ref_type}_plot.pdf",
        bbox_inches="tight")

    # bonus plot
    perfs_df = pd.DataFrame(
        columns=[f"({gs}, {sp})" for gs in _GSS for sp in _SPS])
    for (gs, sp) in itertools.product(_GSS, _SPS):
        if sp == 0:
            glob_expr = f"./frozen_redux/detrm/gs_{gs}/6*"
        else:
            glob_expr = f"./frozen_redux/stoca/gs_{gs}_sp_{sp}/6*"
        exp_dirs = glob.glob(glob_expr)
        assert len(exp_dirs) == _EXPECTED_NUM_EXP_DIRS

        perfs = []
        for exp_dir in exp_dirs:
            with open(f"{exp_dir}/best_indiv_test_perf_history.pkl",
                      "rb") as fp:
                hist = pickle.load(fp)
            perf = hist[_NUM_GENS].perf
            perfs.append(perf)

        perfs_df[f"({gs}, {sp})"] = perfs

    print(perfs_df)

    fig, axs = plt.subplots(nrows=len(_GSS), ncols=len(_SPS))
    for (row_idx, gs) in enumerate(_GSS):
        for (col_idx, sp) in enumerate(_SPS):
            ax = axs[row_idx][col_idx]
            str_ = f"({gs}, {sp})"
            x = errors_df[str_]
            y = perfs_df[str_]
            ax.scatter(x, y, s=4)
            ax.set_title(str_)
    plt.savefig("test.pdf", bbox_inches="tight")


def _calc_ruleset_errors(gs, sp, err_type, q_ref_type):

    print(gs, sp)

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
