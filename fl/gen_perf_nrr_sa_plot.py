import glob
import pickle
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

#matplotlib.use("Agg")

_EXPECTED_NUM_EXP_DIRS = 30
_NUM_GENS = 250
_EXPECTED_HISTORY_LEN = (_NUM_GENS + 1)

_GRID_SIZE = 8
_NRR_VALS = (1, 2, 5, 10, 20, 50)

_EMP_OPT_PERFS_PKL_FILE = \
    "./FrozenLake_emp_opt_perfs_gamma_0.95_iodsb_frozen.pkl"

_VAR_STD_DDOF = 1  # use bessel's correction


def main():
    case = sys.argv[1]
    assert case in ("detrm", "stoca")
    if case == "detrm":
        sp = 0
        base_dir = "./frozen_redux/nrr_sens_analysis/detrm"
    elif case == "stoca":
        sp = 0.3
        base_dir = "./frozen_redux/nrr_sens_analysis/stoca"

    with open(f"{_EMP_OPT_PERFS_PKL_FILE}", "rb") as fp:
        empirical_opt_perfs = pickle.load(fp)
    emp_opt_perf = empirical_opt_perfs[(_GRID_SIZE, sp)].perf

    # fill end test perf map
    nrr_end_test_perf_map = OrderedDict({nrr: [] for nrr in _NRR_VALS})
    for nrr in _NRR_VALS:
        exp_dirs = glob.glob(f"{base_dir}/{nrr}/*")
        assert len(exp_dirs) == _EXPECTED_NUM_EXP_DIRS

        for exp_dir in exp_dirs:
            with open(f"{exp_dir}/best_indiv_test_perf_history.pkl",
                      "rb") as fp:
                perf_history = pickle.load(fp)
            assert len(perf_history) == _EXPECTED_HISTORY_LEN
            # for perf history, key is num gens,
            # val is perf assessment response
            end_test_perf = perf_history[_NUM_GENS].perf
            pcnt_perf = (end_test_perf / emp_opt_perf)
            nrr_end_test_perf_map[nrr].append(pcnt_perf)

    # check map is valid
    for (k, v) in nrr_end_test_perf_map.items():
        assert len(v) == _EXPECTED_NUM_EXP_DIRS

    mean_perfs = {k: np.mean(v) for (k, v) in nrr_end_test_perf_map.items()}
    std_perfs = {
        k: np.std(v, ddof=_VAR_STD_DDOF)
        for (k, v) in nrr_end_test_perf_map.items()
    }

    print(list(zip(mean_perfs.values(), std_perfs.values())))

    plt.figure()
    xs = np.array(list(mean_perfs.keys()))
    ys = np.array(list(mean_perfs.values()))
    yerr = np.array(list(std_perfs.values()))
    print(yerr)
    plt.errorbar(xs, ys, yerr=yerr, fmt="o", capsize=6, markersize=6)
    plt.axhline(y=1.0, linestyle="dashed", linewidth=1.0, color="black")
    plt.xscale("log", base=10)
    plt.xticks(_NRR_VALS)
    plt.gca().set_xticklabels([str(nrr) for nrr in _NRR_VALS])
    plt.xlabel("Num. reinforcement rollouts")
    plt.ylabel("Final testing performance")
    plt.grid(axis="y")
    plt.show()
    plt.savefig(f"{base_dir}/pplst_nrr_plot_{case}.png", bbox_inches="tight")
    plt.savefig(f"{base_dir}/pplst_nrr_plot_{case}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
