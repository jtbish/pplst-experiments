#!/usr/bin/python3
import argparse
import glob
import logging
import os
import pickle
import shutil
import subprocess
import time
from multiprocessing import set_start_method
from pathlib import Path

import __main__
import numpy as np
from pplst.encoding import IntegerUnorderedBoundEncoding
from pplst.pplst import PPLST
from rlenvs.environment import assess_perf
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
_FL_SEED = 0
_ROLLS_PER_SI_PERF_DETERMINISTIC = 1
_ROLLS_PER_SI_PERF_STOCHASTIC = 30
_USE_INDIV_POLICY_CACHE = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--fl-grid-size", type=int, required=True)
    parser.add_argument("--fl-slip-prob", type=float, required=True)
    parser.add_argument("--fl-iod-strat-base-train", required=True)
    parser.add_argument("--fl-iod-strat-base-test", required=True)
    parser.add_argument("--pplst-num-gens", type=int, required=True)
    parser.add_argument("--pplst-seed", type=int, required=True)
    parser.add_argument("--pplst-pop-size", type=int, required=True)
    parser.add_argument("--pplst-indiv-size", type=int, required=True)
    parser.add_argument("--pplst-tourn-size", type=int, required=True)
    parser.add_argument("--pplst-p-cross", type=float, required=True)
    parser.add_argument("--pplst-p-cross-swap", type=float, required=True)
    parser.add_argument("--pplst-p-mut", type=float, required=True)
    parser.add_argument("--pplst-num-reinf-rollouts", type=int, required=True)
    parser.add_argument("--pplst-weight-i-min", type=float, required=True)
    parser.add_argument("--pplst-weight-i-max", type=float, required=True)
    parser.add_argument("--pplst-x-nought", type=float, required=True)
    parser.add_argument("--pplst-eta", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    return parser.parse_args()


def main(args):
    save_path = _setup_save_path(args.experiment_name)
    _setup_logging(save_path)
    logging.info(str(args))

    if args.fl_slip_prob == 0:
        is_deterministic = True
    elif 0 < args.fl_slip_prob < 1:
        is_deterministic = False
    else:
        assert False

    if is_deterministic:
        iod_strat_train_perf = args.fl_iod_strat_base_train + "_no_repeat"
        iod_strat_test_perf = args.fl_iod_strat_base_test + "_no_repeat"
    else:
        iod_strat_train_perf = args.fl_iod_strat_base_train + "_repeat"
        iod_strat_test_perf = args.fl_iod_strat_base_test + "_repeat"
    # reinf env always same as train, uniform rand
    iod_strat_reinf = args.fl_iod_strat_base_train + "_uniform_rand"

    train_perf_env = _make_env(args, iod_strat_train_perf)
    test_perf_env = _make_env(args, iod_strat_test_perf)
    reinf_env = _make_env(args, iod_strat_reinf)

    train_si_size = train_perf_env.si_size
    test_si_size = test_perf_env.si_size
    reinf_si_size = reinf_env.si_size
    assert train_si_size == reinf_si_size

    logging.info(f"Training on iod strat: {iod_strat_train_perf}")
    logging.info(f"Testing on iod strat: {iod_strat_test_perf}")
    logging.info(f"Reinforcing on iod strat: {iod_strat_reinf}")
    logging.info(f"Train si size = {train_si_size}")
    logging.info(f"Test si size = {test_si_size}")
    logging.info(f"Reinf si size = {reinf_si_size}")

    # calc num rollouts for perf envs
    if is_deterministic:
        num_train_perf_rollouts = (train_si_size * _ROLLS_PER_SI_PERF_DETERMINISTIC)
        logging.info(f"Using {train_si_size} * {_ROLLS_PER_SI_PERF_DETERMINISTIC} = "
                     f"{num_train_perf_rollouts} rollouts for training")
        num_test_perf_rollouts = (test_si_size * _ROLLS_PER_SI_PERF_DETERMINISTIC)
        logging.info(f"Using {test_si_size} * {_ROLLS_PER_SI_PERF_DETERMINISTIC} = "
                     f"{num_test_perf_rollouts} rollouts for testing")
    else:
        num_train_perf_rollouts = (train_si_size * _ROLLS_PER_SI_PERF_STOCHASTIC)
        logging.info(f"Using {train_si_size} * {_ROLLS_PER_SI_PERF_STOCHASTIC} = "
                     f"{num_train_perf_rollouts} rollouts for training")
        num_test_perf_rollouts = (test_si_size * _ROLLS_PER_SI_PERF_STOCHASTIC)
        logging.info(f"Using {test_si_size} * {_ROLLS_PER_SI_PERF_STOCHASTIC} = "
                     f"{num_test_perf_rollouts} rollouts for testing")

    pplst_hyperparams = {
        "seed": args.pplst_seed,
        "pop_size": args.pplst_pop_size,
        "indiv_size": args.pplst_indiv_size,
        "tourn_size": args.pplst_tourn_size,
        "p_cross": args.pplst_p_cross,
        "p_cross_swap": args.pplst_p_cross_swap,
        "p_mut": args.pplst_p_mut,
        "num_reinf_rollouts": args.pplst_num_reinf_rollouts,
        "num_perf_rollouts": num_train_perf_rollouts,
        "weight_I_min": args.pplst_weight_i_min,
        "weight_I_max": args.pplst_weight_i_max,
        "x_nought": args.pplst_x_nought,
        "eta": args.pplst_eta,
        "gamma": args.gamma,
        "use_indiv_policy_cache": _USE_INDIV_POLICY_CACHE
    }
    logging.info(pplst_hyperparams)
    encoding = IntegerUnorderedBoundEncoding(train_perf_env.obs_space)
    pplst = PPLST(reinf_env,
                  train_perf_env,
                  encoding,
                  hyperparams_dict=pplst_hyperparams)

    best_indiv_history = {}
    best_indiv_test_perf_history = {}
    init_pop = pplst.init()
    gen_num = 0
    _calc_pop_stats(gen_num, init_pop, test_perf_env, num_test_perf_rollouts,
                    args, best_indiv_history, best_indiv_test_perf_history)
    _save_pplst(save_path, pplst, gen_num)
    num_gens = args.pplst_num_gens
    for gen_num in range(1, num_gens + 1):
        pop = pplst.run_gen()
        _calc_pop_stats(gen_num, pop, test_perf_env, num_test_perf_rollouts,
                        args, best_indiv_history, best_indiv_test_perf_history)
        _save_pplst(save_path, pplst, gen_num)

    _save_histories(save_path, best_indiv_history,
                    best_indiv_test_perf_history)
    _save_main_py_script(save_path)
#    _compress_pplst_pkl_files(save_path, num_gens)
#    _delete_uncompressed_pplst_pkl_files(save_path)


def _setup_save_path(experiment_name):
    save_path = Path(args.experiment_name)
    save_path.mkdir(exist_ok=False)
    return save_path


def _setup_logging(save_path):
    logging.basicConfig(filename=save_path / "experiment.log",
                        format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)


def _make_env(args, iod_strat):
    return make_fl(grid_size=args.fl_grid_size,
                   slip_prob=args.fl_slip_prob,
                   iod_strat=iod_strat,
                   seed=_FL_SEED)


def _calc_pop_stats(gen_num, pop, test_perf_env, num_test_perf_rollouts, args,
                    best_indiv_history, best_indiv_test_perf_history):
    logging.info(f"gen num {gen_num}")
    fitnesses = [indiv.fitness for indiv in pop]
    min_ = np.min(fitnesses)
    mean = np.mean(fitnesses)
    median = np.median(fitnesses)
    max_ = np.max(fitnesses)
    logging.info(f"min, mean, median, max fitness in pop: {min_}, {mean}, "
                 f"{median}, {max_}")

    # find out test perf of max fitness indiv
    best_indiv = sorted(pop, key=lambda indiv: indiv.fitness, reverse=True)[0]
    res = assess_perf(test_perf_env, best_indiv, num_test_perf_rollouts,
                      args.gamma)
    logging.info(f"best test perf assess res: {res}")
    best_indiv_history[gen_num] = best_indiv
    best_indiv_test_perf_history[gen_num] = res

#    logging.debug("Best rule weight vec and vars")
#    for rule in best_indiv.rules:
#        logging.debug(f"w_vec: {rule.weight_vec}")
#        logging.debug(f"var: {rule.payoff_var}")
#    for indiv in pop:
#        logging.debug(f"Next indiv id: {indiv.id}")

    # statistics about truncations and failures
    num_truncs = len([
        indiv for indiv in pop if indiv.perf_assessment_res.time_limit_trunced
    ])
    num_failures = len(
        [indiv for indiv in pop if indiv.perf_assessment_res.failed])
    pop_size = len(pop)
    logging.info(f"Trunc rate = {num_truncs}/{pop_size} = "
                 f"{num_truncs/pop_size:.4f}")
    logging.info(f"Failure rate = {num_failures}/{pop_size} = "
                 f"{num_failures/pop_size:.4f}")


def _save_pplst(save_path, pplst, gen_num):
    with open(save_path / f"pplst_gen_{gen_num}.pkl", "wb") as fp:
        pickle.dump(pplst, fp)


def _save_histories(save_path, best_indiv_history,
                    best_indiv_test_perf_history):
    with open(save_path / "best_indiv_history.pkl", "wb") as fp:
        pickle.dump(best_indiv_history, fp)
    with open(save_path / "best_indiv_test_perf_history.pkl", "wb") as fp:
        pickle.dump(best_indiv_test_perf_history, fp)


def _save_main_py_script(save_path):
    main_file_path = Path(__main__.__file__)
    shutil.copy(main_file_path, save_path)


def _compress_pplst_pkl_files(save_path, num_gens):
    pplst_pkl_files = glob.glob(f"{save_path}/pplst*.pkl")
    assert len(pplst_pkl_files) == (num_gens + 1)
    os.environ["XZ_OPT"] = "-T0 -9e"
    subprocess.run(["tar", "-cJf", f"{save_path}/pplsts.tar.xz"] + pplst_pkl_files,
                   check=True)


def _delete_uncompressed_pplst_pkl_files(save_path):
    pplst_pkl_files = glob.glob(f"{save_path}/pplst*.pkl")
    for file_ in pplst_pkl_files:
        os.remove(file_)


if __name__ == "__main__":
    set_start_method("spawn")
    start_time = time.time()
    args = parse_args()
    main(args)
    end_time = time.time()
    elpased = end_time - start_time
    logging.info(f"Runtime: {elpased:.3f}s with {_NUM_CPUS} cpus")
