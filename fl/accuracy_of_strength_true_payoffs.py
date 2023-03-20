import glob
import os
import pickle
import sys

import numpy as np
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

_EXPECTED_NUM_EXP_DIRS = 30
_NUM_GENS = 250
CONVERGE_TOL = 1e-50
# both not important
_FL_IOD_STRAT = "top_left"
_FL_SEED = 0

_GAMMA = 0.95


def main():
    gs = int(sys.argv[1])
    sp = float(sys.argv[2])
    print(f"gs = {gs}, sp = {sp}\n")

    # TODO nope not in here we aren't
    # measuring two things
    # 1: how good is the weight vector of each rule? i.e. how well does it
    # predict the true payoff?
    # 2: how accurate is the variance estimation of each rule? i.e. how closely
    # does it match the true variance in payoff?

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

    for (i, exp_dir) in enumerate(exp_dirs):
        print(f"Exp dir {i+1} / {_EXPECTED_NUM_EXP_DIRS}")

        with open(f"{exp_dir}/best_indiv_history.pkl", "rb") as fp:
            hist = pickle.load(fp)
        indiv = hist[_NUM_GENS]

        # first, calc pi for this indiv
        pi = _make_policy(indiv, env)

        # second, do VI on pi to calc V^pi and Q^pi
        (_, q) = _do_value_iteration(env, pi)

        # third, find the covered (s,a) pairs by rules of this indiv
        covered_s_a_pairs = _find_covered_state_action_pairs(
            indiv, nonterm_obs_tuples)

        # fourth, filter Q^pi based on the covered (s, a) pairs
        true_payoffs = _get_true_payoffs(q, covered_s_a_pairs, env)

        print(len(true_payoffs))
        for (key, val) in true_payoffs.items():
            print(key, val)
        print("\n")

        # finally, save the true payoffs
        with open(f"{exp_dir}/best_final_indiv_true_payoffs.pkl", "wb") as fp:
            pickle.dump(true_payoffs, fp)

        # and delete old file if it exists
        old_file = f"{exp_dir}/best_final_indiv_true_payoff_estimates.pkl"
        try:
            os.remove(old_file)
        except FileNotFoundError:
            pass


def _make_policy(indiv, env):
    # store states as raw in dict so can be easily queried in VI state loop
    pi = {}
    for obs in env.nonterminal_states:
        action = indiv.select_action(obs)
        raw_obs = env._convert_x_y_obs_to_raw(obs)
        pi[raw_obs] = action
    return pi


def _do_value_iteration(env, pi):
    num_states = (env.grid_size**2)
    num_actions = len(env.action_space)

    raw_nonterminal_states = [
        env._convert_x_y_obs_to_raw(obs) for obs in env.nonterminal_states
    ]

    v_old = np.zeros(num_states)
    q_old = np.zeros((num_states, num_actions))

    converged = False
    iter_num = 0
    while not converged:
        print(f"VI iter {iter_num}")
        sys.stdout.flush()
        v_new = np.copy(v_old)
        q_new = np.copy(q_old)

        for state in raw_nonterminal_states:

            # first, apply Bellman equation for V^pi(s)
            pi_s = pi[state]
            transitions = env.P[state][pi_s]
            expected_reward = sum(
                [prob * reward for (prob, _, reward, _) in transitions])
            expected_value = sum([
                prob * v_old[new_state]
                for (prob, new_state, _, _) in transitions
            ])
            v_new[state] = (expected_reward + _GAMMA * expected_value)

            # second, apply Bellman equation for Q^pi(s,a)
            for action in range(num_actions):
                transitions = env.P[state][action]
                expected_reward = sum(
                    [prob * reward for (prob, _, reward, _) in transitions])
                expected_value = sum([
                    prob * v_old[new_state]
                    for (prob, new_state, _, _) in transitions
                ])
                q_new[state][action] = \
                    (expected_reward + _GAMMA*expected_value)

        converged = has_converged(v_old, v_new, q_old, q_new)
        v_old = v_new
        q_old = q_new
        iter_num += 1

    return v_new, q_new


def has_converged(v_old, v_new, q_old, q_new):
    v_max_diff = np.max(np.abs(v_old - v_new))
    print(f"Max diff in V: {v_max_diff}")
    q_max_diff = np.max(np.abs(q_old - q_new))

    has_converged = ((v_max_diff < CONVERGE_TOL)
                     and (q_max_diff < CONVERGE_TOL))
    if has_converged:
        print("VI Converged...")
    else:
        print("VI not converged, running another iter...")
    return has_converged


def _find_covered_state_action_pairs(indiv, nonterm_obs_tuples):
    rules = indiv.rules
    covered_s_a_pairs = []

    for obs in nonterm_obs_tuples:
        match_set = [rule for rule in rules if rule.does_match(obs)]
        assert len(match_set) > 0
        reprd_actions = set([rule.action for rule in match_set])
        for action in reprd_actions:
            covered_s_a_pairs.append((obs, action))

    return covered_s_a_pairs


def _get_true_payoffs(q, covered_s_a_pairs, env):
    true_payoffs = {s_a_pair: None for s_a_pair in covered_s_a_pairs}

    for s_a_pair in covered_s_a_pairs:
        (obs, action) = s_a_pair
        raw_obs = env._convert_x_y_obs_to_raw(obs)
        payoff = q[raw_obs][action]
        assert not np.isnan(payoff)
        true_payoffs[s_a_pair] = payoff

    assert (None not in true_payoffs.items())
    return true_payoffs


if __name__ == "__main__":
    main()
