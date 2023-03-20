import itertools
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pplst.inference import infer_action_and_action_set
from pplst.util import augment_obs
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

_GS = 8
_SP = 0.3

_XS = list(range(0, _GS))
_YS = list(range(0, _GS))

_EDGE_EXTEND = 0.5

_NUM_GENS = 250

_ACTION_SPACE = (0, 1, 2, 3)
_ACTION_COLORS = {0: 'blue', 1: 'red', 2: 'orange', 3: 'green'}
_ACTION_LABELS = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

_ALPHA = 0.3

_SCATTER_MARKERSIZE = 3

_LEGEND_LINEWIDTH = 3

_X_Y_AXIS_TICK_FONTSIZE = "x-small" if _GS == 12 else "medium"

_AZIM = 225
_ELEV = 25


def main():
    env = make_fl(_GS, _SP, iod_strat="top_left")
    nonterm_obs_tuples = [tuple(s) for s in env.nonterminal_states]

    if _SP == 0:
        pkl_path = f"./frozen_redux/detrm/gs_{_GS}/best/best_indiv_history.pkl"
    else:
        assert _SP in (0.1, 0.3, 0.5)
        pkl_path = f"./frozen_redux/stoca/gs_{_GS}_sp_{_SP}/best/best_indiv_history.pkl"

    with open(pkl_path, "rb") as fp:
        hist = pickle.load(fp)
    best = hist[_NUM_GENS]

    x_nought = best.x_nought

    _gen_piecewise_rule_surf_plot(
        best.rules,
        x_nought,
        nonterm_obs_tuples,
        filename=f"./pplst_best_rule_surf_plot_gs_{_GS}_sp_{_SP}_full.pdf")

    rules_bam = _find_rules_in_bam(best, nonterm_obs_tuples)
    assert len(rules_bam) <= len(best.rules)

    _gen_piecewise_rule_surf_plot(
        rules_bam,
        x_nought,
        nonterm_obs_tuples,
        filename=f"./pplst_best_rule_surf_plot_gs_{_GS}_sp_{_SP}_bam.pdf")

    print(len(best.rules), len(rules_bam))
    #sys.exit(1)
    _gen_max_strength_wireframe_plot(best, nonterm_obs_tuples,
                                     env.action_space)


def _gen_piecewise_rule_surf_plot(rules, x_nought, nonterm_obs_tuples,
                                  filename):
    plt.figure()
    ax = plt.axes(projection='3d', computed_zorder=False)

    for rule in rules:
        print("\n")

        phenotype = rule.condition.phenotype
        x_min = phenotype[0].lower
        x_max = phenotype[0].upper
        y_min = phenotype[1].lower
        y_max = phenotype[1].upper
        print(f"[{x_min}, {x_max}] X [{y_min}, {y_max}]")

        xs = ([x_min - _EDGE_EXTEND] + list(range(x_min, x_max + 1, 1)) +
              [x_max + _EDGE_EXTEND])
        ys = ([y_min - _EDGE_EXTEND] + list(range(y_min, y_max + 1, 1)) +
              [y_max + _EDGE_EXTEND])
        xs, ys = np.meshgrid(xs, ys)

        def _strengths(xs, ys):
            def _strength(x, y):
                obs = np.asarray([x, y])
                aug_obs = augment_obs(obs, x_nought)
                return rule.strength(aug_obs)

            assert xs.shape == ys.shape
            (num_rows, num_cols) = xs.shape
            zs = np.full(shape=(num_rows, num_cols), fill_value=np.nan)

            for row_idx in range(0, num_rows):
                for col_idx in range(0, num_cols):
                    x = xs[row_idx][col_idx]
                    y = ys[row_idx][col_idx]
                    zs[row_idx][col_idx] = _strength(x, y)

            assert not (np.isnan(zs).all())
            return zs

        zs = _strengths(xs, ys)
        print(zs)
        print(rule.weight_vec, rule.payoff_stdev)

        action = rule.action
        assert action in _ACTION_SPACE
        color = _ACTION_COLORS[action]
        # plot the surface, then the individual points
        ax.plot_surface(xs, ys, zs, color=color, alpha=_ALPHA, zorder=1337)

        # individual "inside" points excluding the boundaries used for plotting
        # the plane
        xx = xs[1:-1, 1:-1]
        yy = ys[1:-1, 1:-1]
        zz = zs[1:-1, 1:-1]

        ax.scatter(xx,
                   yy,
                   zz,
                   color=color,
                   s=_SCATTER_MARKERSIZE,
                   alpha=_ALPHA,
                   zorder=1337)

    ax.set_xticks(_XS)
    ax.set_yticks(_YS)
    ax.tick_params(axis='x', labelsize=_X_Y_AXIS_TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=_X_Y_AXIS_TICK_FONTSIZE)
    ax.set_xlim((_XS[0] - _EDGE_EXTEND), (_XS[-1] + _EDGE_EXTEND))
    ax.set_ylim((_YS[0] - _EDGE_EXTEND), (_YS[-1] + _EDGE_EXTEND))

    ax.grid(False)

    _plot_frozen_lake_grid(ax, nonterm_obs_tuples)

    ax.azim = _AZIM
    ax.elev = _ELEV

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel("Strength")

    legend_handles = []
    for color in _ACTION_COLORS.values():
        legend_handles.append(
            Line2D([0], [0], color=color, lw=_LEGEND_LINEWIDTH))
    legend_labels = _ACTION_LABELS.values()
    plt.legend(legend_handles,
               legend_labels,
               bbox_to_anchor=(1, 0.5),
               loc="center left")

    plt.savefig(filename, bbox_inches="tight")


def _plot_frozen_lake_grid(ax, nonterm_obs_tuples):
    # plot the actual frozen lake grid at the bottom of the zaxis
    # first extend the zlims a tiny bit
    (zlim_low, zlim_high) = ax.get_zlim()
    if zlim_low < 0:
        new_zlim_low = (zlim_low * 1.1)
    else:
        new_zlim_low = (zlim_low * 0.9)
    ax.set_zlim(new_zlim_low, zlim_high)

    grid_z = new_zlim_low

    # plot a plane for each cell
    for (x, y) in itertools.product(_XS, _YS):
        xs = [(x - _EDGE_EXTEND), (x + _EDGE_EXTEND)]
        ys = [(y - _EDGE_EXTEND), (y + _EDGE_EXTEND)]
        xs, ys = np.meshgrid(xs, ys)
        zs = np.full(shape=(xs.shape), fill_value=grid_z)
        if (x, y) in nonterm_obs_tuples:
            # frozen
            ax.plot_surface(xs, ys, zs, color="white", shade=False, zorder=420)
        else:
            # hole / goal
            ax.plot_surface(xs,
                            ys,
                            zs,
                            color="darkgray",
                            shade=False,
                            zorder=420)

    # plot gridlines between cells
    num_points = 100
    for x in np.arange((_XS[0] - _EDGE_EXTEND), (_XS[-1] + 2 * _EDGE_EXTEND),
                       _EDGE_EXTEND * 2):
        xs = np.asarray([x] * num_points)
        ys = np.linspace((_YS[0] - _EDGE_EXTEND), (_YS[-1] + _EDGE_EXTEND),
                         num_points,
                         endpoint=True)
        zs = np.asarray([grid_z] * num_points)
        ax.plot(xs, ys, zs, color="black", zorder=420, linewidth=0.33)

    for y in np.arange((_YS[0] - _EDGE_EXTEND), (_YS[-1] + 2 * _EDGE_EXTEND),
                       _EDGE_EXTEND * 2):
        xs = np.linspace((_XS[0] - _EDGE_EXTEND), (_XS[-1] + _EDGE_EXTEND),
                         num_points,
                         endpoint=True)
        ys = np.asarray([y] * num_points)
        zs = np.asarray([grid_z] * num_points)
        ax.plot(xs, ys, zs, color="black", zorder=420, linewidth=0.33)


def _find_rules_in_bam(best, nonterm_obs_tuples):
    rules_bam = []
    for (x, y) in itertools.product(_XS, _YS):
        if (x, y) in nonterm_obs_tuples:
            obs = np.asarray([x, y])
            (_, best_action_set) = infer_action_and_action_set(best, obs)
            for rule in best_action_set:
                if rule not in rules_bam:
                    rules_bam.append(rule)
    return rules_bam


def _gen_max_strength_wireframe_plot(best, nonterm_obs_tuples, action_space):
    xs, ys = np.meshgrid(_XS, _YS)

    assert xs.shape == ys.shape
    (num_rows, num_cols) = xs.shape
    zs = np.full(shape=(num_rows, num_cols), fill_value=np.nan)
    best_actions = np.full(shape=(num_rows, num_cols), fill_value=np.nan)

    for row_idx in range(0, num_rows):
        for col_idx in range(0, num_cols):
            x = xs[row_idx][col_idx]
            y = ys[row_idx][col_idx]
            obs = np.asarray([x, y])
            if tuple(obs) in nonterm_obs_tuples:
                (best_action,
                 best_action_set) = infer_action_and_action_set(best, obs)
                best_actions[row_idx][col_idx] = best_action
                aug_obs = augment_obs(obs, x_nought=best.x_nought)
                max_strength = max(
                    [rule.strength(aug_obs) for rule in best_action_set])
                zs[row_idx][col_idx] = max_strength

    with np.printoptions(linewidth=200, precision=3):
        print(zs)
        print(best_actions)

    plt.figure()
    # https://stackoverflow.com/questions/37611023/3d-parametric-curve-in-matplotlib-does-not-respect-zorder-workaround
    ax = plt.axes(projection='3d', computed_zorder=False)

    ax.plot_wireframe(xs,
                      ys,
                      zs,
                      color="black",
                      alpha=_ALPHA * 1.75,
                      zorder=1337)

    for row_idx in range(0, num_rows):
        for col_idx in range(0, num_cols):
            x = xs[row_idx][col_idx]
            y = ys[row_idx][col_idx]
            z = zs[row_idx][col_idx]
            best_action = best_actions[row_idx][col_idx]
            if not np.isnan(best_action):
                color = _ACTION_COLORS[int(best_action)]
            else:
                color = "black"
            ax.scatter(x,
                       y,
                       z,
                       color=color,
                       s=_SCATTER_MARKERSIZE * 5,
                       zorder=1337)

    # also plot the max strength for each non-optimal action in each frozen
    # state
    # TODO brain no worky atm, something about meshgrid that assert down below
    # is dependent on....
    for row_idx in range(0, num_rows):
        for col_idx in range(0, num_cols):
            x = xs[row_idx][col_idx]
            y = ys[row_idx][col_idx]
            z = zs[row_idx][col_idx]
            best_action = best_actions[row_idx][col_idx]
            if not np.isnan(best_action):
                obs = np.asarray([x, y])
                other_actions = (set(action_space) - set([int(best_action)]))
                print(best_action, other_actions)
                for action in other_actions:
                    match_set = [
                        rule for rule in best.rules if rule.does_match(obs)
                    ]
                    action_set = [
                        rule for rule in match_set if rule.action == action
                    ]
                    if len(action_set) > 0:
                        aug_obs = augment_obs(obs, x_nought=best.x_nought)
                        max_strength = max(
                            [rule.strength(aug_obs) for rule in action_set])
                        assert max_strength <= z
                        color = _ACTION_COLORS[int(action)]
                        print(obs, action, max_strength)
                        ax.scatter(x,
                                   y,
                                   max_strength,
                                   color=color,
                                   s=_SCATTER_MARKERSIZE * 2.5,
                                   alpha=0.33,
                                   zorder=(1337-1))

    ax.grid(False)

    _plot_frozen_lake_grid(ax, nonterm_obs_tuples)

    ax.set_xticks(_XS)
    ax.set_yticks(_YS)
    ax.tick_params(axis='x', labelsize=_X_Y_AXIS_TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=_X_Y_AXIS_TICK_FONTSIZE)
    ax.azim = _AZIM + 20
    ax.elev = _ELEV * 1.75

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel("Strength")

    # thanks https://stackoverflow.com/questions/11423369/matplotlib-legend-circle-markers
    legend_handles = []
    for color in _ACTION_COLORS.values():
        legend_handles.append(
            Line2D([0], [0], color="white", marker="o", markerfacecolor=color))
    legend_labels = list(_ACTION_LABELS.values())
    plt.legend(legend_handles,
               legend_labels,
               bbox_to_anchor=(1, 0.5),
               loc="center left")

    plt.savefig(f"./pplst_max_strength_wireframe_plot_gs_{_GS}_sp_{_SP}.pdf",
                bbox_inches="tight")


if __name__ == "__main__":
    main()
