#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Double Bachelor mathematics and computer science.
#
# optimal_params.py:
# This file will output 2 different plots when plot_optimal_params is run.
# The first plot shows for different values of parachute opening heights
# what the difference in y-value is between the landing position and the
# target (at y-value 400).
# The second plot is a scatter plot and shows the landing locations of the
# skydiver when he used a static and a dynamic opening height. Static in this
# context refers to the fact that the skydiver chooses the optimal opening
# height before flight while in the dynamic seting, the skydiver can choose
# the opening height during flight.

import constants as const
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from diver import Diver
from wind_and_rho_generator import Wind_generator
from dynamic_opening import pred_func
from statistics import stdev

H_VAL = 0.05


def simulate_params(x: float, y: float, h_opening: float,
                    dynamic_funcs: tuple = None,
                    seeds: list = [None]) -> list:
    """Simulates a skydiver for given seeds and returns landing locations."""

    # Initialize the skydiver position and velocity and the wind.
    pos = np.array([x, y, const.h_plane], dtype=float)
    v = np.array([const.v_plane, 0, 0], dtype=float)
    w = Wind_generator(const.w_x_bounds, const.w_y_bounds)

    # Simulate the skydiver for all given seeds.
    landing_locations = []
    for seed in seeds:
        myDiver = Diver(pos, v, w, H_VAL, 'rk4', seed,
                        h_opening, dynamic_funcs)
        myDiver.simulate_trajectory()
        landing_locations.append(myDiver.x[:2])

    return landing_locations


def simulate_height(h_opening: float) -> tuple:
    """Returns the average and standard deviation of the y travel distance."""
    result = np.array(simulate_params(0, 400, h_opening, seeds=range(10)))
    y_vals = result.transpose()[1]
    return abs(np.mean(y_vals)), stdev(y_vals)


def find_optimal_height() -> float:
    """Plots different opening heights and returns the best one."""
    heights = np.arange(100, 401, 20)

    # Simulates different heights in parallel.
    pool = multiprocessing.Pool()
    result = np.array(pool.map(simulate_height, heights))
    pool.close()

    avg = np.array([item[0] for item in result])
    std_dev = np.array([item[1] for item in result])

    # Plot the average distance and the standard deviaton as transparant area.
    plt.figure(figsize=(5, 4), dpi=200)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, bottom=0.19, right=0.95, top=0.9)
    plt.title('Landing location opening height')
    plt.fill_between(heights, avg - std_dev, avg + std_dev,
                     alpha=0.2, color=const.color_fitted_dev,
                     label='standard deviation')
    plt.plot(heights, avg, '#6D0DD5', label='Opening')
    plt.xlabel('opening height(m)')
    plt.ylabel('distance from origin(m)')
    plt.legend()
    plt.show()

    result = np.array(result)[:, 0]

    return heights[result.argsort()][0]


def parallel_func(params: list) -> list:
    x, y, h_opening, dynamic_funcs, seed = params

    if dynamic_funcs:
        # Define the dist_func (returns parachute travel distance given height
        # and wind speed) and dir_func (returns parachute travel direction
        # given height and wind speed in the x and y direction).
        dist_func = pred_func('dist', 'h', 'w')
        dir_func = pred_func('dir', 'h', 'wx', 'wy')
        return simulate_params(x, y, h_opening,
                               dynamic_funcs=(dist_func, dir_func),
                               seeds=[seed])
    return simulate_params(x, y, h_opening, seeds=[seed])


def find_optimal_x(h_opening: float) -> float:
    """Returns the optimal x coordinate of the jump position given a
    parachute opening height.
    """
    seeds = range(100)

    pool = multiprocessing.Pool()
    landing_locations = pool.map(parallel_func,
                                 [(0, 0, h_opening, None, s) for s in seeds])
    pool.close()

    landing_locations = np.array(landing_locations).reshape((-1, 2))
    return -np.mean(landing_locations[:, 0])


def plot_optimal_params() -> None:
    """Finds the optimal opening height and jump position and plots the
    landing locations of a skydiver in the static and dynamic setting that
    uses the found optimal parameters.
    """
    h = find_optimal_height()
    x = find_optimal_x(h)

    print('Optimal values for parachute opening height en jump position:')
    print(f'{h = }')
    print(f'{x = }')

    seeds = range(70)

    pool = multiprocessing.Pool()
    stat_locs = pool.map(parallel_func, [(x, 400, h, None, s) for s in seeds])
    dyn_locs = pool.map(parallel_func, [(x, 400, h, True, s) for s in seeds])
    pool.close()

    stat_locs = np.array(stat_locs).reshape((-1, 2)).transpose()
    dyn_locs = np.array(dyn_locs).reshape((-1, 2)).transpose()

    # Calculates the average landing location and average distance from
    # the average landing location (comparable to standard deviation).
    stat_avg_loc = np.array([np.mean(stat_locs[0]), np.mean(stat_locs[1])])
    dyn_avg_loc = np.array([np.mean(dyn_locs[0]), np.mean(dyn_locs[1])])

    stat_avg_dist = np.mean([np.linalg.norm(loc - stat_avg_loc)
                            for loc in stat_locs.transpose()])
    dyn_avg_dist = np.mean([np.linalg.norm(loc - dyn_avg_loc)
                           for loc in dyn_locs.transpose()])

    print(f'{stat_avg_loc = }')
    print(f'{stat_avg_dist = }')
    print(f'{dyn_avg_loc = }')
    print(f'{dyn_avg_dist = }')

    # Define colors for the static and dynamic setting and
    # use a scatter plot to plot the different landing locations.
    stat_color = '#4b64cf'
    dyn_color = '#cf744b'

    plt.figure(figsize=(5, 4), dpi=300)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)
    plt.title('Static and dynamic opening simulations')
    plt.scatter(*stat_locs, s=4, color=stat_color, label='static')
    plt.scatter(*dyn_locs, s=4, color=dyn_color, label='dynamic')
    plt.plot([0], [0], color='black', marker='x', linestyle='None',
             label='landing target')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()
