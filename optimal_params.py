import constants as const
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from diver import Diver
from wind_and_rho_generator import Wind_generator
from dynamic_opening import pred_func

H_VAL = 0.05


def simulate_params(x, y, h_opening, dynamic_funs=None, seeds=[None]):
    pos = np.array([x, y, const.h_plane], dtype=float)
    v = np.array([const.v_plane, 0, 0], dtype=float)
    w = Wind_generator(const.w_x_bounds, const.w_y_bounds)

    landing_locations = []
    for seed in seeds:
        myDiver = Diver(pos, v, w, H_VAL, 'rk4', seed, h_opening, dynamic_funs)
        myDiver.simulate_trajectory()
        landing_locations.append(myDiver.x[:2])

    return landing_locations


def simulate_height(h_opening):
    result = np.array(simulate_params(0, 400, h_opening, seeds=range(3)))
    y_vals = result.transpose()[1]
    return abs(np.mean(y_vals))


def find_optimal_height():
    heights = np.arange(100, 400, 10)
    pool = multiprocessing.Pool()
    result = np.array(pool.map(simulate_height, heights))
    pool.close()
    return heights[result.argsort()][0]


def parallel_func(params):
    x,y, h_opening, seed = params
    return simulate_params(x, y, h_opening, seeds=[seed])


def find_optimal_x(h_opening):
    pool = multiprocessing.Pool()
    seeds = range(10)
    landing_locations = pool.map(parallel_func, [(0, 0, h_opening, s) for s in seeds])
    landing_locations = np.array(landing_locations).reshape((-1, 2))
    print(landing_locations)
    pool.close()
    return -np.mean(landing_locations[:,0])
    # return -np.mean([simulate_params(0, 0, h_opening) for _ in range(10)])


def plot_optimal_params():
    h = find_optimal_height()
    print(h)
    x = find_optimal_x(h)
    print(x)

    dist_func = pred_func('dist', 'h', 'w')
    dir_func = pred_func('dir', 'h', 'wx', 'wy')
    seeds = range(3)

    pool = multiprocessing.Pool()
    static_locs = simulate_params(x, 400, h, seeds=seeds)
    dynamic_locs = simulate_params(x, 400, h, (dist_func, dir_func), seeds=seeds)
    pool.close()

    static_locs = np.array(static_locs).transpose()
    dynamic_locs = np.array(dynamic_locs).transpose()


    print(static_locs)
    print(dynamic_locs)

    plt.figure(figsize=(14,10), dpi=100)
    plt.title('Static and dynamic opening simulations')
    plt.scatter(*static_locs, s=1, label='static')
    plt.scatter(*dynamic_locs, s=1, label='dynamic')
    plt.plot([0], [0], color='black', marker='x', linestyle='None', label='landing target')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()
