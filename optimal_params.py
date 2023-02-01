import constants as const
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from diver import Diver
from wind_and_rho_generator import Wind_generator
from dynamic_opening import pred_func
from statistics import stdev

H_VAL = 0.05


def simulate_params(x, y, h_opening, dynamic_funcs=None, seeds=[None]):
    pos = np.array([x, y, const.h_plane], dtype=float)
    v = np.array([const.v_plane, 0, 0], dtype=float)
    w = Wind_generator(const.w_x_bounds, const.w_y_bounds)

    landing_locations = []
    for seed in seeds:
        myDiver = Diver(pos, v, w, H_VAL, 'rk4', seed, h_opening, dynamic_funcs)
        myDiver.simulate_trajectory()
        landing_locations.append(myDiver.x[:2])

    return landing_locations

def simulate_height(h_opening):
    result = np.array(simulate_params(0, 400, h_opening, seeds=range(100)))
    y_vals = result.transpose()[1]
    return abs(np.mean(y_vals)), stdev(y_vals)


def find_optimal_height():
    heights = np.arange(200, 221, 2)
    pool = multiprocessing.Pool()
    result = np.array(pool.map(simulate_height, heights))
    pool.close()

    res = np.array([i[0] for i in result])
    dev = np.array([i[1] for i in result])
    plt.figure(figsize=(5,4), dpi=200)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    plt.title('Landing location opening height')
    plt.fill_between(heights, res - dev, res + dev,
                     alpha=0.2, color=const.color_fitted_dev, label='standard deviation')
    plt.plot(heights, res, '#6D0DD5', label='Opening')
    plt.xlabel('opening height(m)')
    plt.ylabel('distance from origin(m)')
    plt.legend()
    plt.show()

    return heights[result.argsort()][0]


def parallel_func(params):
    x,y, h_opening, dynamic_funcs, seed = params
    if dynamic_funcs:
        dist_func = pred_func('dist', 'h', 'w')
        dir_func = pred_func('dir', 'h', 'wx', 'wy')
        return simulate_params(x, y, h_opening, dynamic_funcs=(dist_func, dir_func), seeds=[seed])
    return simulate_params(x, y, h_opening, seeds=[seed])

def find_optimal_x(h_opening):
    pool = multiprocessing.Pool()
    seeds = range(10)
    landing_locations = pool.map(parallel_func, [(0, 0, h_opening, None, s) for s in seeds])
    landing_locations = np.array(landing_locations).reshape((-1, 2))
    # print(landing_locations)
    pool.close()
    return -np.mean(landing_locations[:,0])
    # return -np.mean([simulate_params(0, 0, h_opening) for _ in range(10)])


def plot_optimal_params():
    h = find_optimal_height()
    # print(h)
    x = find_optimal_x(h)
    # print(x)

    seeds = range(20)

    pool = multiprocessing.Pool()
    stat_locs = pool.map(parallel_func, [(x, 400, h, None, s) for s in seeds])
    dyn_locs = pool.map(parallel_func, [(x, 400, h, True, s) for s in seeds])
    pool.close()

    stat_locs = np.array(stat_locs).reshape((-1, 2)).transpose()
    dyn_locs = np.array(dyn_locs).reshape((-1, 2)).transpose()


    # print(stat_locs)
    # print(dyn_locs)

    stat_avg = np.array([np.mean(stat_locs[0]), np.mean(stat_locs[1])])
    dyn_avg = np.array([np.mean(dyn_locs[0]), np.mean(dyn_locs[1])])

    stat_std_dev = [stdev(stat_locs[0]), stdev(stat_locs[1])]
    dyn_std_dev = [stdev(dyn_locs[0]), stdev(dyn_locs[1])]

    print(f'{stat_avg = }')
    print(f'{stat_std_dev = }')
    print(f'{dyn_avg = }')
    print(f'{dyn_std_dev = }')
    print()

    stat_std_dev = np.mean([np.linalg.norm(loc - stat_avg) for loc in stat_locs.transpose()])
    dyn_std_dev = np.mean([np.linalg.norm(loc - dyn_avg) for loc in dyn_locs.transpose()])


    print(f'{stat_avg = }')
    print(f'{stat_std_dev = }')
    print(f'{dyn_avg = }')
    print(f'{dyn_std_dev = }')


    plt.figure(figsize=(5,4), dpi=300)

    stat_color = '#4b64cf'
    dyn_color = '#cf744b'

    # circle_stat = plt.Circle(stat_avg, stat_std_dev, alpha=0.3, color=stat_color)
    # plt.gca().add_patch(circle_stat)

    # circle_dyn = plt.Circle(dyn_avg, dyn_std_dev, alpha=0.3, color=dyn_color)
    # plt.gca().add_patch(circle_dyn)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    plt.title('Static and dynamic opening simulations')
    plt.scatter(*stat_locs, s=4, color=stat_color, label='static')
    plt.scatter(*dyn_locs, s=4, color=dyn_color ,label='dynamic')
    plt.plot([0], [0], color='black', marker='x', linestyle='None', label='landing target')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()
