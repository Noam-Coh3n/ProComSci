import constants as const
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from diver import Diver
from wind_and_rho_generator import Wind_generator
from scipy.optimize import curve_fit

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
    result = np.array(simulate_params(0, 400, h_opening, seeds=range(20)))
    y_vals = result.transpose()[1]
    return abs(np.mean(y_vals))


def find_optimal_height():
    pool = multiprocessing.Pool()
    heights = np.arange(100, 400, 10)
    result = np.array(pool.map(simulate_height, heights))
    return heights[result.argsort()][0]


def find_optimal_x(h_opening):
    return -np.mean([simulate_params(0, 0, h_opening) for _ in range(100)])


def bilin_func(X, a, b, c, d):
    x, y = X
    return a * x * y + b * x + c * y + d


def trilin_func(X, a1, a2, a3, a4, a5, a6, a7, a8):
    x, y, z = X
    return a1*x*y*z + a2*x*y + a3*x*z + a4*y*z + a5*x + a6*y + a7*z + a8


def chute_opening_func(plot=False):
    h = np.linspace(const.min_h_opening, const.max_h_opening, 50)
    seeds = np.arange(len(h)) + 1000

    pool = multiprocessing.Pool()
    results = pool.map(chute_opening_simulation, zip(h, seeds))
    pool.close()

    wx, wy, dx, dy = np.array(results).transpose()
    w = np.sqrt(wx ** 2 + wy ** 2)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    dir = dy / dx

    names = ['h_w_dist',
             'h_wx_wy_dist',
             'h_w_dir',
             'h_wx_wy_dir',
             'dist_w_h',
             'dist_wx_wy_h']

    params = [curve_fit(bilin_func, (h, w), dist)[0],
              curve_fit(trilin_func, (h, wx, wy), dist)[0],
              curve_fit(bilin_func, (h, w), dir)[0],
              curve_fit(trilin_func, (h, wx, wy), dir)[0],
              curve_fit(bilin_func, (dist, w), h)[0],
              curve_fit(trilin_func, (dist, wx, wy), h)[0]]

    with open('params.txt', 'w') as f:
        for name, param in zip(names, params):
            f.write(f'{name} {list(param)}\n')


def pred_func(output, *inputs):
    with open('params.txt', 'r') as f:
        for line in f:
            name, *params = line.split()

            params = [float(param.replace('[', '').replace(']', '').replace(',', '')) for param in params]
            names = name.split('_')
            if names[:-1] == list(inputs) and names[-1] == output:
                if len(names) == 3:
                    return lambda x, y: bilin_func((x, y), *params)
                else:
                    return lambda x, y, z: trilin_func((x, y, z), *params)


def chute_opening_simulation(params):
    h_opening, seed = params
    pos = np.array([0, 0, const.h_plane])
    v = np.array([const.v_plane, 0, 0])

    wind = Wind_generator()
    myDiver = Diver(pos, v, wind, H_VAL, seed=seed, h_opening=h_opening)
    myDiver.simulate_trajectory()

    for x, y, z in myDiver.x_list:
        if z < h_opening:
            w_x = myDiver.wind_x(z)
            w_y = myDiver.wind_y(z)
            x_end, y_end = myDiver.x_list[-1][:2]
            dx = x - x_end
            dy = y - y_end
            return w_x, w_y, dx, dy
