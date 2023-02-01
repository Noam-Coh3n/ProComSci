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
