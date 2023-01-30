import constants as const
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from diver import Diver
from wind_and_rho_generator import Wind_generator
from scipy.optimize import curve_fit

NR_OF_SIMS = 5
H_VAL = 0.05


def simulate_params(params):
    x, y, d, w_x_bounds, w_y_bounds = params[:5]
    pos = np.array([x, y, const.h_plane])
    v = np.array([np.cos(d) * const.v_plane, np.sin(d) * const.v_plane, 0])
    wind = Wind_generator(w_x_bounds, w_y_bounds)

    landing_locations = []
    for seed in params[5:]:
        myDiver = Diver(pos, v, wind, H_VAL, seed=seed)
        myDiver.simulate_trajectory()
        landing_locations.append(myDiver.x[:2])

    return params[:5] + landing_locations

def find_optimal_params(params):
    d, w_x_bounds, w_y_bounds, h_opening = params
    params = [0, 0, d, w_x_bounds, w_y_bounds, h_opening, *range(10)]
    locations = simulate_params(params)[5:]
    avg_x, avg_y = sum(np.array(locations)) / len(locations)
    avg_distance = sum(np.sqrt((x - avg_x) ** 2 + (y - avg_y) ** 2)
                    for x, y in locations) / len(locations)
    return avg_distance

def fit_func(X, a, b, c, d):
    # print(list(X))
    h, v = X
    # print(v)
    # print(h)
    # print()
    return a * v * h + b * h + c * v + d

def chute_opening_plot():
    h_opening_vals = np.linspace(150, 450, 50)
    v_vals = []
    dist_vals = []
    seeds = np.arange(len(h_opening_vals))

    pool = multiprocessing.Pool()
    results = pool.map(chute_opening_simulation, zip(h_opening_vals, seeds))
    pool.close()

    for v, dist in results:
        v_vals.append(v)
        dist_vals.append(dist)

    params, _, = curve_fit(fit_func, (h_opening_vals, v_vals), dist_vals)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(h_opening_vals, v_vals, dist_vals, label='data')
    reg_v_vals = np.linspace(min(v_vals), max(v_vals), len(h_opening_vals))
    # h_vals = np.array(list(h_opening_vals) * len(h_opening_vals))
    # reg_v_vals = np.array(list(reg_v_vals) * len(h_opening_vals))

    X, Y = np.meshgrid(h_opening_vals, reg_v_vals)
    ax.plot_surface(X, Y, fit_func((X, Y), *params))
    ax.set_xlabel('h_opening')
    ax.set_ylabel('current velocity')
    ax.set_zlabel('distance covered')
    plt.show()


def chute_opening_simulation(params):
    h_opening, seed = params
    pos = np.array([0, 0, const.h_plane])
    v = np.array([const.v_plane, 0, 0])

    wind = Wind_generator()
    myDiver = Diver(pos, v, wind, H_VAL, seed=seed, h_opening=h_opening)
    myDiver.simulate_trajectory()

    for (x, y, z), (v_x, v_y, _) in zip(myDiver.x_list, myDiver.v_list):
        if z < h_opening:
            x_end, y_end = myDiver.x_list[-1][:2]
            distance = np.sqrt((x - x_end) ** 2 + (y - y_end) ** 2)
            v = np.sqrt(v_x ** 2 + v_y ** 2)
            return v, distance

def plot_params(x_vals, y_vals, dir_vals, w_x_bounds, w_y_bounds):
    seeds = np.arange(len(x_vals) * len(y_vals) * len(dir_vals) * NR_OF_SIMS)
    seeds = np.reshape(seeds, (-1, NR_OF_SIMS))
    params_list = [[x, y, d, w_x_bounds, w_y_bounds]
                   for x in x_vals for y in y_vals for d in dir_vals]

    params_list = [params + list(r) for params, r in zip(params_list, seeds)]
    pool = multiprocessing.Pool()
    landing_locations = pool.map(simulate_params, params_list)
    pool.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_list, y_list, z_list, p_list = [], [], [], []
    for x, y, d, _, _, locations in landing_locations:
        p = len(1 for x, y in locations \
                if x ** 2 + y ** 2 < const.radius_landing_area ** 2)
        x_list.append(x)
        y_list.append(y)
        z_list.append(d)
        p_list.append(p)

    plot = ax.scatter(x_list, y_list, z_list, c=p_list, cmap=plt.cm.RdYlGn,
                      vmin=0, vmax=1)
    ax.set_xlabel('x-position')
    ax.set_ylabel('y-position')
    ax.set_zlabel('approach angle')

    fig.colorbar(plot, ax=ax)

    plt.show()


# if __name__ == '__main__':
#     dir_vals = np.linspace(0, 2 * np.pi, NUMBER_D)
#     x_vals = np.linspace(-1000, -500, NUMBER_X)
#     y_vals = np.linspace(0, 0, NUMBER_Y)
#     simulate_params(x_vals, y_vals, dir_vals)
