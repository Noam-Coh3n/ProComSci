import constants as const
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from diver import Diver
from wind_and_rho_generator import Wind_generator
from scipy.optimize import curve_fit

NR_OF_SIMS = 5
H_VAL = 0.05

NR_OF_PARAMS = 6
chute_func_params = [0.20227224, 0.12543934, -4.04154588, -42.77106607]


def simulate_params(params):
    x, y, d, w_x_bounds, w_y_bounds, h_opening_func = params[:NR_OF_PARAMS]
    pos = np.array([x, y, const.h_plane])
    v = np.array([np.cos(d) * const.v_plane, np.sin(d) * const.v_plane, 0])
    wind = Wind_generator(w_x_bounds, w_y_bounds)

    landing_locations = []
    for seed in params[NR_OF_PARAMS:]:
        myDiver = Diver(pos, v, wind, H_VAL, seed=seed, h_opening_func=h_opening_func)
        myDiver.simulate_trajectory()
        landing_locations.append(myDiver.x[:2])

    return params[:NR_OF_PARAMS] + landing_locations

def find_optimal_params(params):
    x, y, d, w_x_bounds, w_y_bounds, func_params = params
    h_opening_func = (lambda X : fit_func(X, *func_params)) if func_params else None
    params = [x, y, d, w_x_bounds, w_y_bounds, h_opening_func, *range(5)]
    locations = simulate_params(params)[NR_OF_PARAMS:]
    avg_x, avg_y = sum(np.array(locations)) / len(locations)
    distances = [np.sqrt((x - avg_x) ** 2 + (y - avg_y) ** 2)
                         for x, y in locations]

    avg_distance = sum(distances) / len(distances)
    std_dev_distance = np.sqrt(sum((d - avg_distance) ** 2
                               for d in distances) / (len(distances) - 1))
    return avg_distance, std_dev_distance, avg_x, avg_y

def plot_optimal_params(dir_vals, w_x_bounds=const.w_x_bounds,
                        w_y_bounds=const.w_y_bounds):
    pool = multiprocessing.Pool()
    results = pool.map(find_optimal_params, [(0, 0, d, w_x_bounds, w_y_bounds, None) for d in dir_vals])
    pool.close()

    avg_dist, dev, avg_loc_x, avg_loc_y = np.array(results).transpose()

    avg_dist = np.array(avg_dist)
    dev = np.array(dev)

    lower = list(avg_dist - dev)
    upper = list(avg_dist + dev)

    plt.figure(figsize=(14,8), dpi=100)
    plt.fill_between(dir_vals, lower, upper,
                     alpha=0.3, color=const.color_dev, label='constant (std dev)')
    plt.plot(dir_vals, avg_dist, color=const.color_avg, label='constant (avg)')

    plt.show()

    # func = lambda X : fit_func(X, *chute_func_params)
    # print(func((150, 5)))


    pool = multiprocessing.Pool()
    results = pool.map(find_optimal_params, [(-x, -y, d, w_x_bounds, w_y_bounds, chute_func_params) for x, y, d in zip(avg_loc_x, avg_loc_y, dir_vals)])
    pool.close()

    avg_dist, dev, avg_loc_x, avg_loc_y = np.array(results).transpose()

    print(f'{avg_loc_x = }\n{avg_loc_y = }\n')

    plt.figure(figsize=(14,8), dpi=100)
    plt.fill_between(dir_vals, avg_dist - dev, avg_dist + dev,
                     alpha=0.3, color=const.color_dev, label='variable (std dev)')
    plt.plot(dir_vals, avg_dist, color=const.color_avg, label='variable (avg)')
    plt.show()


def fit_func(X, a, b, c, d):
    h, v = X
    return a * v * h + b * h + c * v + d

def chute_opening_func(plot=False):
    h_opening_vals = np.linspace(const.min_h_opening, const.max_h_opening, 50)
    v_vals = []
    dist_vals = []
    seeds = np.arange(len(h_opening_vals)) + 1000

    pool = multiprocessing.Pool()
    results = pool.map(chute_opening_simulation, zip(h_opening_vals, seeds))
    pool.close()

    for v, dist in results:
        v_vals.append(v)
        dist_vals.append(dist)

    params, _, = curve_fit(fit_func, (h_opening_vals, v_vals), dist_vals)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        reg_v_vals = np.linspace(min(v_vals), max(v_vals), len(h_opening_vals))
        X, Y = np.meshgrid(h_opening_vals, reg_v_vals)

        ax.scatter(h_opening_vals, v_vals, dist_vals, label='data')
        ax.plot_surface(X, Y, fit_func((X, Y), *params))
        ax.set_xlabel('h_opening')
        ax.set_ylabel('current velocity')
        ax.set_zlabel('distance covered')
        plt.show()

    print(f'{params = }')

    return lambda X : fit_func(X, *params)


def chute_opening_simulation(params):
    h_opening, seed = params
    pos = np.array([0, 0, const.h_plane])
    v = np.array([const.v_plane, 0, 0])

    wind = Wind_generator()
    myDiver = Diver(pos, v, wind, H_VAL, seed=seed, h_opening=h_opening)
    myDiver.simulate_trajectory()

    for x, y, z in myDiver.x_list:
        w_x = myDiver.wind_x(z)
        w_y = myDiver.wind_y(z)
        if z < h_opening:
            x_end, y_end = myDiver.x_list[-1][:2]
            distance = np.sqrt((x - x_end) ** 2 + (y - y_end) ** 2)
            v = np.sqrt(w_x ** 2 + w_y ** 2)
            return v, distance

# def plot_params(x_vals, y_vals, dir_vals, w_x_bounds, w_y_bounds):
#     seeds = np.arange(len(x_vals) * len(y_vals) * len(dir_vals) * NR_OF_SIMS)
#     seeds = np.reshape(seeds, (-1, NR_OF_SIMS))
#     params_list = [[x, y, d, w_x_bounds, w_y_bounds]
#                    for x in x_vals for y in y_vals for d in dir_vals]

#     params_list = [params + list(r) for params, r in zip(params_list, seeds)]
#     pool = multiprocessing.Pool()
#     landing_locations = pool.map(simulate_params, params_list)
#     pool.close()

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     x_list, y_list, z_list, p_list = [], [], [], []
#     for x, y, d, _, _, locations in landing_locations:
#         p = len(1 for x, y in locations \
#                 if x ** 2 + y ** 2 < const.radius_landing_area ** 2)
#         x_list.append(x)
#         y_list.append(y)
#         z_list.append(d)
#         p_list.append(p)

#     plot = ax.scatter(x_list, y_list, z_list, c=p_list, cmap=plt.cm.RdYlGn,
#                       vmin=0, vmax=1)
#     ax.set_xlabel('x-position')
#     ax.set_ylabel('y-position')
#     ax.set_zlabel('approach angle')

#     fig.colorbar(plot, ax=ax)

#     plt.show()


# if __name__ == '__main__':
#     dir_vals = np.linspace(0, 2 * np.pi, NUMBER_D)
#     x_vals = np.linspace(-1000, -500, NUMBER_X)
#     y_vals = np.linspace(0, 0, NUMBER_Y)
#     simulate_params(x_vals, y_vals, dir_vals)
