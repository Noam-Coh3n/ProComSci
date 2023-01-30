import constants as const
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from diver import Diver
from wind_and_rho_generator import Wind_generator

NR_OF_SIMS = 5
H_VAL = 0.05
# NUMBER_X = 3
# NUMBER_Y = 3
# NUMBER_D = 1

# W_X_BOUNDS = (-np.inf, -2)
# W_Y_BOUNDS = (-np.inf, -2)


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
    d, w_x_bounds, w_y_bounds = params
    params = [0, 0, d, w_x_bounds, w_y_bounds, *range(50)]
    locations = simulate_params(params)[5:]
    avg = sum(np.array(locations)) / len(locations)
    dev = np.sqrt(sum((loc - avg) ** 2 / (len(locations) - 1)
                      for loc in locations))
    return avg, dev

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
