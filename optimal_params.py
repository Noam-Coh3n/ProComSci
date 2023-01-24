import constants as const
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from diver import Diver
from wind_and_rho_generator import Wind_generator

NR_OF_SIMS = 5
H_VAL = 0.01
NUMBER_X = 8
NUMBER_Y = 1
NUMBER_D = 1


def simulate_params(params):
    x, y, d = params[:3]
    pos = np.array([x, y, const.h_plane])
    v = np.array([np.cos(d) * const.v_plane, np.sin(d) * const.v_plane, 0])
    wind = Wind_generator((0, np.inf), (0, np.inf))

    nr_of_successes = 0
    for seed in params[3:]:
        myDiver = Diver(pos, v, wind, H_VAL, seed)
        myDiver.simulate_trajectory('rk4')
        x, y, _ = myDiver.x
        if x ** 2 + y ** 2 < const.radius_landing_area ** 2:
            nr_of_successes += 1

    return params[:3] + [nr_of_successes / NR_OF_SIMS]


def find_optimal_params(x_vals, y_vals, dir_vals):
    seeds = np.arange(NUMBER_Y * NUMBER_D * NUMBER_X * NR_OF_SIMS)
    seeds = np.reshape(seeds, (-1, 5))
    params_list = [[x, y, d] for x in x_vals for y in y_vals for d in dir_vals]

    params_list = [params + list(r) for params, r in zip(params_list, seeds)]
    pool = multiprocessing.Pool()
    succes_probs = pool.map(simulate_params, params_list)
    pool.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_list, y_list, z_list, p_list = [], [], [], []
    for item in succes_probs:
        x, y, d, p = item
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


if __name__ == '__main__':
    dir_vals = np.linspace(0, 2 * np.pi, NUMBER_D)
    x_vals = np.linspace(-1000, -500, NUMBER_X)
    y_vals = np.linspace(0, 0, NUMBER_Y)
    find_optimal_params(x_vals, y_vals, dir_vals)
