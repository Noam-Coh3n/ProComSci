from constants import *
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from diver import Diver
import matplotlib

NR_OF_SIMS = 5
H_VAL = 0.01


def simulate_params(params):
    x, y, d = params
    pos = np.array([x, y, h_airplane])
    vel = np.array([np.cos(d) * v_airplane, np.sin(d) * v_airplane, 0])

    nr_of_successes = 0
    for _ in range(NR_OF_SIMS):
        myDiver = Diver(x=pos, vel=vel, h_opening=h_opening, stepsize=H_VAL)
        myDiver.simulate_trajectory('RK4')
        x, y, _ = myDiver.x
        if x ** 2 + y ** 2 < radius_landing_area:
            nr_of_successes += 1


    return params + [nr_of_successes / NR_OF_SIMS]


def find_optimal_params(dir_vals):
    params_list = list([[x, y, d] for x in x_vals for y in y_vals for d in dir_vals])
    pool = multiprocessing.Pool()
    succes_probs = pool.map(simulate_params, params_list)
    pool.close()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x_list, y_list, z_list, p_list = [], [], [], []
    for i in succes_probs:
        x, y, d, p = i
        x_list.append(x)
        y_list.append(y)
        z_list.append(d)
        p_list.append(p)

    cmap = matplotlib.cm.get_cmap('RdYlGn')
    ax.scatter(x_list, y_list, z_list, c=cmap(p_list))
    ax.set_xlabel('x-direction')
    ax.set_ylabel('y-direction')
    ax.set_zlabel('z-direction')

    plt.show()


if __name__ == '__main__':
    dir_vals = np.linspace(0, 2 * np.pi, 1)
    x_vals = np.linspace(0, 2000, 1)
    y_vals = np.linspace(0, 500, 2)
    find_optimal_params(dir_vals)