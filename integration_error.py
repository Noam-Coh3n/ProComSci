from diver import Diver
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import constants as const
from wind_and_rho_generator import Wind_generator

# Define the methods that will be compared
methods = ['rk4', 'euler', 'central diff', 'pred-corr']
nr_of_seeds = 5


def simulate_method(params):
    method, h_vals, y_parts = params
    wind = Wind_generator()
    sum_errors = []
    for h in h_vals:

        total_error = 0
        for seed, y_part in enumerate(y_parts):
            # Add a diver.
            x = np.array([0., 0., const.h_plane])
            velocity = np.array([const.v_plane, 0., 0.])
            myDiver = Diver(x, velocity, wind, h, method, seed)

            # Run model with the different models and stepsizes.
            myDiver.simulate_trajectory()

            # Store the data.
            # y_n = [x[0] for x in myDiver.x_list]
            # step = len(myDiver.x_list) / 1000

            # Determine the error and take the sum
            # y_n_part = [y_n[int(np.floor(l))]
            #             for l in np.arange(0, len(y_n), step)
            #             if int(np.floor(l)) != len(y_n)]
            y_n_part = np.array(myDiver.x_list[-1][:2])
            # total_error += sum(np.abs(np.array(y_part) - np.array(y_n_part)))
            total_error += np.linalg.norm(y_part - y_n_part)
        sum_errors.append(total_error / len(y_parts))
    return sum_errors


def simulate_control_experiment(seed):
    wind = Wind_generator()
    # Get diver data with stepsize equal to 0.001 and the Runge Kutta method.
    x = np.array([0., 0., const.h_plane])
    velocity = np.array([const.v_plane, 0., 0.])
    myDiver = Diver(x, velocity, wind, 0.001, 'rk4', seed)

    # Simulate the diver
    myDiver.simulate_trajectory()

    # Get the positions
    # y = [x[0] for x in myDiver.x_list]
    # step_size = len(myDiver.x_list) / 1000
    # return [y[int(np.floor(i))] for i in np.arange(0, len(y), step_size)
    #     if int(np.floor(i)) != len(y)]
    return myDiver.x_list[-1][:2]


def simulate_error(h_vals):
    """Simulate the error of the methods Runge-kutta order 4, Euler, Central
    difference and Predictor-corrector. All the methods will be compared to the
    Runge-kutta order 4 with stepsize 0.001 (This is the good simulation).
    """

    pool = multiprocessing.Pool()
    # params = list(enumerate([wind] * nr_of_seeds))
    params = np.arange(nr_of_seeds)
    y_parts = pool.map(simulate_control_experiment, params)
    pool.close()

    pool = multiprocessing.Pool()
    sum_errors = pool.map(simulate_method,
                          [(method, h_vals, y_parts) for method in methods])
    pool.close()

    # Plot the sum of the error.
    plt.figure("Error methods")
    for method, sum_error in zip(methods, sum_errors):
        plt.plot(h_vals, sum_error, label=method)
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Step size")
    plt.ylabel("Global error")
    plt.title("Integration error of different methods")
    plt.show()
