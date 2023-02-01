#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Double Bachelor mathematics and computer science.
#
# integration_error.py:
# Here we run the simulation multiple times for all the different integration
# methods. All the methods will be compared against the Runge-kutta order 4
# with a very small stapsize. This file will give a plot with the global error
# of the different methods. This global error represents the difference in
# landingzones.

from diver import Diver
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import constants as const
from wind_and_rho_generator import Wind_generator

# Define the methods that will be compared
methods = ['rk4', 'euler', 'central diff', 'pred-corr']
nr_of_seeds = 3


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
            y_n_part = np.array(myDiver.x_list[-1][:2])
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
    plt.figure("Error methods", dpi=300)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
    for method, sum_error in zip(methods, sum_errors):
        plt.plot(h_vals, sum_error, label=method)
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Step size")
    plt.ylabel("Global error")
    plt.title("Integration error of different methods")
    plt.show()
