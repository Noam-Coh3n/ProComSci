from diver import Diver
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import constants as const
from wind_and_rho_generator import Wind_generator

# Define parameters used for the diver and define the methods that will be
# comperd.
h = 0.01
methods = ['RK4', 'Euler', 'Central diff', 'Pred-corr']
nr_of_seeds = 1


def simulate_method(params):
    method, h_vals, y_parts, wind = params
    sum_errors = []
    for h in h_vals:

        total_error = 0
        for seed, y_part in enumerate(y_parts):
            # Add a diver.
            myDiver = Diver(x=np.array([0., 0., const.h_airplane]),
                            velocity=np.array([const.v_airplane, 0., 0.]),
                            wind=wind, stepsize=h, seed=seed)
            # Run model with the different models and stepsizes.
            myDiver.simulate_trajectory(method)

            # Store the data.
            y_n = [x[0] for x in myDiver.x_list]
            step = len(myDiver.x_list) / 1000

            # Determine the error and take the sum
            y_n_part = [y_n[int(np.floor(l))]
                        for l in np.arange(0, len(y_n), step)
                        if int(np.floor(l)) != len(y_n)]
            total_error += sum(np.abs(np.array(y_part) - np.array(y_n_part)))
        sum_errors.append(total_error / len(y_parts))
    return sum_errors


def simulate_control_experiment(params):
    seed, wind = params
    # Get diver data with stepsize equal to h.
    myDiver = Diver(x=np.array([0., 0., const.h_airplane]),
                    velocity=np.array([const.v_airplane, 0, 0]),
                    wind=wind, stepsize=h, seed=seed)

    # Simulate the diver with Runge-kutta order 4
    myDiver.simulate_trajectory('RK4')

    # Get the positions
    y = [x[0] for x in myDiver.x_list]
    step_size = len(myDiver.x_list) / 1000
    return [y[int(np.floor(i))] for i in np.arange(0, len(y), step_size)
            if int(np.floor(i)) != len(y)]


def simulate_error(h_vals):
    """
    Simulate the error of the methods Runge-kutta order 4, Euler, Central
    difference and Predictor-corrector. All the methods will be compared to the
    Runge-kutta order 4 with stepsize 0.001 (This is the good simulation).
    """

    wind = Wind_generator()

    pool = multiprocessing.Pool()
    y_parts = pool.map(simulate_control_experiment, enumerate([wind] * nr_of_seeds))
    pool.close()

    pool = multiprocessing.Pool()
    sum_errors = pool.map(simulate_method, [(method, h_vals, y_parts, wind) for method in methods])
    pool.close()

    # Plot the sum of the error.
    plt.figure("Error methods.")
    for method, sum_error in zip(methods, sum_errors):
        plt.plot(h_vals, sum_error,label=method)
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Step size")
    plt.ylabel("Sum of error")
    plt.show()

