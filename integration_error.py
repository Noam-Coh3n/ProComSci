from diver import Diver
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

# Define parameters used for the diver and define the methods that will be
# compared.
h = 0.0001
methods = ['rk4', 'euler', 'central diff', 'pred-corr']


def simulate_method(params):
    method, h_vals, y_part = params
    sum_error = []
    for h in h_vals:
        # Add a diver.
        myDiver = Diver(x=np.array([0., 0., 3600.]),
                        vel=np.array([45., 0., 0.]),
                        h_opening=200, stepsize=h, seed=0)
        # Run model with the different models and stepsizes.
        myDiver.simulate_trajectory(method)

        # Store the data.
        y_n = [x[0] for x in myDiver.x_list]
        step = len(myDiver.x_list) / 1000

        # Determine the error and take the sum
        y_n_part = [y_n[int(np.floor(l))]
                    for l in np.arange(0, len(y_n), step)
                    if int(np.floor(l)) != len(y_n)]
        sum_error.append(sum(np.abs(np.array(y_part) -
                                    np.array(y_n_part))))
    return sum_error


def simulate_error(h_vals):
    """
    Simulate the error of the methods Runge-kutta order 4, Euler, Central
    difference and Predictor-corrector. All the methods will be compared to the
    Runge-kutta order 4 with stepsize 0.001 (This is the good simulation).
    """
    # Get diver data with stepsize 0.001.
    myDiver = Diver(x=np.array([0., 0., 3600.]),
                    vel=np.array([45., 0., 0.]),
                    h_opening=200, stepsize=h, seed=0)

    # Simulate the diver with Runge-kutta order 4
    myDiver.simulate_trajectory('RK4')

    # Get the positions
    y = [x[0] for x in myDiver.x_list]
    step_size = len(myDiver.x_list) / 1000
    y_part = [y[int(np.floor(i))] for i in np.arange(0, len(y), step_size)
              if int(np.floor(i)) != len(y)]

    pool = multiprocessing.Pool()
    sum_errors = pool.map(simulate_method, [(method, h_vals, y_part) for method in methods])
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

