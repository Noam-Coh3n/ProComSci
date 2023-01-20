import integration_methods as integration
from diver import Diver
import numpy as np
import matplotlib.pyplot as plt

# Define parameters used for the diver and define the methods that will be
# comperd.
h = 0.001
wind_list = np.array([[-30, 0], [10, 0], [30, 60]])
air_pres_list = [100000]
temp_list = [280]
method = ['RK4', 'Euler', 'Central diff', 'Pred-corr']


def simulate_error(h_vals):
    """
    Simulate the error of the methods Runge-kutta order 4, Euler, Central
    difference and Predictor-corrector. All the methods will be compared to the
    Runge-kutta order 4 with stepsize 0.001 (This is the good simulation).
    """
    # Get diver data with stepsize 0.001.
    myDiver = Diver(x=np.array([0., 0., 3600.]),
                    vel=np.array([0., -600., 0.]),
                    wind=wind_list, air_pressure=air_pres_list,
                    temperature=temp_list, h_shute=200, stepsize=h)

    # Simulate the diver with Runge-kutta order 4
    myDiver.simulate_trajectory('RK4')

    # Get the positions
    y = [x[0] for x in myDiver.x_list]
    step_size = len(myDiver.x_list) / 1000
    y_part = [y[int(np.floor(i))] for i in np.arange(0, len(y), step_size)
              if int(np.floor(i)) != len(y)]

    # Loop through all the methods and loop through all the stepsize values.
    for i in method:
        sum_error = []
        for j in h_vals:
            # Add a diver.
            myDiver = Diver(x=np.array([0., 0., 3600.]),
                            vel=np.array([0., -600., 0.]),
                            wind=wind_list, air_pressure=air_pres_list,
                            temperature=temp_list, h_shute=200, stepsize=j)
            # Run model with the different models and stepsizes.
            myDiver.simulate_trajectory(i)

            # Store the data.
            y_n = [x[0] for x in myDiver.x_list]
            step = len(myDiver.x_list) / 1000

            # Determine the error and take the sum
            y_n_part = [y_n[int(np.floor(l))]
                        for l in np.arange(0, len(y_n), step)
                        if int(np.floor(l)) != len(y_n)]
            sum_error.append(sum(np.abs(np.array(y_part) -
                                 np.array(y_n_part))))
        # Plot the sum of the error.
        plt.figure("Error methods.")
        plt.plot(h_vals, sum_error, label=i)
        plt.legend()
        plt.xlabel("Step size")
        plt.ylabel("Sum of error")
        plt.yscale("log")
    plt.show()