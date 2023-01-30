import numpy as np
from diver import Diver
import matplotlib.pyplot as plt
import integration_error as err
from wind_and_rho_generator import Wind_generator
from wind_data_analysis import retrieve_data_combined
from plot_data import plot_and_fit
import constants as const

STEP_SIZE = 0.005


def visual(myDiver):
    from visual import Visual
    myVisual = Visual()
    myVisual.add_diver(myDiver)
    myVisual.select_line_interval(300)
    myVisual.run(slowed=10)


def plot(myDiver):
    plt.figure(figsize=(16, 10), dpi=100)

    # Plot location.
    for i, direction in enumerate(['x', 'y', 'z']):
        plt.subplot(int(f'33{i+1}'))
        plt.title(f"{direction}-direction, location")
        time = [i * STEP_SIZE for i in range(len(myDiver.x_list))]

        # Location.
        plt.plot(time, [x[i] for x in myDiver.x_list], label="location")

        plt.legend()
        plt.xlabel("time (sec)")

    # Plot velocity and acceleration.
    for i, direction in enumerate(['x', 'y', 'z']):
        plt.subplot(int(f'33{i+4}'))
        plt.title(f"{direction}-direction, velocity and acceleration")
        time = [i * STEP_SIZE for i in range(len(myDiver.x_list))]

        # Location.
        plt.plot(time, [v[i] for v in myDiver.v_list], label="velocity")
        plt.plot(time, [a[i] for a in myDiver.a_list], label="acceleration")

        plt.legend()
        plt.xlabel("time (sec)")

    for i, (dir, func) in enumerate(zip(['x', 'y'],
                                        [myDiver.wind_x, myDiver.wind_y])):
        plt.subplot(int(f'33{i+7}'))
        plt.plot(np.arange(0, const.h_plane, 10),
                 func(np.arange(0, const.h_plane, 10)))
        plt.xlabel(r'height (m)')
        plt.ylabel(r'$v (m/s)$')
        plt.title(f'Wind in {dir}-direction.')

    plt.tight_layout()
    plt.show()


def errors():
    # Setup variables
    h_vals = np.logspace(-3, -1, 10)
    err.simulate_error(h_vals)

def plot_wind(nr_of_sims):
    x_res = (-np.inf, -2)
    y_res = (-np.inf, -2)
    wind = Wind_generator(x_res, y_res)
    data = retrieve_data_combined(w_x_bounds=x_res, w_y_bounds=y_res)

    plt.figure(figsize=(8,8), dpi=150)
    plot_and_fit(data[0], data[1], xlabel='height', ylabel=r'$v (m/s)$',
                 title=r'Wind speed in the $x$ direction with generated wind.',
                 only_plot_fitted=True)

    h_vals = np.arange(0, const.h_plane, 10)
    for _ in range(nr_of_sims):
        wind_func = wind.wind(wind_dir='x')
        plt.plot(h_vals, wind_func(h_vals), 'r')
    plt.show()


if __name__ == '__main__':
    num = int(input('Press 1 for visual, '
                    '2 for plot, '
                    '3 for errors plot, '
                    '4 for wind simulations: '))

    while num not in [1, 2, 3, 4]:
        num = int(input('Please press 1, 2, 3 or 4: '))

    if num in [1, 2]:
        # Initialize a wind_generator.
        wind = Wind_generator()

        # Add a diver.
        x = np.array([0., 0., const.h_plane])
        velocity = np.array([const.v_plane, 0., 0.]),
        myDiver = Diver(x, velocity, wind, STEP_SIZE)

        # Run model.
        myDiver.simulate_trajectory()
    if num == 1:
        visual(myDiver)
    elif num == 2:
        plot(myDiver)
    elif num == 3:
        errors()
    elif num == 4:
        plot_wind(7)
