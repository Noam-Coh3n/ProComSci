import numpy as np
from diver import Diver
import matplotlib.pyplot as plt
import integration_error as err
from wind_and_rho_generator import Wind_generator
from wind_data_analysis import retrieve_data_combined
from plot_data import plot_and_fit
from optimal_params import chute_opening_func, plot_optimal_params
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
    h_vals = np.logspace(-2, -1, 10)
    err.simulate_error(h_vals)


def plot_wind(nr_of_sims):
    wind = Wind_generator()
    data = retrieve_data_combined()

    plt.figure(figsize=(8, 8), dpi=150)
    plot_and_fit(data[0], data[1], xlabel='height', ylabel=r'$v (m/s)$',
                 title=r'Wind speed in the $x$ direction with generated wind.',
                 only_plot_fitted=True)

    h_vals = np.arange(0, const.h_plane, 10)
    for i in range(nr_of_sims):
        wind_func = wind.wind(wind_dir='x')
        if i == 0:
            plt.plot(h_vals, wind_func(h_vals), '#6D0DD5', label="Generated Wind")
        else:
            plt.plot(h_vals, wind_func(h_vals), '#6D0DD5')
    plt.legend()
    plt.show()


def optimal_params():
    dir_vals = np.linspace(0, 2 * np.pi, 10)
    plot_optimal_params(dir_vals)
    # landing_locs_plot(w_x_bounds=w_x_bounds, w_y_bounds=w_y_bounds)
    pass


if __name__ == '__main__':
    num = int(input('Press 1 for visual, '
                    '2 for trajectory plot, '
                    '3 for errors plot, '
                    '4 for wind simulations, '
                    '5 for optimal params plot, '
                    '6 for parachute opening plot: '))

    while num not in range(1, 7):
        num = int(input('Please press 1, 2, 3, 4, 5 or 6: '))

    if num in [1, 2]:
        # Initialize a wind_generator.
        wind = Wind_generator(const.w_x_bounds, const.w_y_bounds)

        # Add a diver.
        x = np.array([0., 0., const.h_plane])
        velocity = np.array([const.v_plane, 0., 0.])
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
        plot_wind(nr_of_sims=3)
    elif num == 5:
        optimal_params()
    elif num == 6:
        chute_opening_func(plot=True)
