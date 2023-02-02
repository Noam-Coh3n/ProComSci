#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Double Bachelor mathematics and computer science.
#
# wind_and_rho_generator.py:
# This file can give the air density (rho) at different heights and has a
# Wind_generator class. This class can be used to generate wind where the
# model will only consider wind measurements with a specific wind speed
# at ground level.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from wind_data_analysis import change_of_wind, retrieve_data_combined
from plot_data import avg_and_dev_fitter
import constants as const


def wind_data_functions(w_x_bounds=const.w_x_bounds,
                        w_y_bounds=const.w_y_bounds) -> tuple:
    """Returns the average and standard deviation of the wind speed and
    change in wind speed. Furthermore, it returns the probability that the
    wind increases in speed and the average distance in height between
    measurements done by the weather balloon.
    """

    # The average and standard deviation of the wind speed.
    h, w_x, w_y, _ = retrieve_data_combined(w_x_bounds, w_y_bounds)
    w_x_avg, w_x_dev = avg_and_dev_fitter(h, w_x)[-1]
    w_y_avg, w_y_dev = avg_and_dev_fitter(h, w_y)[-1]

    # The average and standard deviation of the change of wind speed.
    h, c_x, c_y, inc_rates, avg_h_diff = change_of_wind(w_x_bounds, w_y_bounds)
    c_x_avg, c_x_dev = avg_and_dev_fitter(h, c_x)[-1]
    c_y_avg, c_y_dev = avg_and_dev_fitter(h, c_y)[-1]

    return [[w_x_avg, w_y_avg], [w_x_dev, w_y_dev],
            [c_x_avg, c_y_avg], [c_x_dev, c_y_dev]], \
        inc_rates, int(np.floor(avg_h_diff))


def rho(h: float) -> float:
    """Returns the air density at a given height."""
    return 1.70708e-09 * h ** 2 - 9.65846e-05 * h + 1.10647e+00


class Wind_generator:

    def __init__(self, w_x_bounds=const.w_x_bounds,
                 w_y_bounds=const.w_y_bounds):
        funcs, a, b = wind_data_functions(w_x_bounds, w_y_bounds)
        self.inc_rates, self.stepsize = a, b

        # Define the average and standard deviation functions.
        self.w_avg, self.w_dev = funcs[0:2]
        self.c_avg, self.c_dev = funcs[2:]

        # Define the minimum and the maximum windspeed at the given height.
        self.w_lower = [lambda h: self.w_avg[0](h) - 1 * self.w_dev[0](h),
                        lambda h: self.w_avg[1](h) - 1 * self.w_dev[1](h)]

        self.w_upper = [lambda h: self.w_avg[0](h) + 1 * self.w_dev[0](h),
                        lambda h: self.w_avg[1](h) + 1 * self.w_dev[1](h)]

    def wind(self, seed=None, wind_dir='x'):
        """Generate wind for the given seed and wind direction."""
        if seed is not None:
            np.random.seed(seed)


        # Initialize wind height list.
        i = 0 if wind_dir == 'x' else 1
        wind_heights = np.arange(0, const.h_plane + self.stepsize,
                                 self.stepsize)
        wind = [0] * len(wind_heights)

        # Determine wind speed at height 0.
        s = np.random.normal(self.w_avg[i](0), self.w_dev[i](0))

        # Resample if new wind speed is not within bounds.
        while s < self.w_lower[i](0) or s > self.w_upper[i](0):
            s = np.random.normal(self.w_avg[i](0), self.w_dev[i](0))
        wind[0] = s


        # Sample a change of wind speed for all heights.
        for h in wind_heights[:-1]:
            cur_wind = wind[int(h / self.stepsize)]
            mu = sum(self.c_avg[i](np.arange(h, h + self.stepsize)))

            sigma = sum(self.c_dev[i](x)
                        for x in range(h, h + self.stepsize))

            # The wind speed increases in absolute value with probability
            # defined in self.inc_rates.
            increase = 1 if np.random.binomial(1, self.inc_rates[i]) else -1

            s = increase * np.abs(np.random.normal(mu, sigma))
            cur_wind += s if np.sign(cur_wind) == 1 else -s

            # Resample if new wind speed is not within bounds.
            while not (self.w_lower[i](h) <= cur_wind <= self.w_upper[i](h)):
                cur_wind -= s if np.sign(cur_wind) == 1 else -s
                s = increase * np.abs(np.random.normal(mu, sigma))
                cur_wind += s if np.sign(cur_wind) == 1 else -s

            wind[int(h / self.stepsize) + 1] = cur_wind

        return CubicSpline(wind_heights, wind)

    def plot_wind(self, seed=None, wind_dir='x'):
        """Plot the wind in the given direcetion."""
        wind_func = self.wind(seed=seed, wind_dir=wind_dir)
        h_vals = np.arange(0, const.h_plane, 10)

        plt.plot(h_vals, wind_func(h_vals))
        plt.xlabel(r'h (m)')
        plt.ylabel(r'$v (m/s)$')
        plt.title(f'Wind in {wind_dir}-direction.')
        plt.show()


if __name__ == '__main__':
    wind = Wind_generator(const.w_x_bounds, const.w_y_bounds)
    wind.plot_wind(wind_dir='x')
