import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from wind_data_analysis import retrieve_data_combined, change_of_wind
from plot_data import avg_and_dev_fitter

INCREASE_RATES = [0.600752594027682, 0.5729401464205213]
AVG_H_DIFF = 247


def wind_data_functions(w_x_bounds=None, w_y_bounds=None):
    h, w_x, w_y, _ = retrieve_data_combined(w_x_bounds, w_y_bounds)
    w_x_avg, w_x_dev = avg_and_dev_fitter(h, w_x)[-1]
    w_y_avg, w_y_dev = avg_and_dev_fitter(h, w_y)[-1]

    h, c_x, c_y, inc_rates, avg_h_diff = change_of_wind(w_x_bounds, w_y_bounds)
    c_x_avg, c_x_dev = avg_and_dev_fitter(h, c_x)[-1]
    c_y_avg, c_y_dev = avg_and_dev_fitter(h, c_y)[-1]

    return [[w_x_avg, w_y_avg], [w_x_dev, w_y_dev],
            [c_x_avg, c_y_avg], [c_x_dev, c_y_dev]], \
        inc_rates, int(np.floor(avg_h_diff))


def rho(h):
    return 1.70708e-09 * h ** 2 - 9.65846e-05 * h + 1.10647e+00


class Wind_generator:

    def __init__(self, w_x_bounds=None, w_y_bounds=None):
        funcs, a, b = wind_data_functions(w_x_bounds, w_y_bounds)
        self.inc_rates, self.stepsize = a, b

        # Define the average and standard deviation functions.
        self.w_avg, self.w_dev = funcs[0:2]
        self.c_avg, self.c_dev = funcs[2:]

        self.w_lower = [lambda h: self.w_avg[0](h) - 2 * self.w_dev[0](h),
                        lambda h: self.w_avg[1](h) - 2 * self.w_dev[1](h)]

        self.w_upper = [lambda h: self.w_avg[0](h) + 2 * self.w_dev[0](h),
                        lambda h: self.w_avg[1](h) + 2 * self.w_dev[1](h)]

    def wind(self, seed=None, wind_dir='x'):
        if seed is not None:
            np.random.seed(seed)

        i = 0 if wind_dir == 'x' else 1
        wind_heights = np.arange(0, h_plane, self.stepsize)
        wind = [0] * len(wind_heights)

        s = np.random.normal(self.w_avg[i](0), self.w_dev[i](0))
        while s < self.w_lower[i](0) or s > self.w_upper[i](0):
            s = np.random.normal(self.w_avg[i](0), self.w_dev[i](0))
        wind[0] = s

        for h in wind_heights[:-1]:
            cur_wind = wind[int(h / self.stepsize)]
            mu = sum(self.c_avg[i](np.arange(h, h + self.stepsize)))

            sigma = sum(self.c_dev[i](x)
                        for x in range(h, h + self.stepsize))

            increase = 1 if np.random.binomial(1, INCREASE_RATES[i]) else -1

            s = increase * np.abs(np.random.normal(mu, sigma))
            cur_wind += s if np.sign(cur_wind) == 1 else -s
            while not (self.w_lower[i](0) <= cur_wind <= self.w_upper[i](0)):
                cur_wind -= s if np.sign(cur_wind) == 1 else -s
                s = increase * np.abs(np.random.normal(mu, sigma))
                cur_wind += s if np.sign(cur_wind) == 1 else -s

            wind[int(h / self.stepsize) + 1] = cur_wind

        return CubicSpline(wind_heights, wind)

    def plot_wind(self, seed=None, wind_dir='x'):
        wind_func = self.wind(seed=seed, wind_dir=wind_dir)
        h_vals = np.arange(0, h_plane, 10)
        plt.plot(h_vals, wind_func(h_vals))
        plt.xlabel(r'h (m)')
        plt.ylabel(r'$v (m/s)$')
        plt.title(f'Wind in {dir}-direction.')
        plt.show()


if __name__ == '__main__':
    wind = Wind_generator()
    wind.plot_wind(wind_dir='x')
