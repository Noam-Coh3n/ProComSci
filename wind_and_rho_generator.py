import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

wind_avg = [lambda h : -9.92173e-08 * h ** 2 - 1.00840e-03 * h - 6.75451e-01,
           lambda h : -6.25094e-08 * h ** 2 + 1.92637e-04 * h - 7.97702e-02]

wind_std_dev = [lambda h : -6.89111e-09 * h ** 2 + 1.35294e-03 * h + 2.97040e+00,
                lambda h : -8.93228e-08 * h ** 2 + 1.61771e-03 * h + 4.55504e+00]

change_avg = [lambda h : 2.90173e-12 * h ** 2 - 4.29020e-07 * h - 1.00713e-03,
              lambda h : 1.51600e-10 * h ** 2 - 6.80284e-07 * h + 9.23514e-04]

change_std_dev = [lambda h : 1.46347e-10 * h ** 2 - 1.06378e-06 * h + 1.06364e-02,
                  lambda h : 5.65310e-10 * h ** 2 - 3.44977e-06 * h + 1.32486e-02]


wind_lowerbound = [lambda h : wind_avg[0](h) - 2 * wind_std_dev[0](h),
                   lambda h : wind_avg[1](h) - 2 * wind_std_dev[1](h)]

wind_upperbound = [lambda h : wind_avg[0](h) + 2 * wind_std_dev[0](h),
                   lambda h : wind_avg[1](h) + 2 * wind_std_dev[1](h)]

INCREASE_RATES = [0.600752594027682, 0.5729401464205213]
AVG_H_DIFF = 247


def rho(h):
    return 1.70708e-09 * h ** 2 - 9.65846e-05 * h + 1.10647e+00

def wind(wind_dir='x', seed=None, max_height=4800,
         height_stepsize=AVG_H_DIFF, plot=False):
    if seed is not None:
        np.random.seed(seed)

    i = 0 if wind_dir == 'x' else 1
    wind_heights = np.arange(0, max_height, height_stepsize)
    wind = [0] * len(wind_heights)

    s = np.random.normal(wind_avg[i](0), wind_std_dev[i](0))
    while s < wind_lowerbound[i](0) or s > wind_upperbound[i](0):
        s = np.random.normal(wind_avg[i](0), wind_std_dev[i](0))
    wind[0] = s


    for h in wind_heights[:-1]:
        cur_wind = wind[int(h / height_stepsize)]
        mu = sum(change_avg[i](np.arange(h,
                                         h + height_stepsize)))

        sigma = sum(change_std_dev[i](x)
                            for x in range(h, h + height_stepsize))

        increase = 1 if np.random.binomial(1, INCREASE_RATES[i]) else -1

        s = increase * np.abs(np.random.normal(mu, sigma))
        cur_wind += s if np.sign(cur_wind) == 1 else -s
        while cur_wind < wind_lowerbound[i](0) or cur_wind > wind_upperbound[i](0):
            cur_wind -= s if np.sign(cur_wind) == 1 else -s
            s = increase * np.abs(np.random.normal(mu, sigma))
            cur_wind += s if np.sign(cur_wind) == 1 else -s


        wind[int(h / height_stepsize) + 1] = cur_wind

    cs = CubicSpline(wind_heights, wind)
    if plot:
        plt.plot(np.arange(0, 4800, 10), cs(np.arange(0, 4800, 10)))
        plt.xlabel(r'height (m)')
        plt.ylabel(r'$v (m/s)$')
        plt.title(f'Wind in {wind_dir}-direction.')
        plt.show()
    return cs


if __name__ == '__main__':
    wind(wind_dir='x', seed=0)
