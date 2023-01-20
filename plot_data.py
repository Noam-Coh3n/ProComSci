#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Double Bachelor mathematics and computer science.
#
# plot_data.py:
# This file plots the dataset, determines the average value
# and standard deviation.

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from IPython.display import Latex
import numpy as np

# Define colors for the plots.
COLOR_AVG = '#319600'
COLOR_STD_DEV = '#00696b'
COLOR_FITTED_AVG = '#1c0f00'
COLOR_FITTED_STD_DEV = '#700000'
COLOR_DATASET = '#02006b'


def fit_func(x: float, a: float, b: float, c: float) -> float:
    """
    Quadratic function.
    """
    return a * x ** 2 + b * x + c


def plot_data(data, data_variable=['wind', 'rho']):
    """
    Fit and plot the data.
    """

    ylabels = [r'$v (m/s)$', r'$v (m/s)$', r'$\rho (kg / m^3)$']
    title_list = []
    if 'wind' in data_variable:
        title_list.extend([r'Wind speed in the $x$ direction',
                           r'Wind speed in the $y$ direction'])
    if 'change' in data_variable:
        title_list.extend([r'Change of wind speed in $x$ direction',
                           r'Change of wind speed in $y$ direction'])
    if 'rho' in data_variable:
        title_list.append(r'The air density $\rho$')

    plt.figure(figsize=(14, 8), dpi=100)
    nr_of_subplots = len(data) - 1

    for i, title in enumerate(title_list, 1):
        # print(i, ylabel, len(title_list))
        plt.subplot(int(f'1{nr_of_subplots}{i}'))
        plot_and_fit(data[0], data[i], xlabel='height', ylabel=ylabels[i-1],
                     title=title)

    plt.tight_layout()
    plt.show()


def plot_avg_and_std_dev(reg_x_vals, bins):
    # Calculate the average and standard deviation.
    avg = [sum(y for (_, y) in bin) / len(bin) for bin in bins]
    std_dev = [np.sqrt(sum((y - avg[i])**2 for (_, y) in bin) / (len(bin) - 1))
               for i, bin in enumerate(bins)]

    # Fit the average and standard deviation using a quadratic polynomial.
    params_avg, _ = curve_fit(fit_func, reg_x_vals, avg)
    params_std_dev, _ = curve_fit(fit_func, reg_x_vals, std_dev)

    fitted_y_vals = fit_func(reg_x_vals, *params_avg)
    fitted_std_dev_vals = fit_func(reg_x_vals, *params_std_dev)

    # Plot the standard deviation and average.
    plt.fill_between(reg_x_vals,
                     np.array(avg) - np.array(std_dev),
                     np.array(avg) + np.array(std_dev),
                     alpha=0.7, color=COLOR_STD_DEV, label='std dev')
    plt.plot(reg_x_vals, avg, color=COLOR_AVG, label='avg values')

    a1, b1, c1 = params_avg
    a2, b2, c2 = params_std_dev

    print(f'Average fitted:  {a1:.5e}h^2+{b1:.5e}h+{c1:.5e}')
    print(f'Std dev fitted:  {a2:.5e}h^2+{b2:.5e}h+{c2:.5e}')
    print()

    # Plot the quadratic polynomials corresponding to the avg and std dev.
    plt.plot(reg_x_vals, fitted_y_vals, color=COLOR_FITTED_AVG,
             label='avg value fit: quadratic')
    plt.plot(reg_x_vals, fitted_y_vals + fitted_std_dev_vals,
             color=COLOR_FITTED_STD_DEV, label='std dev fit: quadratic')
    plt.plot(reg_x_vals, fitted_y_vals - fitted_std_dev_vals,
             color=COLOR_FITTED_STD_DEV)


def plot_and_fit(x_vals: list, y_vals: list, xlabel: str = '',
                 ylabel: str = '', title: str = '') -> None:
    """
    This function does the following:
    - Plot the given data as a scatter plot.
    - Plot the average value and standard deviation at every point.
    - Plot the average value fitted to a quadratic polynomial.
    - Plot the standard deviation fitted to a quadratic polynomial.
    """
    stepsize = 50
    reg_x_vals = np.arange(min(x_vals), max(x_vals), stepsize)
    # Order the data in bins.
    nr_of_bins = int(np.ceil(max(x_vals) / stepsize))
    bins = [[] for _ in range(nr_of_bins)]
    for x, y in zip(x_vals, y_vals):
        index = int(np.floor(x / stepsize))
        if index == nr_of_bins:
            index -= 1
        bins[index].append((x, y))

    # Plot the dataset.
    plt.scatter(x_vals, y_vals, alpha=0.4, s=1,
                c=COLOR_DATASET, label='observed data')

    plot_avg_and_std_dev(reg_x_vals, bins)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()



