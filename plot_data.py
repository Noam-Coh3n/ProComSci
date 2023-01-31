#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Double Bachelor mathematics and computer science.
#
# plot_data.py:
# This file plots the dataset, determines the average value
# and standard deviation.

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import constants as const


def fit_func(x: float, a: float, b: float, c: float) -> float:
    return a * x**2 + b * x + c


def plot_data(data, data_variable=['wind', 'rho']):
    """Fits the data with a quadratic function and generates a plot."""

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


def avg_and_dev_fitter(x_vals, y_vals):
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

    # Calculate the average and standard deviation.
    avg = [sum(y for (_, y) in bin) / len(bin) if bin else 0 for bin in bins]
    dev = [np.sqrt(sum((y - avg[i])**2 for (_, y) in bin) / (len(bin) - 1))
           if len(bin) >= 2 else 0 for i, bin in enumerate(bins)]

    # Fit the average and standard deviation using a quadratic polynomial.
    params_avg, _ = curve_fit(fit_func, reg_x_vals, avg)
    params_dev, _ = curve_fit(fit_func, reg_x_vals, dev)

    avg_fitter = lambda h: fit_func(h, *params_avg)
    dev_fitter = lambda h: fit_func(h, *params_dev)

    return reg_x_vals, avg, dev, (avg_fitter, dev_fitter)


def plot_and_fit(x_vals: list, y_vals: list, xlabel: str = '',
                 ylabel: str = '', title: str = '',
                 only_plot_fitted: bool = False) -> None:
    """This function does the following:
    - Plot the given data as a scatter plot.
    - Plot the average value and standard deviation at every point.
    - Plot the average value fitted to a quadratic polynomial.
    - Plot the standard deviation fitted to a quadratic polynomial.
    """

    # Plot the dataset.
    plt.scatter(x_vals, y_vals, alpha=0.4, s=1,
                c=const.color_dataset, label='observed data')

    reg_x_vals, avg, dev, fitters = avg_and_dev_fitter(x_vals, y_vals)
    avg_fitter, dev_fitter = fitters

    fitted_y_vals = avg_fitter(reg_x_vals)
    fitted_dev_vals = dev_fitter(reg_x_vals)

    # Plot the standard deviation and average.
    if not only_plot_fitted:
        plt.plot(reg_x_vals,
                        np.array(avg) - np.array(dev),
                        np.array(avg) + np.array(dev),
                        alpha=0.7, color=const.color_dev, label='std dev')
        plt.plot(reg_x_vals, avg, color=const.color_avg, label='avg values')

    # Plot the quadratic polynomials corresponding to the avg and std dev.
    # plt.plot(reg_x_vals, fitted_y_vals, color=const.color_fitted_avg,
    #         label='avg value fit: quadratic')
    plt.fill_between(reg_x_vals, fitted_y_vals - fitted_dev_vals,
                     fitted_y_vals + fitted_dev_vals,
                     alpha=0.2, color=const.color_fitted_dev, label='std dev fit: quadratic')
    # plot_avg_and_dev(reg_x_vals, bins)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend()
