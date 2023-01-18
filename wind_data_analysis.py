#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Dubbel Bachelor mathematics and Informatica.
#
# wind_data_analysis.py:
# This file does the following:
# - It can compress the file size of the dataset by removing all entries
#   with missing values or above a certain height.
# - It can retrieve the air density and windvelocities of Las Vegas
#   at any day from 2018 to 2023 from a dataset.
# - It can plot the air density and windvelocities and calculate
#   the average and standard deviation.
# - It can fit the average of the air density and windvelocities
#   to a quadratic polynomial.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from os import path
import re

from constant import m_air, k_B_air

# Define colors for the plots.
COLOR_AVG = '#319600'
COLOR_STD_DEV = '#00696b'
COLOR_FITTED_AVG = '#1c0f00'
COLOR_FITTED_STD_DEV = '#700000'
COLOR_DATASET = '#02006b'

# Numbers in the dataset that indicate a corrupt entry.
CORRUPT_VALUES = {-8888, -9999}

def density(pressure : float, temperature : float) -> float:
    """
    Calculates the air density from the air pressure and temperature.
    """
    return (pressure * m_air) / (k_B_air * temperature)

def fit_func(x, a, b, c):
    """
    Quadratic function.
    """
    return a * x ** 2 + b * x + c

def clean_up_data_line(line):
    """
    Given a line of data from a dataset, this function
    returns the height and a string containing the height, pressure,
    temperature winddirection and windvelocity seperated by a space.
    """

    # Remove all letters and double spaces.
    line = re.sub('[A-Z]', ' ', line)
    line = re.sub(' +', ' ', line)

    # Split the line and return the data if none of the values are corrupt.
    _, _, pressure, height, temp, _, _, dir, vel = line.split(' ')
    if CORRUPT_VALUES.isdisjoint({int(pressure), int(height), int(temp), int(dir), int(vel)}):
        return int(height), ' '.join([height, pressure, temp, dir, vel])
    return None, None

def remove_data_above_height(max_height):
    """
    Removes all lines of data from the Las Vegas wind data set that
    are measured at a height above the max_height.
    """

    # Open the file with the dataset and a new file.
    data_filename = 'vegas_wind_data.txt'
    if not path.exists(data_filename):
        print(f'{data_filename} does not exist.')
        return None
    data_file = open(data_filename, 'r')
    new_data_file = open('wind_data.txt', 'w')

    # Only write to the new file if it was a header or below the max_height.
    for line in data_file:
        # A header in the file, chars 13 to 23 indicate the yyyy-mm-dd.
        if line[0] == '#':
            new_data_file.write(f'#\n# {line[13:23]} #\n')
            continue

        height, line_data = clean_up_data_line(line)

        # No corrupt entries and below the max_height.
        if height and line[0] != '3' and height < max_height:
            new_data_file.write(line_data)

def retrieve_data_from_dates(dates):
    """
    Returns the air density (rho) and wind data of the specified dates.
    """
    result_data = []
    raw_data = str(open('wind_data.txt', 'r').read())

    # Extends the result_data with the data of all the specified dates.
    for (y, m, d) in dates:
        data = re.findall(f'# {y} {m:02} {d:02} #\n([\s\S]*?)#', raw_data)
        result_data.extend(retrieve_rho_and_wind(data))

    # Combines all the data to get 4 lists: height, rho, w_x, w_y.
    data = [[], [], [], []]
    for item in result_data:
        for i in range(4):
            data[i].extend(item[i])
    return data

def retrieve_rho_and_wind(data):
    result = []
    for data_item in data:
        data_item = np.fromstring(data_item, sep=' ')
        data_item = np.reshape(data_item, (-1, 5))

        height, press, temp, dir, vel = data_item.transpose()
        height -= 697 # Natural elevation of Las Vegas.
        rho = density(press, temp / 10 + 273)
        radials = dir * 2 * np.pi / 360 # Convert deg to radials
        w_x = np.sin(radials) * vel / 10
        w_y = np.cos(radials) * vel / 10
        result.append(np.vstack((height, rho, w_x, w_y)))

    return result

def plot_data_from_date(*date):
    height, rho, w_x, w_y = retrieve_data_from_dates([date])
    plt.figure(figsize=(10, 8), dpi=100)
    plt.subplot(121)
    plot_rho(height, rho)
    plt.subplot(122)
    plot_wind(height, w_x, w_y)
    plt.show()

def all_dates():
    return [(y, m, d) for y in range(2018, 2023) \
                      for m in range(1, 13) \
                      for d in range(1, 32)]

def plot_and_fit(x_vals, y_vals, xlabel='', ylabel='', title=''):
    stepsize = 50
    reg_x_vals = np.arange(min(x_vals), max(x_vals), stepsize)

    nr_of_bins = int(np.ceil(max(x_vals) / stepsize))
    bins = [[] for _ in range(nr_of_bins)]
    for x, y in zip(x_vals, y_vals):
        index = int(np.floor(x / stepsize))
        bins[index].append((x, y))

    # Plot the dataset.
    plt.scatter(x_vals, y_vals, alpha=0.4, s=1, c=COLOR_DATASET, label='observed data')

    # Calculate the average and standard deviation.
    avg = [sum(y for (_, y) in bin) / len(bin) for bin in bins]
    std_dev = [np.sqrt(sum((y - avg[i])**2 for (_, y) in bin) / (len(bin) - 1)) for i, bin in enumerate(bins)]

    # Fit the average and standard deviation using a quadratic polynomial.
    params_avg, _ = curve_fit(fit_func, reg_x_vals, avg)
    params_std_dev, _ = curve_fit(fit_func, reg_x_vals, std_dev)

    fitted_y_vals = fit_func(reg_x_vals, *params_avg)
    fitted_std_dev_vals = fit_func(reg_x_vals, *params_std_dev)

    # Plot the standard deviation and average.
    plt.fill_between(reg_x_vals,
                     np.array(avg) - std_dev,
                     np.array(avg) + std_dev,
                     alpha=0.7, color=COLOR_STD_DEV, label='std dev')
    plt.plot(reg_x_vals, avg, color=COLOR_AVG, label='avg values')

    # Plot the quadratic polynomials fitted to the average and standard deviation.
    plt.plot(reg_x_vals, fitted_y_vals, color=COLOR_FITTED_AVG,
             label='avg value fit: quadratic')
    plt.plot(reg_x_vals, fitted_y_vals + fitted_std_dev_vals,
             color=COLOR_FITTED_STD_DEV, label='std dev fit: quadratic')
    plt.plot(reg_x_vals, fitted_y_vals - fitted_std_dev_vals,
             color=COLOR_FITTED_STD_DEV)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def plot_rho(height, rho):
    plot_and_fit(height, rho, xlabel='height', ylabel=r'$\rho (kg / m^3)$',
                 title=r'The air density $\rho$')

def plot_wind(height, vel, dir):
    plot_and_fit(height, vel, xlabel='height', ylabel=r'$v (m/s)$',
                 title=rf'Wind speed in the ${dir}$ direction')

def average_data(plot=False):
    data = retrieve_data_from_dates(all_dates())

    if plot:
        plt.figure(figsize=(14, 8), dpi=100)
        plt.subplot(131)
        plot_rho(data[0], data[1])
        plt.subplot(132)
        plot_wind(data[0], data[2], 'x')
        plt.subplot(133)
        plot_wind(data[0], data[3], 'y')
        plt.tight_layout()
        plt.show()
    return data

if __name__ == '__main__':
    # remove_data_above_height(5500)
    # plot_data_from_date(2018, 1, 1)
    # print(retrieve_rho_and_wind(2018, 1, 1))
    average_data(plot=True)


