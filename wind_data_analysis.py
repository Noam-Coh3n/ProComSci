#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Double Bachelor mathematics and computer science.
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
from os import path
import re

from constants import m_air, kB
from plot_data import plot_data

# Numbers in the dataset that indicate a corrupt entry.
CORRUPT_VALUES = {-8888, -9999}


def density(pressure: float, temperature: float) -> float:
    """
    Calculates the air density from the air pressure and temperature.
    """
    return (pressure * m_air) / (kB * temperature)


def clean_up_data_line(line: str) -> tuple:
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
    if CORRUPT_VALUES.isdisjoint({int(pressure), int(height),
                                  int(temp), int(dir), int(vel)}):
        return int(height), ' '.join([height, pressure, temp, dir, vel])
    return None, None


def remove_data_above_height(max_height: int) -> None:
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


def retrieve_rho_and_wind(data: list, w_x_bounds, w_y_bounds) -> list:
    """
    Returns the height, air density (rho) and wind data when being given
    the height, pressure, temperature and wind data.
    """
    (x_low, x_high) = w_x_bounds if w_x_bounds else (-np.inf, np.inf)
    (y_low, y_high) = w_y_bounds if w_y_bounds else (-np.inf, np.inf)

    # print(w_x_bounds)
    # print(w_y_bounds)
    # x_low, x_high = w_x_bounds
    # y_low, y_high = w_y_bounds
    # print(x_low, x_high, y_low, y_high)
    # print(data)

    result = []
    for data_item in data:
        if not data_item:
            continue
        # Converts the data string into a numpy array.
        data_item = np.fromstring(data_item, sep=' ')
        data_item = np.reshape(data_item, (-1, 5))

        # Calculate the needed values from the given data.
        height, press, temp, dir, vel = data_item.transpose()
        height -= 697  # Natural elevation of Las Vegas.
        rho = density(press, temp / 10 + 273)
        radials = dir * 2 * np.pi / 360  # Convert deg to radials.

        # Wind.
        w_x = np.sin(radials) * vel / 10
        w_y = np.cos(radials) * vel / 10
        # print(w_x)
        # print(w_y)
        if x_low < w_x[0] and w_x[0] < x_high and \
           y_low < w_y[0] and w_y[0] < y_high:
            result.append(np.vstack((height, w_x, w_y, rho)))

    return result


def retrieve_data_separate(w_x_bounds=None, w_y_bounds=None):
    result_data = []
    raw_data = str(open('wind_data.txt', 'r').read())

    # Extends the result_data with the data of all dates.
    for (y, m, d) in all_dates():
        data = re.findall(rf'# {y} {m:02} {d:02} #\n([\s\S]*?)#', raw_data)
        result_data.extend(retrieve_rho_and_wind(data, w_x_bounds, w_y_bounds))
    return result_data


def retrieve_data_combined(w_x_bounds=None, w_y_bounds=None) -> list:
    """
    Returns the air density (rho) and wind data of the specified dates.
    """

    separate_data = retrieve_data_separate(w_x_bounds, w_y_bounds)

    # Combines all the data to get 4 lists: height, rho, w_x, w_y.
    data = [[], [], [], []]
    for item in separate_data:
        for i in range(4):
            data[i].extend(item[i])
    return data


def all_dates() -> list:
    """
    Return list of all dates from 2018 to 2023.
    """
    return [(y, m, d) for y in range(2018, 2023)
            for m in range(1, 13) for d in range(1, 32)]


def change_of_wind(w_x_bounds=None, w_y_bounds=None):
    height, w_x, w_y, _ = retrieve_data_combined(w_x_bounds, w_y_bounds)

    rate_changes_x = []
    rate_changes_y = []
    new_height = []

    h_diffs = []

    x_increase = 0
    y_increase = 0
    counter = 0

    for cur_height, cur_w_x, cur_w_y in zip(height, w_x, w_y):
        if cur_height == 0:
            prev_height = cur_height
            prev_w_x = cur_w_x
            prev_w_y = cur_w_y
            continue
        h_diff = cur_height - prev_height
        h_diffs.append(h_diff)

        rate_changes_x.append((cur_w_x - prev_w_x) / h_diff)
        rate_changes_y.append((cur_w_y - prev_w_y) / h_diff)

        avg_height = int(np.floor((prev_height + cur_height) / 2))
        new_height.append(avg_height)

        x_increase += 1 if np.abs(prev_w_x) < np.abs(cur_w_x) else 0
        y_increase += 1 if np.abs(prev_w_y) < np.abs(cur_w_y) else 0
        counter += 1

        prev_height = cur_height
        prev_w_x = cur_w_x
        prev_w_y = cur_w_y

    incr_rates = [x_increase / counter, y_increase / counter]
    avg_h_diff = sum(h_diffs) / len(h_diffs)
    return new_height, rate_changes_x, rate_changes_y, incr_rates, avg_h_diff


if __name__ == '__main__':
    begin = 0
    restrictions = [(-np.inf, begin), (begin, np.inf)]
    for x_res in restrictions:
        for y_res in restrictions:
            wind_data = retrieve_data_combined(x_res, y_res)
            plot_data(wind_data)

            h, c_x, c_y, increase_rates, avg_h_diff = change_of_wind()
            # print(f'{increase_rates = }')
            # print(f'{avg_h_diff = }')
            plot_data([h, c_x, c_y], data_variable='change')
