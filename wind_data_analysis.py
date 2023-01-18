import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from os import path
import re

from constant import m_air, k_B_air

def density(pressure, temperature):
    return (pressure * m_air) / (k_B_air * temperature)

def inv_prop(x, a, b, c):
    return a / (x + b) + c

def clean_up_data_line(line):
    line = re.sub('[A-Z]', ' ', line) # Remove all letters
    line = re.sub(' +', ' ', line) # Remove all double spaces
    _, _, pressure, height, temp, _, _, dir, vel = line.split(' ')
    if {-9999, -8888}.isdisjoint({int(pressure), int(height), int(temp), int(dir), int(vel)}):
        return int(height), ' '.join([height, pressure, temp, dir, vel])
    return None, None

def remove_data_above_height(max_height):
    data_filename = 'vegas_wind_data.txt'
    if not path.exists(data_filename):
        print(f'{data_filename} does not exist.')
        return None
    data_file = open(data_filename, 'r')
    new_data_file = open('wind_data.txt', 'w')

    for line in data_file:
        if line[0] == '#':
            new_data_file.write(f'#\n# {line[13:23]} #\n')
            continue

        height, line_data = clean_up_data_line(line)

        # height = re.search('^[\-0-9]+ *[\-0-9]* *[\-0-9]* *([\-0-9]*)', line)
        if height and line[0] != '3' and height < max_height:
            new_data_file.write(line_data)

def retrieve_data_from_dates(dates, output_rho=True, output_wind=True):
    result_data = []
    raw_data = open('wind_data.txt', 'r')
    raw_data = str(raw_data.read())
    for (y, m, d) in dates:

        data = re.findall(f'# {y} {m:02} {d:02} #\n([\s\S]*?)#', raw_data)
        result_data.extend(retrieve_rho_and_wind(data, output_rho, output_wind))
    return result_data

def retrieve_rho_and_wind(data, output_rho=True, output_wind=True):
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
        return_data = np.vstack((height, rho)) if output_rho else height
        result.append(np.vstack((return_data, w_x, w_y)) if output_wind else return_data)

    return result

def plot_data_from_date(*date):
    height, rho, w_x, w_y = retrieve_rho_and_wind(*date)[0]
    plot_rho(height, rho)
    plot_wind(height, w_x, w_y)

def all_possible_dates():
    return [(y, m, d) for y in range(2018, 2023) \
                      for m in range(1, 13) \
                      for d in range(1, 32)]

def average_rho_of_all_days():
    items = retrieve_data_from_dates(all_possible_dates(), output_wind=False)
    height_lst = []
    rho_lst = []
    for (height, rho) in items:
        height_lst.extend(height)
        rho_lst.extend(rho)
    plot_rho(height_lst, rho_lst, scatter=True)
    return height_lst, rho

def plot_rho(height, rho, scatter=False):
    params, _ = curve_fit(inv_prop, height, rho)
    x_values = np.linspace(0, 5000, 10)
    plt.plot(x_values, inv_prop(x_values, *params), 'b', label='fitted values')
    if scatter:
        plt.scatter(height, rho, s=1, c='black')
    else:
        plt.plot(height, rho, 'r', label='observed values')
    plt.title(r'The value of the air density $\rho$')
    plt.xlabel('height')
    plt.ylabel(r'$\rho (kg / m^3)$')
    plt.legend()
    plt.show()

def plot_wind(height, w_x, w_y):
    plt.plot(height, w_x, 'r', label=r'$w_x$')
    plt.plot(height, w_y, 'b', label=r'$w_y$')
    plt.title(r'The value of the wind in the $x$ and $y$ direction')
    plt.xlabel('height')
    plt.ylabel(r'$v (m/s)$')
    plt.show()

if __name__ == '__main__':
    # remove_data_above_height(5500)
    # plot_data_from_date(2018, 1, 1)
    # print(retrieve_rho_and_wind(2018, 1, 1))
    average_rho_of_all_days()

