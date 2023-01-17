import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re

from convert_data import convert_data

def inv_prop(x, a, b, c):
    return a / (x + b) + c

def remove_data_above_height(height):
    data_file = open('vegas_wind_data.txt', 'r')
    new_data_file = open('vegas_compressed.txt', 'w')
    for line in data_file:
        line = re.sub('[A-Z]', ' ', line)
        height = re.search('^[\-0-9]+ *[\-0-9]* *[\-0-9]* *([\-0-9]*)', line)
        if line[0] == '#' \
           or (line[0] != '3' and height and int(height.group(1)) < 4000):
            new_data_file.write(line)

def retrieve_rho_and_wind():
    data_file = open('vegas_wind_data.txt', 'r')
    data = str(data_file.read())[:10000]

    new_data = pd.DataFrame(columns=['year', 'month', 'day', 'rho', 'w_x', 'w_y'])

    # Remove chars
    data = re.sub('[a-zA-Z]', '', data)
    # data = re.sub(' +', ' ', data)

    dates = re.findall('#.*?([0-9]{4}) ([0-9]{2}) ([0-9]{2}) .*?\n([\s\S]*?)\n30   100', data)
    # date1 = re.sub(' +', ' ', dates[0])
    print(dates[0])
    # for date in dates:
    #     print(date)

def plot_test_data():
    data = np.array(pd.read_csv('test_data.txt', sep=' '))
    data[:, [0, 1]] = data[:, [1, 0]]
    height, rho, w_x, w_y = convert_data(data)
    params, _ = curve_fit(inv_prop, height, rho)

    plt.plot(height, inv_prop(height, *params), 'b', label='fitted values')
    plt.plot(height, rho, 'r', label='observed values')
    plt.legend()
    plt.show()

    plt.plot(height, w_x, 'r', label=r'$w_x$')
    plt.plot(height, w_y, 'b', label=r'$w_y$')
    plt.show()

if __name__ == '__main__':

    remove_data_above_height(5000)
    # plot_test_data()


