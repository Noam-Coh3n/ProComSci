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

def remove_data_above_height(max_height):
    data_filename = 'vegas_wind_data.txt'
    if not path.exists(data_filename):
        print(f'{data_filename} does not exist.')
        return None
    data_file = open(data_filename, 'r')
    new_data_file = open('vegas_compressed.txt', 'w')

    for line in data_file:
        if line[0] == '#':
            new_data_file.write(f'#\n# {line[13:23]} #\n')
        line = re.sub('[A-Z]', ' ', line)
        height = re.search('^[\-0-9]+ *[\-0-9]* *[\-0-9]* *([\-0-9]*)', line)
        if line[0] != '3' and height and int(height.group(1)) < max_height:
            new_data_file.write(line)

def retrieve_rho_and_wind(year, month, day, output_rho=True, output_wind=True):
    data = open('vegas_compressed.txt', 'r')
    data = str(data.read())

    data = re.sub(' +', ' ', data)

    data = re.findall(f'# {year} {month:02} {day:02} #\n([\s\S]*?)#', data)

    print(len(data))
    result = []
    for data_item in data:
        data_item = np.fromstring(data_item, sep=' ')
        data_item = np.reshape(data_item, (-1, 9))

        height, press, temp, dir, vel = data_item[:,[3, 2, 4, 7, 8]].transpose()
        height -= 697 # Natural elevation of Las Vegas.
        rho = density(press, temp / 10 + 273)
        radials = dir * 2 * np.pi / 360 # Convert deg to radials
        w_x = np.sin(radials) * vel / 10
        w_y = np.cos(radials) * vel / 10
        return_data = np.vstack((height, rho)) if output_rho else height
        result.append(np.vstack((return_data, w_x, w_y)) if output_wind else return_data)

    return np.array(result)

def plot_test_data():
    # data = np.array(pd.read_csv('test_data.txt', sep=' '))
    # data[:, [0, 1]] = data[:, [1, 0]]
    height, rho, w_x, w_y = retrieve_rho_and_wind(2018, 1, 1)
    params, _ = curve_fit(inv_prop, height, rho)

    plt.plot(height, inv_prop(height, *params), 'b', label='fitted values')
    plt.plot(height, rho, 'r', label='observed values')
    plt.legend()
    plt.show()

    plt.plot(height, w_x, 'r', label=r'$w_x$')
    plt.plot(height, w_y, 'b', label=r'$w_y$')
    plt.show()

def all_possible_days():
    return [(y, m, d) for y in range(2018, 2023) \
                      for m in range(1, 3) \
                      for d in range(1, 4)]

# def average_rho():
#     data = np.array([[], []])
#     for (y, m, d) in all_possible_days():
#         items = retrieve_rho_and_wind(y, m , d, output_wind=False)
#         for item in items:
#             heights, rho = item
#         data = np.hstack((data, items))
#     print(data)

if __name__ == '__main__':

    # remove_data_above_height(5000)
    # plot_test_data()
    print(retrieve_rho_and_wind(2018, 1, 1))
    # average_rho()

