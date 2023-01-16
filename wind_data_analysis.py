import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constant import m_air, k_B_air

def inv_prop(x, k):
    return

def density(pressure, temperature):
    return (pressure * m_air) / (k_B_air * temperature)

def convert_data(data):
    rho = density(data['pressure'], data['temp'] / 10 + 273)
    radials = data['wind_dir'] * 2 * np.pi / 360 # Convert deg to radials
    w_x = np.sin(radials) * data['wind_vel']
    w_y = np.cos(radials) * data['wind_vel']
    return np.vstack((data['height'] / 10, rho, w_x, w_y))

if __name__ == '__main__':
    # data = pd.DataFrame({'height' : [1, 2, 3], 'pressure' : [100000, 99000, 98000], 'temp' : [293, 292, 291], 'degrees' : [89, 90, 91], 'wind_vel' : [10, 10, 10]})
    # data_file = open("test_data.txt", "r")
    data = pd.read_csv('test_data.txt', sep=' ')
    # print(data)
    height, rho, w_x, w_y = convert_data(data)
    print(data['temp'])

    plt.plot(height, rho)
    plt.show()


