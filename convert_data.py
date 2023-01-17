from constant import m_air, k_B_air
import numpy as np


def density(pressure, temperature):
    return (pressure * m_air) / (k_B_air * temperature)

def convert_data(data):
    rho = density(data['pressure'], data['temp'] / 10 + 273)
    radials = data['wind_dir'] * 2 * np.pi / 360 # Convert deg to radials
    w_x = np.sin(radials) * data['wind_vel']
    w_y = np.cos(radials) * data['wind_vel']
    return np.vstack((data['height'] / 10, rho, w_x, w_y))