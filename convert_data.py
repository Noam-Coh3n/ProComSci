from constant import m_air, k_B_air
import numpy as np


def density(pressure, temperature):
    return (pressure * m_air) / (k_B_air * temperature)

def convert_data(data):
    height, pressure, temp, wind_dir, wind_vel = data.transpose()
    rho = density(pressure, temp / 10 + 273)
    radials = wind_dir * 2 * np.pi / 360 # Convert deg to radials
    w_x = np.sin(radials) * wind_vel / 10
    w_y = np.cos(radials) * wind_vel / 10
    return np.vstack((height / 10, rho, w_x, w_y))