import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from convert_data import convert_data


def inv_prop(x, a, b, c):
    return a / (x + b) + c

# def fit_data(x, y):
#     X = np.array([[1 / x_i, 1] for x_i in x]).transpose()
#     return np.linalg.inv((X @ X.transpose())) @ X @ y


if __name__ == '__main__':
    data = pd.read_csv('test_data.txt', sep=' ')
    height, rho, w_x, w_y = convert_data(data)
    params, _ = curve_fit(inv_prop, height, rho)

    plt.plot(height, rho, 'r', label='observed values')
    plt.plot(height, inv_prop(height, *params), 'b', label='fitted values')
    plt.legend()
    plt.show()


