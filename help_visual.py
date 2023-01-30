import numpy as np


def x2pos(x_list):
    pos_list = np.copy(x_list)
    # pos_list = []
    z0 = x_list[0][2]
    # for x in x_list:
    for x in pos_list:
        x[2] = x[2] * -1 + z0/2
    #     pos_list.append(x * np.array([2, 2, -1]) +  np.array([0, 0, z0/2]))
    return pos_list
