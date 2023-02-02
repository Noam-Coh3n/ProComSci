#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Double Bachelor mathematics and computer science.
#
# help_visual.py:
# Helper file for visual.py.
# Here the coordinates of the real world are transferd to coordinates in the
# cube.

import numpy as np


def x2pos(x_list):
    """Calculate the real position to cube position.
    This function is used in visual.py.
    """
    pos_list = np.copy(x_list)
    z0 = x_list[0][2]
    for x in pos_list:
        x[2] = x[2] * -1 + z0/2
    return pos_list
