#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Double Bachelor mathematics and computer science.
#
# constants.py:
# Here we define constants that we will use in different files.

from collections import namedtuple
import numpy as np

v_plane = 45
h_plane = 2500
min_h_opening = 150
max_h_opening = 250
h_opening = 200
radius_landing_area = 200

w_x_bounds = (-np.inf, -2)
w_y_bounds = (-np.inf, -2)

m_diver = 83
m_air = 4.81 * 10**(-26)
g = 9.81
kB = 1.38 * 10**(-23)

sides = namedtuple('Sides', ['front', 'side'])
C_diver = sides(1.18, 1.11)
C_chute = sides(1.68, 0.35)
A_diver = sides(0.55, 0.38)
A_chute = sides(47.8, 23.9)

color_avg = '#319600'
color_dev = '#00696b'
color_fitted_avg = '#49820F'
color_fitted_dev = '#0DD5D1'
color_dataset = '#144A7B'
