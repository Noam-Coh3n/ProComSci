#
# Name: Lars Bijvoet, Erik Leonards, Noam Cohen & Jelle Sipkes
# Study: Double Bachelor mathematics and computer science.
#
# diver.py:
# This is the diver file. In this file we have the class Diver.
# In this class we will calculate everything that is needed for the simulation
# of the diver.
# Also the integration methods are used here which are imported from the
# integration.py file.

import numpy as np
import integration
from wind_and_rho_generator import rho
import constants as const


class Diver():

    def __init__(self, x: np.array, velocity: np.array, wind,
                 stepsize=0.001, int_method='rk4', seed=None,
                 h_opening=const.h_opening, dynamic_funcs=None):
        """Initialise all variables."""

        self.F_gravity = np.array([0, 0, -const.m_diver * const.g])
        self.rho = rho
        self.wind_x = wind.wind(wind_dir='x', seed=seed)
        self.wind_y = wind.wind(wind_dir='y', seed=seed)

        self.x = x
        self.v = velocity
        self.a = [0, 0, 0]
        self.x_list = []
        self.v_list = []
        self.a_list = []

        self.dynamic_funcs = dynamic_funcs
        if dynamic_funcs:
            self.dist_func, self.dir_func = dynamic_funcs
            self.h_opening = const.min_h_opening
        else:
            self.h_opening = h_opening
        self.step_size = stepsize
        self.int_method = int_method

    def _add_new_pos(self):
        self.x_list.append(np.copy(self.x))
        self.v_list.append(np.copy(self.v))
        self.a_list.append(np.copy(self.a))

    def _diff(self, data: np.array):
        """Determine speed and acceleration given the position and speed."""
        x_x, x_y, x_z, v_x, v_y, v_z = data

        w_x = self.wind_x(x_z)
        w_y = self.wind_y(x_z)
        if (self.dynamic_funcs and const.max_h_opening > x_z and
                x_z > self.h_opening):
            dist = self.dist_func(x_z, np.sqrt(w_x ** 2 + w_y ** 2))
            # closest_x_to_origin = ((-x_y * w_x + w_y * x_x) /
            #                        ((w_x ** 2) / w_y + w_y))
            # closest_y_to_origin = - w_x / w_y * closest_x_to_origin

            dir = self.dir_func(x_z, w_x, w_y)

            closest_x_to_origin = (dir * x_x - x_y) / (1 / dir + dir)
            closest_y_to_origin = - closest_x_to_origin / dir

            dist_to_closest_point = np.sqrt((x_x - closest_x_to_origin) ** 2
                                            + (x_y - closest_y_to_origin) ** 2)
            if dist_to_closest_point > dist or x_z < const.min_h_opening:
                self.h_opening = x_z

        # Free fall
        if x_z > self.h_opening:
            C = const.C_diver
            A = const.A_diver
        # Under canopy
        else:
            C = const.C_chute
            A = const.A_chute

        rho = self.rho(x_z)
        w_x = self.wind_x(x_z)
        w_y = self.wind_y(x_z)
        F_drag = -0.5 * rho * np.array([
            C.side * A.side * (v_x - w_x)**2 * np.sign(v_x - w_x),
            C.side * A.side * (v_y - w_y)**2 * np.sign(v_y - w_y),
            C.front * A.front * v_z**2 * np.sign(v_z)])

        F = F_drag + self.F_gravity

        [a_x, a_y, a_z] = F / const.m_diver
        return np.array([v_x, v_y, v_z, a_x, a_y, a_z])

    # Integration methods
    def _step(self):
        h = self.step_size
        prev_y = np.append(self.x_list[-1], self.v_list[-1])
        y = np.append(self.x, self.v)

        if self.int_method == 'central diff':
            data = [y, prev_y]
        else:
            data = [y]
        new_y, k = integration.integrate(self.int_method, h, self._diff, *data)

        self.x = new_y[:3]
        self.v = new_y[3:]
        self.a = k[3:]

    def move(self):
        self._add_new_pos()
        self._step()

    def simulate_trajectory(self):
        while self.x[2] > 0:
            self.move()
