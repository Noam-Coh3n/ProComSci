import numpy as np
import integration
from constants import *


class Diver():

    def __init__(self, x: np.array, vel: np.array, wind: list,
                 air_pressure: list, temperature: list, h_opening: int,
                 stepsize: float):

        self.F_gravity = np.array([0, 0, -m_diver * g])
        self.p_air = air_pressure
        self.temperature = temperature
        self.wind = wind

        self.x = x
        self.v = vel
        self.a = [0, 0, 0]
        self.x_list = []
        self.v_list = []
        self.a_list = []

        self.x_z_0 = self.x[2]
        self.h_opening = h_opening
        self.chute_pos = 0

        self.x2pos()
        self.pos_list = []

        self.step_size = stepsize

    def _add_new_pos(self):
        self.pos_list.append(np.copy(self.pos))
        self.x_list.append(np.copy(self.x))
        self.v_list.append(np.copy(self.v))
        self.a_list.append(np.copy(self.a))

    def _get_derivative(self, data: np.array):
        """ Determine speed and acceleration given the position and speed.
        """
        [x_x, x_y, x_z, v_x, v_y, v_z] = data

        # Free fall
        if x_z > self.h_opening:
            C = C_diver
            A = A_diver

        # Under canopy
        else:
            if self.chute_pos == 0:
                self.chute_pos = self.pos[2]

            C = C_chute
            A = A_chute

        if x_z == self.x_z_0:
            temp_part = len(self.temperature) - 1
            pres_part = len(self.p_air) - 1
            wind_part = len(self.wind) - 1
        else:
            temp_part = int(x_z / (self.x_z_0 / len(self.temperature)))
            pres_part = int(x_z / (self.x_z_0 / len(self.p_air)))
            wind_part = int(x_z / (self.x_z_0 / len(self.wind)))

        rho = (self.p_air[pres_part] * m_air) / (kB * self.temperature[temp_part])
        F_drag = -0.5 * rho * np.array([
            C.side * A.side * (v_x - self.wind[wind_part][0])**2 * np.sign(v_x - self.wind[wind_part][0]),
            C.side * A.side * (v_y - self.wind[wind_part][1])**2 * np.sign(v_y - self.wind[wind_part][1]),
            C.front * A.front * v_z**2 * np.sign(v_z)])

        F = F_drag + self.F_gravity

        [a_x, a_y, a_z] = F / m_diver
        return np.array([v_x, v_y, v_z, a_x, a_y, a_z])

    # Integration methods.
    def integration(self, method):
        h = self.step_size
        prev_y = np.append(self.x_list[-1], self.v_list[-1])
        y = np.append(self.x, self.v)

        if method == 'Central diff':
            next_y, k = integration.integrate(method, h, self._get_derivative, y, prev_y)
        else:
            next_y, k = integration.integrate(method, h, self._get_derivative, y)

        self.x = next_y[:3]
        self.v = next_y[3:]
        self.a = k[3:]

    def move(self, method):
        self.x2pos()
        self._add_new_pos()
        self.integration(method)

    def x2pos(self):
        """Get position in cube from x location in real world."""
        self.pos = self.x * np.array([1, 1, -1]) + np.array([0, 0, self.x_z_0/2])

    def simulate_trajectory(self, method):
        while True:
            if self.x[2] <= 0:
                break
            self.move(method)
