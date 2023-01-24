import numpy as np
import integration
from wind_and_rho_generator import rho
import constants as const


class Diver():

    def __init__(self, x: np.array, velocity: np.array, wind,
                 stepsize: float = 0.001, seed=None):

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

        # self.x_z_0 = self.x[2]
        # self.chute_pos = 0

        # self.x2pos()
        # self.pos_list = []

        self.step_size = stepsize

    def _add_new_pos(self):
        # self.pos_list.append(np.copy(self.pos))
        self.x_list.append(np.copy(self.x))
        self.v_list.append(np.copy(self.v))
        self.a_list.append(np.copy(self.a))

    def _get_derivative(self, data: np.array):
        """ Determine speed and acceleration given the position and speed.
        """
        _, _, x_z, v_x, v_y, v_z = data

        # Free fall
        if x_z > const.h_opening:
            C = const.C_diver
            A = const.A_diver

        # Under canopy
        else:
            # if self.chute_pos == 0:
            #     self.chute_pos = self.pos[2]
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

    # Integration methods.
    def integration(self, method):
        h = self.step_size
        prev_y = np.append(self.x_list[-1], self.v_list[-1])
        y = np.append(self.x, self.v)

        if method == 'central diff':
            next_y, k = integration.integrate(method, h, self._get_derivative, y, prev_y)
        else:
            next_y, k = integration.integrate(method, h, self._get_derivative, y)

        self.x = next_y[:3]
        self.v = next_y[3:]
        self.a = k[3:]

    def move(self, method):
        # self.x2pos()
        self._add_new_pos()
        self.integration(method)

    # def x2pos(self):
    #     """Get position in cube from x location in real world."""
    #     self.pos = self.x * np.array([2, 2, -1]) + np.array([0, 0, self.x_z_0/2])

    def simulate_trajectory(self, method):
        while True:
            if self.x[2] <= 0:
                break
            self.move(method)
