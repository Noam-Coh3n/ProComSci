import numpy as np
import integration_methods as integration


class MovingObject():
    """An object that keeps track of position,
    """
    def __init__(self, pos: np.array):
        self.pos = pos
        self.pos_list = []
        self.x_list = []
        self.v_list = []
        self.a_list = []

    def _add_new_pos(self):
        self.pos_list.append(np.copy(self.pos))
        self.x_list.append(np.copy(self.x))
        self.v_list.append(np.copy(self.v))
        self.a_list.append(np.copy(self.a))


class Diver(MovingObject):

    def __init__(self, x: np.array, vel: np.array, wind: list,
                 air_pressure: list, temperature: list, h_shute: int,
                 stepsize: float):

        self.m_person = 68.8
        self.m_gear = 14
        self.m = self.m_person + self.m_gear
        self.g = 9.81
        self.gravitational_force = np.array([0, 0, -self.m * self.g])
        self.p_air = air_pressure
        self.m_air = 4.81 * 10**(-26)
        self.k_B_air = 1.38 * 10**(-23)
        self.temperature = temperature
        self.C_person_side = 1.11
        self.C_person_front = 1.18
        self.A_person_side = 0.38
        self.A_person_front = 0.55
        self.C_shute_side = 0.35
        self.C_shute_below = 1.68
        self.A_shute_side = 23.9
        self.A_shute_below = 47.8  # top?
        self.w = wind
        self.x = x
        self.v = vel
        self.a = [0, 0, 0]
        self.x_z_0 = self.x[2]
        self.h_shute = h_shute
        self.shute_pos = 0

        self.x2pos()
        super().__init__(self.pos)

        self.step_size = stepsize
        self.line_width = 1

        # self.time = 0

    def _get_derivative(self, data: np.array):
        """ Determine speed and acceleration given the position and speed.
        """
        [x_x, x_y, x_z, v_x, v_y, v_z] = data

        # Free fall or parachute.
        if x_z > self.h_shute:
            C_side = self.C_person_side
            C_front = self.C_person_front
            A_side = self.A_person_side
            A_front = self.A_person_front
        else:
            if self.shute_pos == 0:
                self.shute_pos = self.pos[2]

            C_side = self.C_shute_side
            C_front = self.C_shute_below
            A_side = self.A_shute_side
            A_front = self.A_shute_below

        if x_z == self.x_z_0:
            temp_part = len(self.temperature) - 1
            pres_part = len(self.p_air) - 1
            wind_part = len(self.w) - 1
        else:
            temp_part = int(x_z / (self.x_z_0 / len(self.temperature)))
            pres_part = int(x_z / (self.x_z_0 / len(self.p_air)))
            wind_part = int(x_z / (self.x_z_0 / len(self.w)))

        rho = (self.p_air[pres_part] * self.m_air) / (self.k_B_air * self.temperature[temp_part])
        drag_force = -0.5 * rho * np.array([
            C_side * A_side * (v_x - self.w[wind_part][0])**2 * np.sign(v_x - self.w[wind_part][0]),
            C_side * A_side * (v_y - self.w[wind_part][1])**2 * np.sign(v_y - self.w[wind_part][1]),
            C_front * A_front * v_z**2 * np.sign(v_z)])
        force = drag_force + self.gravitational_force

        [a_x, a_y, a_z] = force / self.m
        return np.array([v_x, v_y, v_z, a_x, a_y, a_z])

    # Integration methods.
    def integration(self, method):
        h = self.step_size
        prev_y = np.append(self.x_list[-1], self.v_list[-1])
        y = np.append(self.x, self.v)

        if method == 'Central diff':
            next_y, k = integration.integration_method(method, h, self._get_derivative, y, prev_y)
        else:
            next_y, k = integration.integration_method(method, h, self._get_derivative, y)

        self.x = next_y[:3]
        self.v = next_y[3:]
        self.a = k[3:]

    def move(self, method):
        self.x2pos()
        self._add_new_pos()
        self.integration(method)

    def x2pos(self):
        """ Get pos in cube from x location in real world.
        """
        self.pos = self.x * np.array([1, 1, -1]) + np.array([0, 0, self.x_z_0/2])

    def simulate_trajectory(self, method):
        while True:
            if self.x[2] <= 0:
                break
            self.move(method)
