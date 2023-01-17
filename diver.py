import numpy as np

class MovingObject():
    """ An Object with an current position.
        Every new position the object has becomes a new Node.
        There exists an edge between every node and the previous node.
    """
    def __init__(self, pos: np.array):
        self.pos = pos
        self.pos_list = []
        self.x_list = []

    def _add_new_pos(self):
        """ Change current position and add the node and edge.
        """
        self.pos_list.append(np.copy(self.pos))
        self.x_list.append(np.copy(self.x))

class Diver(MovingObject):

    def __init__(self, x : np.array, vel : np.array, wind : list, \
                 air_pressure : list, temperature : list, h_shute : int):
        
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
        self.A_shute_below = 47.8 # top?
        self.w = wind
        self.x = x
        self.v = vel
        self.x_z_0 = self.x[2]
        self.h_shute = h_shute
        self.shute_pos = 0

        self.x2pos()
        super().__init__(self.pos)

        self.step_size = 0.005 # Secondes.
        self.line_width = 1

        # self.time = 0

    def _get_derivative(self, data : np.array):
        """ Determine speed and acceleration given the position and speed.
        """
        [x_x, x_y, x_z, v_x, v_y, v_z] = data

        # Free fall or parashute.
        if x_z > self.h_shute:
            C_side = self.C_person_side
            C_front = self.C_person_front
            A_side = self.A_person_side
            A_front = self.A_person_front
        else:
            # self.time +=1
            # if self.time < 250: print(self.time * self.step_size, self.v, self.x)
            if self.shute_pos == 0:
                self.shute_pos = self.pos[2]
            C_side = self.C_shute_side
            C_front = self.C_shute_below
            A_side = self.A_shute_side
            A_front = self.A_shute_below

        if x_z == self.x_z_0:
            temp_part, pres_part, wind_part = len(self.temperature)-1, len(self.p_air)-1, len(self.w)-1
        else:
            temp_part = int(x_z / (self.x_z_0 / len(self.temperature))) # locatie van temp van onder naar boven.
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

    # Intergration methods.
    def RK4(self):
        h = self.step_size
        y = np.append(self.x, self.v)

        k1 = self._get_derivative(y)
        k2 = self._get_derivative(y + h * k1/2)
        k3 = self._get_derivative(y + h * k2/2)
        k4 = self._get_derivative(y + h * k3)
        next_y = y + (k1 + 2*k2 + 2*k3 + k4)* h/6
        self.x = next_y[:3]
        self.v = next_y[3:]

    def move(self):
        self.RK4()
        self.x2pos()
        self._add_new_pos()

    def x2pos(self):
        """ Get pos in cube from x location in real world.
        """
        self.pos = self.x * np.array([1, 1, -1]) + np.array([0, 0, self.x_z_0/2])
