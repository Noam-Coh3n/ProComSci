# ------------------------------ Import ----------------------------------------

import time
import random
import numpy as np
import pygame
pygame.init()

# ------------------------------- Color ----------------------------------------

#                Red    Green   Blue
black =         (0,     0,      0)
light_black =   (50,    50,     50)
white =         (255,   255,    255)
dark_white =    (200,   200,    200)
red =           (255,   0,      0)
light_red =     (100,   0,      0)
green =         (0,     255,    0)
dark_green =    (0,     200,    0)
light_green =   (0,     155,    0)
blue =          (0,     0,      255)
dark_blue =     (0,     0,      80)
grey =          (100,   100,    100)
brown =         (200,   100,     0)
dark_brown =    (50,    52,     0)
sand =          (194,   178,    128)

# ------------------------------ Settings --------------------------------------

GAME_NAME = 'Iets'
FPS = 60
BACKGROUND_COLOR = black

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

# -----------------------------  Variables -------------------------------------

# ---------------------------- Setup display -----------------------------------

game_display = pygame.display.set_mode()
# game_display = pygame.display.set_mode((2400, 950))
# display2 = pygame.display.set_mode((100, 100))

pygame.display.set_caption(GAME_NAME)
clock = pygame.time.Clock()

# ------------------------------ Functions -------------------------------------

def project(pos : np.array):
    """ Project 3 dim position to a 2 dim position.
    """
    y = 1 / (3 - pos[1])
    return np.dot(np.array([[y, 0, 0],[0, 0, y]]), pos)

def rotate(pos : np.array, x_angle : int, y_angle : int, z_angle : int):
    """ Rotate an 3 dim position around the x, y and z axis.
    """

    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(x_angle), -np.sin(x_angle)],
                           [0, np.sin(x_angle), np.cos(x_angle)]])
    rotation_y = np.array([[np.cos(y_angle), 0, -np.sin(y_angle)],
                           [0, 1, 0],
                           [np.sin(y_angle), 0, np.cos(y_angle)]])
    rotation_z = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                           [np.sin(z_angle), np.cos(z_angle), 0],
                           [0, 0, 1]])

    return np.dot(rotation_z, np.dot(rotation_y, np.dot(rotation_x, pos)))

# ------------------------------- Objects --------------------------------------

class Node:
    """ Object that has an 3D position, used in Object3D.
    """
    def __init__(self, pos : np.array):
        self.pos = pos

class Edge:
    """ Object that has two Nodes, used in Object3D.
    """
    def __init__(self, node1 : Node, node2: Node):
        self.node1 = node1
        self.node2 = node2

class Object3D:
    """ An object in a 3 dimentional space that consists of nodes and edges 
        between nodes. 
    """
    def __init__(self, pos : np.array, pos0 : list, zoom : int):

        self.nodes = []
        self.edges = []
        self.pos0 = pos0
        self.zoom = zoom
    
    def draw(self):
        """ Draw the edges of the object.
        """
        for edge in self.edges:
            a_p = tuple(project(edge.node1.pos) * self.zoom + np.array(self.pos0))
            b_p = tuple(project(edge.node2.pos) * self.zoom + np.array(self.pos0))
            pygame.draw.line(game_display, self.color, a_p, b_p)

    def change(self, x_rot, y_rot, z_rot, x_vel, z_vel, zoom):
        """ Change the view of the object because of rotation, shift and zoom.
        """
        rot = 2 * np.pi / 360
        for node in self.nodes:
            node.pos = rotate(node.pos, rot * x_rot, rot * y_rot, rot * z_rot)

        self.pos0[0] += x_vel
        self.pos0[1] += z_vel
        self.zoom = zoom

class Cube(Object3D):
    """ 3D object with specific node positions and edges such that 
        it creates a cube.
    """
    def __init__(self,  measure : tuple, pos : np.array, pos0 : list, zoom : int):
        super().__init__(pos, pos0, zoom)

        self.color = white

        a = measure[0]
        b = measure[1]
        c = measure[2]

        x = np.array([a,0,0])
        y = np.array([0,b,0])
        z = np.array([0,0,c])

        self.nodes = [Node(pos), Node(pos + x), Node(pos + x + y), 
                      Node(pos + x + z), Node(pos + x + y + z), Node(pos + y), 
                      Node(pos + y + z), Node(pos + z)]

        self.edges = [Edge(self.nodes[0], self.nodes[1]), Edge(self.nodes[0], self.nodes[5]), Edge(self.nodes[0], self.nodes[7]), 
                      Edge(self.nodes[2], self.nodes[1]), Edge(self.nodes[2], self.nodes[5]), Edge(self.nodes[2], self.nodes[4]), 
                      Edge(self.nodes[6], self.nodes[4]), Edge(self.nodes[6], self.nodes[7]), Edge(self.nodes[6], self.nodes[5]), 
                      Edge(self.nodes[3], self.nodes[7]), Edge(self.nodes[3], self.nodes[1]), Edge(self.nodes[3], self.nodes[4])]

class Arrow(Object3D):
    """ 3D object with specific node positions and edges such that 
        it creates a arrow.
    """
    def __init__(self, size : int, direction : np.array, pos: np.array, pos0: list, zoom: int):
        super().__init__(pos, pos0, zoom)

        self.color = blue
        self.size = size
        direction = direction * self.size

        [a,b] = direction
        dirc = np.array([a,b,0])
        length = np.sqrt(a**2 + b**2)
        self.nodes = [Node(pos), Node(pos + dirc * 1/2), Node(pos + dirc),
                      Node(pos + (dirc + np.array([b/3, -a/3, 0])) * 4/6),
                      Node(pos + (dirc + np.array([0, 0, length/3])) * 4/6),
                      Node(pos + (dirc + np.array([-b/3, a/3, 0])) * 4/6),
                      Node(pos + (dirc + np.array([0, 0, -length/3])) * 4/6)]

        self.edges = [Edge(self.nodes[0], self.nodes[1]), Edge(self.nodes[1], self.nodes[3]),
                      Edge(self.nodes[1], self.nodes[4]), Edge(self.nodes[1], self.nodes[5]),
                      Edge(self.nodes[1], self.nodes[6]), Edge(self.nodes[2], self.nodes[3]),
                      Edge(self.nodes[2], self.nodes[4]), Edge(self.nodes[2], self.nodes[5]),
                      Edge(self.nodes[2], self.nodes[6]), Edge(self.nodes[3], self.nodes[4]),
                      Edge(self.nodes[4], self.nodes[5]), Edge(self.nodes[5], self.nodes[6]),
                      Edge(self.nodes[6], self.nodes[3])]

class Wind: # Direction in np.array
    """ Multiple arrows stacked on top of eachother,
        evenly spaced between -1 and 1.
    """
    def __init__(self, directions : list, pos0 : list, zoom : int):

        self.arrows = []
        for c, direction in enumerate(directions):
            self.arrows.append(Arrow(size = 0.01, direction=direction, pos=np.array([1.3, 0, (c+1)/(len(directions) + 1)*2 - 1]), pos0=pos0, zoom=zoom))

    def draw(self):
        for arrow in self.arrows: 
            arrow.draw()

    def change(self, x_rot, y_rot, z_rot, x_vel, z_vel, zoom):
        for arrow in self.arrows: 
            arrow.change(x_rot, y_rot, z_rot, x_vel, z_vel, zoom)

class MovingObject(Object3D):
    """ An Object 3D with an current position.
        Every new position the object has becomes a new Node.
        There exists an edge between every node and the previous node.
    """
    def __init__(self, pos: np.array, pos0: list, zoom: int):
        super().__init__(pos, pos0, zoom)
        self.pos = pos
        self.nodes = [Node(self.pos)]

    def change(self, x_rot, y_rot, z_rot, x_vel, z_vel, zoom):
        super().change(x_rot, y_rot, z_rot, x_vel, z_vel, zoom)

        # Also change the current position.
        rot = 2 * np.pi / 360
        self.pos = rotate(self.pos, rot * x_rot, rot * y_rot, rot * z_rot)

    def _add_new_pos(self):
        self.nodes.append(Node(self.pos))
        self.edges.append(Edge(self.nodes[-2], self.nodes[-1]))

class Particle(MovingObject):
    """ A randomly moving MovingObject for testing purposes.
    """
    def __init__(self, pos: np.array, pos0: list, zoom: int):
        super().__init__(pos, pos0, zoom)
        self.color = red
    
    def move(self):
        x = random.randrange(-1,2) * 0.05
        y = random.randrange(-1,2) * 0.05
        z = random.randrange(-1,2) * 0.05
        self.pos += np.array([x, y, z])
        self._add_new_pos()

class Diver(MovingObject):

    def __init__(self, x : np.array, vel : np.array, wind : list, \
                 air_pressure : list, temperature : list, pos0 : list, zoom : int):
        
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
        self.w = wind
        self.x = x
        self.v = vel
        self.x_z_0 = self.x[2]

        self.x2pos()
        super().__init__(self.pos, pos0, zoom)

        self.step_size = 0.1 # Secondes.
        self.color = red
    
    def _get_derivative(self, data : np.array):
        """ Determine speed and acceleration given the position and speed.
        """
        [x_x, x_y, x_z, v_x, v_y, v_z] = data

        if x_z == self.x_z_0:
            temp_part, pres_part, wind_part = len(self.temperature)-1, len(self.p_air)-1, len(self.w)-1
        else:
            temp_part = int(x_z / (self.x_z_0 / len(self.temperature))) # locatie van temp van onder naar boven.
            pres_part = int(x_z / (self.x_z_0 / len(self.p_air)))
            wind_part = int(x_z / (self.x_z_0 / len(self.w)))
        
        rho = (self.p_air[pres_part] * self.m_air) / (self.k_B_air * self.temperature[temp_part])
        drag_force = -0.5 * rho * np.array([
            self.C_person_side * self.A_person_side * (v_x - self.w[wind_part][0])**2 * np.sign(v_x - self.w[wind_part][0]),
            self.C_person_side * self.A_person_side * (v_y - self.w[wind_part][1])**2 * np.sign(v_y - self.w[wind_part][1]),
            self.C_person_front * self.A_person_front * v_z**2 * np.sign(v_z)])
        force = drag_force + self.gravitational_force

        [a_x, a_y, a_z] = force / self.m
        return np.array([v_x, v_y, v_z, a_x, a_y, a_z])

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
        self.pos = (2 * self.x / self.x_z_0 - 1 + np.array([1, 1, 0])) * np.array([1, -1, -1])


# -------------------------------- Game ----------------------------------------

class Game:

    def __init__(self):
        self.zoom = 500
        self.pos0 = [1000, 500]

        # wind_list = [np.array([0,1]), np.array([1,1]), np.array([1,0]),
        #              np.array([2,0]), np.array([4,0]), np.array([2,-2])]
        wind_list = [np.array([60,60])]
        temp_list = [280]
        air_pres_list = [100000]

        self.cube = Cube(measure=(2,2,2), pos=np.array([-1, -1, -1]), pos0=self.pos0, zoom=self.zoom)
        self.particle = Particle(pos=np.array([0., 0., 0.]), pos0=self.pos0, zoom=self.zoom)
        self.diver = Diver(x=np.array([0., 0., 3600.]), vel=np.array([0., 0., 0.]),
                          wind=wind_list, air_pressure=air_pres_list, temperature=temp_list,
                          pos0=self.pos0, zoom=self.zoom)

        self.wind = Wind(wind_list, self.pos0, self.zoom)

    def draw_screen(self):
        game_display.fill(BACKGROUND_COLOR)
        self.cube.draw()
        # self.particle.draw()
        self.wind.draw()
        self.diver.draw()

    def change(self, x_rot, y_rot, z_rot, x_vel, z_vel, zoom):

        # self.particle.move()
        self.diver.move()

        self.cube.change(x_rot, y_rot, z_rot, x_vel, z_vel, zoom)
        self.wind.change(x_rot, y_rot, z_rot, x_vel, z_vel, zoom)
        self.diver.change(x_rot, y_rot, z_rot, x_vel, z_vel, zoom)
        # self.particle.change(x_rot, y_rot, z_rot, x_vel, z_vel, zoom)

        


    def game_loop(self):
        x_rot, y_rot, z_rot, x_vel, z_vel = 0, 0, 0, 0, 0

        # Loop
        while True:

            # Controls
            for event in pygame.event.get():

                # Keyboard Presses
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        x_rot = 1
                    elif event.key == pygame.K_w:
                        x_rot = -1
                    if event.key == pygame.K_a:
                        z_rot = 1
                    elif event.key == pygame.K_d:
                        z_rot = -1
                    # if event.key == pygame.K_LEFT:
                    #     y_rot = 1
                    # elif event.key == pygame.K_RIGHT:
                    #     y_rot = -1

                    if event.key == pygame.K_KP6:
                        x_vel = 5
                    elif event.key == pygame.K_KP4:
                        x_vel = -5
                    if event.key == pygame.K_KP8:
                        z_vel = -5
                    elif event.key == pygame.K_KP2:
                        z_vel = 5

                elif event.type == pygame.KEYUP:
                    x_rot, y_rot, z_rot, x_vel, z_vel = 0, 0, 0, 0, 0

                if event.type == pygame.MOUSEWHEEL:
                    if event.y == -1:
                        self.zoom -= 30
                    
                    if event.y == 1:
                        self.zoom += 30

                # Quit
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # Background
            self.change(x_rot, y_rot, z_rot, x_vel, z_vel, self.zoom)
            self.draw_screen()

            # Update
            pygame.display.update()
            clock.tick(FPS)

# -------------------------------- Run -----------------------------------------
Game().game_loop()

# het probleem is dat de pos gemaakt word opgeslagen word, daarna gechanged word en dan weer wordt overgeschreven
# Het is beter om de engine op een manier te schrijven zdd dat posities niet verandert worden, maar de rotatie opgeslagen word.