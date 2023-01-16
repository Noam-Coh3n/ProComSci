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
FPS = 128
BACKGROUND_COLOR = black

SCREEN_SIZE = 1000
DROP_HEIGHT = 3600
PARAMETER_SIZE = 5000

# -----------------------------  Variables -------------------------------------

# ---------------------------- Setup display -----------------------------------

game_display = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
# game_display = pygame.display.set_mode((2400, 950))
# display2 = pygame.display.set_mode((100, 100))

pygame.display.set_caption(GAME_NAME)
clock = pygame.time.Clock()

# ------------------------------ Functions -------------------------------------



# ------------------------------- Objects --------------------------------------

class Node:
    """ Object that has an 3D position, used in Object3D.
    """
    def __init__(self, pos : np.array):
        self.pos = np.copy(pos)

class Edge:
    """ Object that has two Nodes, used in Object3D.
    """
    def __init__(self, node1 : Node, node2: Node):
        self.node1 = node1
        self.node2 = node2
    
class Cube:
    """ 3D object with specific node positions and edges such that 
        it creates a cube.
    """
    def __init__(self,  measure : tuple, pos : np.array):

        self.color = white
        self.line_width = 1

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

class Circ:

    def __init__(self,  measure : int, pos : np.array):

        self.color = red
        self.line_width = 5
        nr_points = 6
        rot = 2*np.pi/nr_points

        self.nodes = []
        self.edges = []
        for i in range(nr_points):
            self.nodes.append(Node(np.array([np.cos(rot*i) * measure + pos[0], np.sin(rot*i) * measure + pos[1], pos[2]])))
            if i != 0: 
                self.edges.append(Edge(self.nodes[-2], self.nodes[-1]))
        self.edges.append(Edge(self.nodes[-1], self.nodes[0]))

class Arrow:
    """ 3D object with specific node positions and edges such that 
        it creates a arrow.
    """
    def __init__(self, size : int, direction : np.array, pos: np.array):

        self.color = blue
        self.line_width = 1
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

class Wind:
    """ Multiple arrows stacked on top of eachother,
        evenly spaced between -1 and 1.
    """
    def __init__(self, directions: np.array, length: int):

        self.objects = []
        for c, direction in enumerate(directions):
            self.objects.append(Arrow(30, direction, np.array([0, 0, -((c+1)/(len(directions) + 1)*length - length/2)])))

class MovingObject():
    """ An Object with an current position.
        Every new position the object has becomes a new Node.
        There exists an edge between every node and the previous node.
    """
    def __init__(self, pos: np.array):
        self.pos = pos
        self.nodes = [Node(pos)]
        self.edges = []

    def _add_new_pos(self):
        """ Change current position and add the node and edge.
        """
        self.nodes.append(Node(self.pos))
        self.edges.append(Edge(self.nodes[-2], self.nodes[-1]))

class Particle(MovingObject):
    """ A randomly moving MovingObject for testing purposes.
    """
    def __init__(self, pos: np.array):
        super().__init__(pos)
        self.color = red
        self.line_width = 1
    
    def move(self):
        x = random.randrange(-1,2) * 50
        y = random.randrange(-1,2) * 50
        z = random.randrange(-1,2) * 50

        self.pos += np.array([x, y, z])

        self._add_new_pos()

class Diver(MovingObject):

    def __init__(self, x : np.array, vel : np.array, wind : list, \
                 air_pressure : list, temperature : list):
        
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
        super().__init__(self.pos)

        self.step_size = 0.005 # Secondes.  (bekijk in de visualisatie slechts per sec)
        self.color = red
        self.line_width = 1

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
        self.pos = self.x * np.array([1, 1, -1]) + np.array([0, 0, DROP_HEIGHT/2])

# -------------------------------- World ---------------------------------------

class World:

    def __init__(self):
        self.rotation = np.array([0.,0.,0.])

        self.camera_pos = np.array([0, -8000, 0])
        self.camera_max_angle = 0.25 # times pi rad
        self.zoom = 1

        self.objects = []
        self.object_clusters = []

    def _rotate_pos(self, pos: np.array):
        """ Rotate an 3 dim position around the x, y and z axis.
        """
        [r_x, r_y, r_z] = self.rotation
        rotation_x = np.array([[1, 0, 0],
                               [0, np.cos(r_x), -np.sin(r_x)],
                               [0, np.sin(r_x), np.cos(r_x)]])
        rotation_y = np.array([[np.cos(r_y), 0, -np.sin(r_y)],
                               [0, 1, 0],
                               [np.sin(r_y), 0, np.cos(r_y)]])
        rotation_z = np.array([[np.cos(r_z), -np.sin(r_z), 0],
                               [np.sin(r_z), np.cos(r_z), 0],
                               [0, 0, 1]])

        return np.dot(rotation_z, np.dot(rotation_y, np.dot(rotation_x, pos)))

    def _project_pos(self, pos: np.array):
        """ Project 3 dim position to a 2 dim position.
        """
        angle_x = np.arctan2((pos[0] - self.camera_pos[0]), (pos[1] - self.camera_pos[1]))
        angle_z = np.arctan2((pos[2] - self.camera_pos[2]), (pos[1] - self.camera_pos[1]))

        return np.array([SCREEN_SIZE/2 + angle_x/(self.camera_max_angle * np.pi) * SCREEN_SIZE/2, 
                         SCREEN_SIZE/2 + angle_z/(self.camera_max_angle * np.pi) * SCREEN_SIZE/2])

    def add_object(self, obj):
        self.objects.append(obj)

    def add_object_cluster(self, obj):
        self.object_clusters.append(obj)

    def rotate(self, rot):
        self.rotation += rot * 0.05

    def update(self):
        
        for obj in self.objects:
            if 'move' in dir(obj):
                obj.move()

    def draw(self):
        """ Draw all objects of world.
        """

        for obj in self.objects:
            self._draw_object(obj)

        for cluster in self.object_clusters:
            for obj in cluster.objects:
                self._draw_object(obj)

    def _draw_object(self, obj):
        """ Calculate the locations of the nodes from the current angle.
            Project the locations on the 2D creen and draw the edges.
        """
        for edge in obj.edges:
            a, b = edge.node1.pos, edge.node2.pos
            a_s = self._rotate_pos(a)
            b_s = self._rotate_pos(b)
            a_p = tuple(self._project_pos(a_s) * self.zoom)
            b_p = tuple(self._project_pos(b_s) * self.zoom)

            pygame.draw.line(game_display, obj.color, a_p, b_p, obj.line_width)

# -------------------------------- Game ----------------------------------------

class Game:

    def __init__(self):

        temp_list = [280]
        air_pres_list = [100000]
        wind_list = np.array([[0, 0], [50, 0]])

        self.world = World()
        self.world.add_object(Cube(measure=(PARAMETER_SIZE, PARAMETER_SIZE, DROP_HEIGHT), 
            pos=np.array([-PARAMETER_SIZE/2, -PARAMETER_SIZE/2, -DROP_HEIGHT/2])))
        # self.world.add_object(Particle(pos=np.array([0., 0., 0.])))
        self.world.add_object_cluster(Wind(wind_list, 4000))
        self.world.add_object(Diver(x=np.array([0.,0.,3600.]), vel=np.array([-200.,0.,0.]), 
            wind=wind_list, air_pressure=air_pres_list, temperature=temp_list))

        self.cur_rotation = np.array([0,0,0])

    def draw_screen(self):
        game_display.fill(BACKGROUND_COLOR)
        self.world.draw()

    def update_world(self):
        self.world.rotate(self.cur_rotation)
        self.world.update()
        
    def game_loop(self):     

        while True:

            # Controls
            for event in pygame.event.get():

                # Keyboard Presses
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        self.cur_rotation[0] = 1
                    elif event.key == pygame.K_w:
                        self.cur_rotation[0] = -1
                    if event.key == pygame.K_a:
                        self.cur_rotation[2] = 1
                    elif event.key == pygame.K_d:
                        self.cur_rotation[2] = -1

                    # if event.key == pygame.K_KP6:
                    #     x_vel = 5
                    # elif event.key == pygame.K_KP4:
                    #     x_vel = -5
                    # if event.key == pygame.K_KP8:
                    #     z_vel = -5
                    # elif event.key == pygame.K_KP2:
                    #     z_vel = 5

                elif event.type == pygame.KEYUP:
                    self.cur_rotation = np.array([0,0,0])

                if event.type == pygame.MOUSEWHEEL:
                    if event.y == -1:
                        self.world.camera_pos[1] -= 500
                    elif event.y == 1:
                        self.world.camera_pos[1] += 500

                # Quit
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # Update
            self.update_world()
            self.draw_screen()

            # Update
            pygame.display.update()
            clock.tick(FPS)

# -------------------------------- Run -----------------------------------------




Game().game_loop()

# de particle laat geen lijn zien.