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
yellow =        (255,   255,    0)
purple =        (255,   0,      255)

# ------------------------------ Settings --------------------------------------

GAME_NAME = 'SkyDive'
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
    def __init__(self, size : int, direction : np.array, pos: np.array, color=blue):

        self.color = color
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

class Line:

    def __init__(self):
        self.nodes = []
        self.edges = []

        self.color = red
        self.line_width = 1

    def add_pos(self, pos):
        self.nodes.append(Node(pos))
        if len(self.nodes) >= 2:
            self.edges.append(Edge(self.nodes[-2], self.nodes[-1]))
    
class Ball:
    def __init__(self, pos, color, size):
        self.pos = pos
        self.color = color
        self.size = size

class Text_X:

    def __init__(self, pos, size, color):
        self.pos = pos
        self.color = color
        self.line_width = 2
        self.nodes = [Node(pos + np.array([-size, 0, -size])), Node(pos + np.array([size, 0, -size])),
                      Node(pos + np.array([-size, 0, size])), Node(pos + np.array([size, 0, size]))]
        self.edges = [Edge(self.nodes[0], self.nodes[3]), Edge(self.nodes[1], self.nodes[2])]

class Text_Y:

    def __init__(self, pos, size, color):
        self.pos = pos
        self.color = color
        self.line_width = 2
        self.nodes = [Node(pos + np.array([0, -size, -size])), Node(pos + np.array([0, size, -size])),
                      Node(pos + np.array([0, -size, size])), Node(pos + np.array([0, 0, 0]))]
        self.edges = [Edge(self.nodes[0], self.nodes[3]), Edge(self.nodes[1], self.nodes[2])]
    
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

    def update(self, next_pos, end):
        
        for obj in self.objects:
            if 'move' in dir(obj):
                obj.move()

            if 'add_pos' in dir(obj) and not end:
                obj.add_pos(next_pos)

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
        if 'edges' in dir(obj):
            for edge in obj.edges:
                a, b = edge.node1.pos, edge.node2.pos
                a_s = self._rotate_pos(a)
                b_s = self._rotate_pos(b)
                a_p = tuple(self._project_pos(a_s) * self.zoom)
                b_p = tuple(self._project_pos(b_s) * self.zoom)

                pygame.draw.line(game_display, obj.color, a_p, b_p, obj.line_width)

        # Point object.
        else:
            a = obj.pos
            a_s = self._rotate_pos(a)
            a_p = tuple(self._project_pos(a_s) * self.zoom)
            pygame.draw.circle(game_display, obj.color, a_p, obj.size)

# -------------------------------- Game ----------------------------------------

class Visual:

    def __init__(self, wind: list, air_pressure: list, temperature: list):

        self.world = World()

        self.world.add_object(Cube(measure=(PARAMETER_SIZE, PARAMETER_SIZE, DROP_HEIGHT), 
            pos=np.array([-PARAMETER_SIZE/2, -PARAMETER_SIZE/2, -DROP_HEIGHT/2])))

        self.world.add_object_cluster(Wind(wind, 4000))
        
        self.world.add_object(Arrow(500, np.array([1,0]), np.array([PARAMETER_SIZE/2 + 200, -PARAMETER_SIZE/2, DROP_HEIGHT/2]), green))
        self.world.add_object(Arrow(500, np.array([0,1]), np.array([-PARAMETER_SIZE/2, PARAMETER_SIZE/2 + 200, DROP_HEIGHT/2]), purple))
        self.world.add_object(Text_X(np.array([PARAMETER_SIZE/2 + 1000, -PARAMETER_SIZE/2, DROP_HEIGHT/2]), 100, green))
        self.world.add_object(Text_Y(np.array([-PARAMETER_SIZE/2, PARAMETER_SIZE/2 + 1000, DROP_HEIGHT/2]), 100, purple))

        self.cur_rotation = np.array([0,0,0])
        self.drawn2 = False

    def add_diver(self, diver):
        self.diver = diver
        self.world.add_object(Line())

    def select_line_interval(self, interval: int):
        self.traject = [pos for c, pos in enumerate(self.diver.pos_list) if c % interval == 0]

    def _draw_screen(self):
        game_display.fill(BACKGROUND_COLOR)
        self.world.draw()

    def _update_world(self):
        self.world.rotate(self.cur_rotation)

        # Update trajectory
        if self.frame + 1 < len(self.traject):
            if self.time % self.slowed == 0:
                self.next_pos = self.traject[self.frame]
                self.end = False
                self.frame += 1

            # Show phase points.
            if self.traject[self.frame][2] > self.diver.shute_pos and not self.drawn2:
                self.world.add_object(Ball(self.traject[self.frame], yellow, 4))
                self.drawn2 = True

        else:
            self.world.add_object(Ball(self.traject[self.frame], blue, 4))
            self.next_pos = None
            self.end = True
        self.world.update(self.next_pos, self.end)
        
    def run(self, slowed):     
        self.slowed = slowed
        self.time = 0
        self.frame = 0

        self.world.add_object(Ball(self.traject[self.frame], green, 4))

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
            self._update_world()
            self._draw_screen()

            # Update
            pygame.display.update()
            clock.tick(FPS)
            self.time += 1

# -------------------------------- Run -----------------------------------------




# Game().game_loop()

# de particle laat geen lijn zien.