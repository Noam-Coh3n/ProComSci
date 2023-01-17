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
FPS = 45
BACKGROUND_COLOR = black

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

# -----------------------------  Variables -------------------------------------

# ---------------------------- Setup display -----------------------------------

game_display = pygame.display.set_mode()

pygame.display.set_caption(GAME_NAME)
clock = pygame.time.Clock()

# ------------------------------ Functions -------------------------------------

def project(pos : np.array):
    y = 1 / (3 - pos[1])
    return np.dot(np.array([[y, 0, 0],[0, 0, y]]), pos)

def rotate(pos : np.array, x_angle : int, y_angle : int, z_angle : int):

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

class Cube:

    def __init__(self, measure : tuple, pos : np.array, pos0 : list, zoom : int):

        self.pos0 = pos0
        self.color = white
        self.zoom = zoom

        a = measure[0]
        b = measure[1]
        c = measure[2]

        x = np.array([a,0,0])
        y = np.array([0,b,0])
        z = np.array([0,0,c])

        self.nodes = [pos, pos + x, pos + x + y, 
                      pos + x + z, pos + x + y + z, pos + y, 
                      pos + y + z, pos + z]

        self.edges = [[self.nodes[0], self.nodes[1]], [self.nodes[0], self.nodes[5]], [self.nodes[0], self.nodes[7]], 
                      [self.nodes[2], self.nodes[1]], [self.nodes[2], self.nodes[5]], [self.nodes[2], self.nodes[4]], 
                      [self.nodes[6], self.nodes[4]], [self.nodes[6], self.nodes[7]], [self.nodes[6], self.nodes[5]], 
                      [self.nodes[3], self.nodes[7]], [self.nodes[3], self.nodes[1]], [self.nodes[3], self.nodes[4]]]
    
    def draw(self):
        for edge in self.edges:
            (a, b) = edge
            a_p = tuple(project(a) * self.zoom + np.array(self.pos0))
            b_p = tuple(project(b) * self.zoom + np.array(self.pos0))
            pygame.draw.line(game_display, self.color, a_p, b_p)

    def change(self, x_rot, y_rot, z_rot, x_vel, z_vel, zoom):
        new_nodes = []
        rot = 2 * np.pi / 360
        for node in self.nodes:
            new_nodes.append(rotate(node, rot * x_rot, rot * y_rot, rot * z_rot))
        self.nodes = new_nodes

        self.edges = [[self.nodes[0], self.nodes[1]], [self.nodes[0], self.nodes[5]], [self.nodes[0], self.nodes[7]], 
                      [self.nodes[2], self.nodes[1]], [self.nodes[2], self.nodes[5]], [self.nodes[2], self.nodes[4]], 
                      [self.nodes[6], self.nodes[4]], [self.nodes[6], self.nodes[7]], [self.nodes[6], self.nodes[5]], 
                      [self.nodes[3], self.nodes[7]], [self.nodes[3], self.nodes[1]], [self.nodes[3], self.nodes[4]]]

        self.pos0[0] += x_vel
        self.pos0[1] += z_vel
        self.zoom = zoom

class Particle:

    def __init__(self, pos : np.array, pos0 : list, zoom : int):
        self.pos = pos
        self.pos0 = pos0
        self.zoom = zoom
        self.color = red
        self.line = [pos]

    def draw(self):
        pos_p = tuple(project(self.pos) * self.zoom + np.array(self.pos0))
        pygame.draw.circle(game_display, self.color, pos_p, 3)

        a = self.line[0]
        for b in self.line[1:]:
            a_p = tuple(project(a) * self.zoom + np.array(self.pos0))
            b_p = tuple(project(b) * self.zoom + np.array(self.pos0))
            pygame.draw.line(game_display, self.color, a_p, b_p)
            a = b

    def change(self, x_rot, y_rot, z_rot, x_vel, z_vel, zoom):
        rot = 2 * np.pi / 360
        self.pos = rotate(self.pos, rot * x_rot, rot * y_rot, rot * z_rot)
        self.pos0[0] += x_vel
        self.pos0[1] += z_vel
        self.zoom = zoom

        new_line = []
        for point in self.line:
            new_line.append(rotate(point, rot * x_rot, rot * y_rot, rot * z_rot))
        self.line = new_line

    def move(self):
        x = random.randrange(-1,2) * 0.05
        y = random.randrange(-1,2) * 0.05
        z = random.randrange(-1,2) * 0.05
        self.pos += np.array([x, y, z])
        self.line.append(self.pos)
    
# -------------------------------- Game ----------------------------------------

class Game:

    def __init__(self):
        self.zoom = 500
        self.pos0 = [1000, 500]

        self.cube = Cube(measure=(2,2,2), pos=np.array([-1, -1, -1]), pos0=self.pos0, zoom=self.zoom)
        self.particle = Particle(pos=np.array([0., 0., 0.]), pos0=self.pos0, zoom=self.zoom)

    def draw_screen(self):
        game_display.fill(BACKGROUND_COLOR)
        self.cube.draw()
        self.particle.draw()

    def change(self, x_rot, y_rot, z_rot, x_vel, z_vel, zoom):

        self.particle.move()

        self.cube.change(x_rot, y_rot, z_rot, x_vel, z_vel, zoom)
        self.particle.change(x_rot, y_rot, z_rot, x_vel, z_vel, zoom)

    def drag(self, on, a):
        if on:
            a = np.array(a)
            b = np.array(pygame.mouse.get_pos())
            print(a, b, b-a)
            x_rot = b-a


    def game_loop(self):
        x_rot, y_rot, z_rot, x_vel, z_vel = 0, 0, 0, 0, 0
        drag_on = False
        drag_a = 0

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
                        y_rot = 1
                    elif event.key == pygame.K_d:
                        y_rot = -1
                    if event.key == pygame.K_LEFT:
                        z_rot = 1
                    elif event.key == pygame.K_RIGHT:
                        z_rot = -1

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

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        drag_a = pygame.mouse.get_pos()
                        drag_on = True

                if drag_on:
                    drag_b = pygame.mouse.get_pos()
                    a_ar = np.array(drag_a)
                    b_ar = np.array(drag_b)
                    (x_rot, z_rot) = tuple(b_ar - a_ar)
                    drag_a = drag_b

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        drag_a = 0
                        drag_on = False

                # Quit
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # Background
            self.drag(drag_on, drag_a)
            self.change(x_rot, y_rot, z_rot, x_vel, z_vel, self.zoom)
            self.draw_screen()

            # Update
            pygame.display.update()
            clock.tick(FPS)


# -------------------------------- Run -----------------------------------------
Game().game_loop()


# keys a en d gebruik je wss niet, deze optie kan weg en dan kunne alle bewegingen op 1 d pad.