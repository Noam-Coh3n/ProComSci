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

GAME_NAME = 'Ecosystem'
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

    def __init__(self, measure : tuple, zoom : int, pos : tuple):

        self.zoom = zoom
        self.pos = pos
        self.color = white

        a = measure[0]
        b = measure[1]
        c = measure[2]

        self.nodes = [np.array([0,0,0]), np.array([a,0,0]), np.array([a,b,0]), 
                      np.array([a,0,c]), np.array([a,b,c]), np.array([0,b,0]), 
                      np.array([0,b,c]), np.array([0,0,c])]

        self.edges = [[self.nodes[0], self.nodes[1]], [self.nodes[0], self.nodes[5]], [self.nodes[0], self.nodes[7]], 
                      [self.nodes[2], self.nodes[1]], [self.nodes[2], self.nodes[5]], [self.nodes[2], self.nodes[4]], 
                      [self.nodes[6], self.nodes[4]], [self.nodes[6], self.nodes[7]], [self.nodes[6], self.nodes[5]], 
                      [self.nodes[3], self.nodes[7]], [self.nodes[3], self.nodes[1]], [self.nodes[3], self.nodes[4]]]
    
    def draw(self):
        for edge in self.edges:
            (a, b) = edge
            a_p = tuple(project(a) * self.zoom + np.array(self.pos))
            b_p = tuple(project(b) * self.zoom + np.array(self.pos))
            pygame.draw.line(game_display, self.color, a_p, b_p)

    def move(self, x_rot, y_rot, z_rot, x_vel, z_vel):
        new_nodes = []
        rot = 2 * np.pi / 360
        for node in self.nodes:
            new_nodes.append(rotate(node, rot * x_rot, rot * y_rot, rot * z_rot))
        self.nodes = new_nodes

        self.edges = [[self.nodes[0], self.nodes[1]], [self.nodes[0], self.nodes[5]], [self.nodes[0], self.nodes[7]], 
                      [self.nodes[2], self.nodes[1]], [self.nodes[2], self.nodes[5]], [self.nodes[2], self.nodes[4]], 
                      [self.nodes[6], self.nodes[4]], [self.nodes[6], self.nodes[7]], [self.nodes[6], self.nodes[5]], 
                      [self.nodes[3], self.nodes[7]], [self.nodes[3], self.nodes[1]], [self.nodes[3], self.nodes[4]]]

        self.pos[0] += x_vel
        self.pos[1] += z_vel

    
# -------------------------------- Game ----------------------------------------

class Game:

    def __init__(self):
        self.cube = Cube(measure=(1,1,1), zoom=500, pos=[800, 100])

    def draw_screen(self):
        game_display.fill(BACKGROUND_COLOR)
        self.cube.draw()
        # print(self.cube.edges)

    def change(self, x_rot, y_rot, z_rot, x_vel, z_vel):
        self.cube.move(x_rot, y_rot, z_rot, x_vel, z_vel)
        # print(self.cube.nodes)

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
                        self.cube.zoom -= 30
                    
                    if event.y == 1:
                        self.cube.zoom += 30

                # Quit
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # Background
            self.change(x_rot, y_rot, z_rot, x_vel, z_vel)
            self.draw_screen()

            # Update
            pygame.display.update()
            clock.tick(FPS)


# -------------------------------- Run -----------------------------------------
Game().game_loop()
