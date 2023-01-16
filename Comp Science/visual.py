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
FPS = 30
BACKGROUND_COLOR = black

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

# -----------------------------  Variables -------------------------------------



# ---------------------------- Setup display -----------------------------------

game_display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.display.set_caption(GAME_NAME)
clock = pygame.time.Clock()

# ------------------------------ Functions -------------------------------------

def project(pos : np.array):
    return np.dot(np.array([[1, 0, 0],[0, 1, 0]]), pos)

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

    def __init__(self):

        # self.a = np.array([0,0,0])
        # self.b = np.array([1,0,0])
        # self.c = np.array([1,1,0])
        # self.d = np.array([1,0,1])
        # self.e = np.array([1,1,1])
        # self.f = np.array([0,1,0])
        # self.g = np.array([0,1,1])
        # self.h = np.array([0,0,1])

        self.nodes = [np.array([0,0,0]), np.array([1,0,0]), np.array([1,1,0]), 
                      np.array([1,0,1]), np.array([1,1,1]), np.array([0,1,0]), 
                      np.array([0,1,1]), np.array([0,0,1])]

        self.edges = [[self.nodes[0], self.nodes[1]], [self.nodes[0], self.nodes[5]], [self.nodes[0], self.nodes[7]], 
                      [self.nodes[2], self.nodes[1]], [self.nodes[2], self.nodes[5]], [self.nodes[2], self.nodes[4]], 
                      [self.nodes[6], self.nodes[4]], [self.nodes[6], self.nodes[7]], [self.nodes[6], self.nodes[5]], 
                      [self.nodes[3], self.nodes[7]], [self.nodes[3], self.nodes[1]], [self.nodes[3], self.nodes[4]]]

        self.color = white
    
    def draw(self):

        # # Draw nodes.
        # for node in self.nodes:
        #     projection = tuple(project(node) * 200 + 400)
        #     pygame.draw.circle(game_display, self.color, projection, 1)
        
        # Draw edges.
        for edge in self.edges:
            (a, b) = edge
            a_p = tuple(project(a) * 200 + 400)
            b_p = tuple(project(b) * 200 + 400)
            pygame.draw.line(game_display, self.color, a_p, b_p)

    def move(self):
        new_nodes = []
        rot = 2 * np.pi / 360
        for node in self.nodes:
            new_nodes.append(rotate(node, rot, rot, rot))
        self.nodes = new_nodes

        self.edges = [[self.nodes[0], self.nodes[1]], [self.nodes[0], self.nodes[5]], [self.nodes[0], self.nodes[7]], 
                      [self.nodes[2], self.nodes[1]], [self.nodes[2], self.nodes[5]], [self.nodes[2], self.nodes[4]], 
                      [self.nodes[6], self.nodes[4]], [self.nodes[6], self.nodes[7]], [self.nodes[6], self.nodes[5]], 
                      [self.nodes[3], self.nodes[7]], [self.nodes[3], self.nodes[1]], [self.nodes[3], self.nodes[4]]]

    
# -------------------------------- Game ----------------------------------------

class Game:

    def __init__(self):
        self.cube = Cube()

    def draw_screen(self):
        game_display.fill(BACKGROUND_COLOR)
        self.cube.draw()
        # print(self.cube.edges)

    def change(self):
        self.cube.move()
        # print(self.cube.nodes)

    def game_loop(self):

        # Loop
        while True:

            # Controls
            for event in pygame.event.get():

                # Quit
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # Background
            self.change()
            self.draw_screen()

            # Update
            pygame.display.update()
            clock.tick(FPS)


# -------------------------------- Run -----------------------------------------
Game().game_loop()


# meerdere keren drukken op de zelfde tile creert 2 tiles op de zelfde pos.