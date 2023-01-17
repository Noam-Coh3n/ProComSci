# ------------------------------ Import ----------------------------------------

import time
import random
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
FPS = 60
BACKGROUND_COLOR = black

TILE_SIZE = 10
BOARD_WIDTH = 50 # maak het even
BOARD_HEIGHT = 50 # maak het even
LINE_WIDTH = 5


# -----------------------------  Variables -------------------------------------

screen_width = BOARD_WIDTH * (TILE_SIZE + LINE_WIDTH) + LINE_WIDTH
screen_height = BOARD_HEIGHT * (TILE_SIZE + LINE_WIDTH) + LINE_WIDTH

# ---------------------------- Setup display -----------------------------------

game_display = pygame.display.set_mode((screen_width, screen_height))

pygame.display.set_caption(GAME_NAME)
clock = pygame.time.Clock()

# ------------------------------ Functions -------------------------------------

def pos2cor(pos):
    """ Get tile position and return pixel coordinates of up left corner.
    """
    (x,y) = pos
    return (x+1)*LINE_WIDTH + x*TILE_SIZE, (y+1)*LINE_WIDTH + y*TILE_SIZE

def cor2pos(cor):
    """ Get pixel coordinates and return tile position.
    """
    (x,y) = cor
    sqr = TILE_SIZE + LINE_WIDTH
    return (int(x / sqr), int(y / sqr))

# ------------------------------- Objects --------------------------------------

class Tile:

    def __init__(self, pos):
        x, y = pos2cor(pos)
        self.rect = [x, y, TILE_SIZE, TILE_SIZE]
        self.color = sand

        y = pos[1]
        self.y_percentage = abs((y - (BOARD_HEIGHT-1)/2) / (BOARD_HEIGHT/2))

    def draw(self):
        pygame.draw.rect(game_display, self.color, self.rect)

# -------------------------------- Game ----------------------------------------

class Game:

    def __init__(self):
        self.tiles = []

    def highlight_tile(self):
        """ Draw a white square around the tile where the mouse is located. 
        """
        mouse_pos = pygame.mouse.get_pos()
        x,y = pos2cor(cor2pos(mouse_pos))
        pygame.draw.rect(game_display, white, [x, y, TILE_SIZE, TILE_SIZE], 1)

    def insert_tile(self):
        """ Create new Tile object and place it on the pos of the mouse.
        """
        mouse_pos = pygame.mouse.get_pos()
        pos = cor2pos(mouse_pos)
        self.tiles.append(Tile(pos))

    def draw_tiles(self):
        """ Draw all tiles.
        """
        for tile in self.tiles:
            tile.draw()

    def draw_screen(self):
        """ Here stand all the functions that draw every frame.
        """
        game_display.fill(BACKGROUND_COLOR)
        self.highlight_tile()
        self.draw_tiles()

    def game_loop(self):
        game_exit = False

        # Loop
        while not game_exit:

            # Controls
            for event in pygame.event.get():

                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.insert_tile()

                # Quit
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # Background
            self.draw_screen()

            # Update
            pygame.display.update()
            clock.tick(FPS)


# -------------------------------- Run -----------------------------------------
Game().game_loop()


# meerdere keren drukken op de zelfde tile creert 2 tiles op de zelfde pos.