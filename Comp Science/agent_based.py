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

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

NR_BUNNIES = 30
GRASS_GROW_RATE = 100

# -----------------------------  Variables -------------------------------------



# ---------------------------- Setup display -----------------------------------

game_display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.display.set_caption(GAME_NAME)
clock = pygame.time.Clock()

# ------------------------------ Functions -------------------------------------



# ------------------------------- Objects --------------------------------------

class Bunny:

    def __init__(self):
        self.x = random.randrange(SCREEN_WIDTH)
        self.y = random.randrange(SCREEN_HEIGHT)
        self.size = 10
        self.color = brown
        self.speed = 5

    def draw(self):
        pygame.draw.rect(game_display, self.color, [self.x, self.y, 
                         self.size, self.size])

    def move(self):
        self._move_random()

    def _move_random(self):
        x_delta = random.randrange(-1, 2)
        y_delta = random.randrange(-1, 2)
        self.x += x_delta * self.speed
        self.y += y_delta * self.speed

class Grass:
    def __init__(self):
        self.x = random.randrange(SCREEN_WIDTH)
        self.y = random.randrange(SCREEN_HEIGHT)
        self.size = 5
        self.color = green

    def draw(self):
        pygame.draw.rect(game_display, self.color, [self.x, self.y, 
                         self.size, self.size])
    
    def move(self):
        pass

# -------------------------------- Game ----------------------------------------

class Game:

    def __init__(self):
        self.entities = []
        self._spawn_entities()

    def _spawn_entities(self):
        for _ in range(NR_BUNNIES):
            self.entities.append(Bunny())

    def draw_entities(self):
        for entity in self.entities:
            entity.draw()

    def move_entities(self):
        for entity in self.entities:
            entity.move()

        r = random.randrange(100)
        if r < GRASS_GROW_RATE:
            self.entities.append(Grass())

    def draw_screen(self):
        """ Here stand all the functions that draw every frame.
        """
        game_display.fill(BACKGROUND_COLOR)
        self.draw_entities()

    def game_loop(self):
        game_exit = False

        # Loop
        while not game_exit:

            # Controls
            for event in pygame.event.get():

                # Quit
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # Background
            self.move_entities()
            self.draw_screen()

            # Update
            pygame.display.update()
            clock.tick(FPS)


# -------------------------------- Run -----------------------------------------
Game().game_loop()


# meerdere keren drukken op de zelfde tile creert 2 tiles op de zelfde pos.