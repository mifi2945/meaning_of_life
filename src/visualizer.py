from typing import List

import pygame.sprite

from environment import Thing, Dust

# from environment import DustCleanUp as dc

WALKABLE_COLOR = pygame.Color("white")
IMPASSABLE_COLOR = pygame.Color("black")
AGENT_COLOR = pygame.Color("yellow")
DUST_COLOR = pygame.Color("gray")
DISPLAY_COLOR = pygame.Color("white")
START_COLOR = pygame.Color("green")
GOAL_COLOR = pygame.Color("red")

CELL_WIDTH = 100
CELL_HEIGHT = 100


def make_tile(width: int, height: int, tile_color: pygame.Color) -> pygame.Surface:
    """
    Creates a simple tile that can be drawn onto the display.

    :param width: Width of the tile in pixels.
    :param height: Height of the tile in pixels.
    :param tile_color: Color to file the tile with.
    :return: Surface that represents the tile.
    """
    image = pygame.Surface([width, height])
    pygame.draw.rect(image, tile_color, pygame.Rect(0, 0, width, height))
    pygame.draw.rect(image, pygame.Color("black"), pygame.Rect(0, 0, width, height), width=1)
    return image


def make_entity(width: int, height: int, tile_color: pygame.Color, entity_color: pygame.Color) -> pygame.Surface:
    """
    Makes a tile with a circle on it to represent an entity like a piece of dust or agent.

    :param width:
    :param height:
    :param agent_color:
    :return:
    """
    image = make_tile(width, height, tile_color)
    image.set_colorkey(tile_color)

    rect = image.get_rect()
    cx = rect.width/2
    cy = rect.height/2
    pygame.draw.circle(image, entity_color, (cx,cy), (rect.width/2) * .5, 0)
    return image


class MazeVisualizer:
    """
    Class for visualizing a maze navigation task.

    The two main functions in this class are display_state() and run_actions().
    display_state() allows you to pass in a state one at a time and display it.
    This is usually for dynamic environments where you only know the agents
    next action. run_actions() allows you to run a series of actions. This is
    useful for static environments like those in a search since you have all
    the moves before the agent even starts.
    """
    def __init__(self, width:int, height: int, cols:int, rows: int):
        """
        Initializes the visualizer.

        When this class is made, pygame is initialized and the display is set.
        :param width: Width of the display window.
        :param height: Height of the display window.
        """

        pygame.init()
        self.clock = pygame.time.Clock()

        self.cols = cols
        self.rows = rows
        self.win = pygame.display.set_mode([width, height])
        self.cell_width = self.win.get_width()//cols
        self.cell_height = self.win.get_height()//rows
        pygame.display.set_caption("Maze Visualizer")

    def get_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("exiting")
                return "exit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return "exit"

    def display_maze(self, things:list[Thing], location:list[(int,int)] = None):
        self.draw_grid()

        if location is None:
            location = [t.location for t in things]

        for i, thing in enumerate(things):
            upper_left = (location[i][0] * self.cell_width,
                          location[i][1] * self.cell_height)
            center = (upper_left[0]+self.cell_width//2, upper_left[1]+self.cell_height//2)

            c = pygame.Color(thing.color)
            if thing.shape == "circle":
                # if type(thing) == Dust:
                #     print(f"Drawing dust at {center}")
                pygame.draw.circle(self.win, c, center, thing.d1 * (self.cell_width + self.cell_height)//4)
            else: #if thing.shape == "rectangle":
                r = pygame.Rect(0, 0, self.cell_width * thing.d1, self.cell_height * thing.d1)
                r.center = center
                pygame.draw.rect(self.win, c, r)

    def quit(self) -> None:
        """
        Simple method that will allow quitting of pygame from outside of this class.
        """
        print(f"Closing visualizer.")
        pygame.quit()

    def update_display(self, delay: int = 0):
        pygame.display.flip()
        self.clock.tick(60)
        pygame.time.delay(delay)

    def draw_grid(self):
        self.win.fill(pygame.Color("white"))

        for j in range(self.cols):
            pygame.draw.line(self.win, pygame.Color("black"),
                             (j * self.cell_width, 0), (j * self.cell_width, self.win.get_height()))
        for i in range(self.rows):
            pygame.draw.line(self.win, pygame.Color("black"),
                             (0, i * self.cell_height), (self.win.get_width(), i * self.cell_height))


