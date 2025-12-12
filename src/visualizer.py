"""
Conway's Game of Life Visualization

PyGame-based visualizer for displaying Conway's Game of Life simulations.
Supports interactive playback with keyboard controls (space to play/pause, arrow keys to navigate).
Visualizer defaults to paused and you must press space to start the animation.
"""

import pygame
import numpy as np
from typing import List
import random

DELAY = 100
GRID_SIZE_PIXELS = 1000
GRID_WIDTH = 100

class GameVisualizer:
    """
    PyGame-based visualizer for Conway's Game of Life simulations.
    
    Uses PyGame to show the game
    """
    
    def __init__(self, grid_width: int = None, delay: int = DELAY, auto_fit: bool = True):
        """
        Initialize the GameVisualizer with display settings.
        
        Args:
            grid_width: Number of cells to display horizontally/vertically. If None,
                       will be auto-fitted based on the first grid displayed (capped at GRID_WIDTH).
            delay: Delay in milliseconds between frames during auto-playback (default: 100).
            auto_fit: Whether to automatically fit grid size to display window (default: True).
        
        Initializes PyGame, creates a display window, and sets up colors for visualization.
        """
        pygame.init()
        self.clock = pygame.time.Clock()
        
        self.grid_width = grid_width
        self.delay = delay
        self.auto_fit = auto_fit
        self.win = pygame.display.set_mode([GRID_SIZE_PIXELS, GRID_SIZE_PIXELS])
        self.cell_width = GRID_SIZE_PIXELS // grid_width
        
        self.ALIVE_COLOR = pygame.Color(*self._random_contrasty_color())
        self.DEAD_COLOR = pygame.Color("white")
        self.GRID_COLOR = pygame.Color("gray")

        
        pygame.display.set_caption("Conway's Game of Life")

    def _random_contrasty_color(self):
        """
        AI-generated random color generator for the living cells, for the spunk.
        
        Uses luminance calculation to ensure the color is dark enough to be visible
        on a white background. Continues generating colors until one with luminance < 220 is found.
        
        Returns:
            tuple: RGB color tuple (r, g, b) with values 0-255, guaranteed to be visible on white.
        """
        while True:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            if luminance < 220:
                return (r, g, b)
    
    def _fit_to_grid(self, width: int):
        """
        Adjust display settings to fit the given grid width.
        
        If grid_width was not specified during initialization, this method sets it
        based on the actual grid size
        Recalculates cell_width to ensure proper scaling.
        
        Args:
            width: The width (and height, assuming square grid) of the grid to display.
        """
        # Use the provided grid_width if set, otherwise fit to actual grid size (capped at GRID_WIDTH for performance)
        if self.grid_width is None:
            self.grid_width = min(width, GRID_WIDTH)
        self.cell_width = GRID_SIZE_PIXELS // self.grid_width
    
    def display_grid(self, grid: np.ndarray):
        """
        Display a single grid state on the PyGame window.
        
        Called for every step of the simulation.
        
        Args:
            grid: 2D numpy array representing the game state (1 = alive, 0 = dead).
        """
        self.win.fill(self.DEAD_COLOR)
        
        start_row = 0
        end_row = grid.shape[0]
        start_col = 0
        end_col = grid.shape[1]
    
        display_rows = end_row - start_row
        display_cols = end_col - start_col
        
        for i in range(display_rows):
            for j in range(display_cols):
                grid_row = start_row + i
                grid_col = start_col + j
                
                if grid_row < grid.shape[0] and grid_col < grid.shape[1] and grid[grid_row, grid_col] == 1:
                    x = j * self.cell_width
                    y = i * self.cell_width
                    pygame.draw.rect(
                        self.win,
                        self.ALIVE_COLOR,
                        pygame.Rect(x, y, self.cell_width, self.cell_width)
                    )
        
        # Draw grid lines
        for i in range(display_rows + 1):
            y = i * self.cell_width
            pygame.draw.line(
                self.win,
                self.GRID_COLOR,
                (0, y),
                (self.win.get_width(), y)
            )
        for j in range(display_cols + 1):
            x = j * self.cell_width
            pygame.draw.line(
                self.win,
                self.GRID_COLOR,
                (x, 0),
                (x, self.win.get_height())
            )
    
    def display_sequence(self, grids: List[np.ndarray], auto_advance: bool = False):
        """
        Display a sequence of grid states with interactive controls.
        
        Shows each grid state in sequence, allowing the user to control playback.
        The visualization starts paused by default; press SPACE to start auto-playback.
        
        Keyboard Controls:
            SPACE: Toggle auto-advance (play/pause animation)
            RIGHT ARROW: Advance to next frame manually
            LEFT ARROW: Go back to previous frame manually
            Q or ESC: Quit the visualization
        
        Args:
            grids: List of 2D numpy arrays, each representing a game state at a different time step.
            auto_advance: Whether to automatically advance frames (default: False, starts paused).
        
        The method runs until the user quits or all frames have been displayed.
        """
        self._fit_to_grid(grids[0].shape[0])
        
        running = True
        frame_index = 0
        
        while running and frame_index < len(grids):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        auto_advance = not auto_advance
                    elif event.key == pygame.K_RIGHT:
                        # Next frame
                        frame_index = min(frame_index + 1, len(grids) - 1)
                    elif event.key == pygame.K_LEFT:
                        # Previous frame
                        frame_index = max(frame_index - 1, 0)
            
            # Display current frame
            self.display_grid(grids[frame_index])
            self.update_display()
            
            # Auto-advance if enabled
            if auto_advance:
                frame_index += 1
                pygame.time.delay(self.delay)
    
    def update_display(self):
        """
        Update the PyGame display window.
        
        Flips the display buffer to show the current frame and updates the clock.
        Should be called after drawing operations to make changes visible.
        """
        pygame.display.flip()
        self.clock.tick()
    
    def quit(self):
        pygame.quit()
