import pygame
import numpy as np
from typing import List


class GameVisualizer:
    
    SCREEN_WIDTH = 1000
    
    def __init__(self, grid_size: int, window_width: int = None, window_height: int = None):
        """
        Initializes the visualizer.
        
        :param grid_size: Size of the grid (grid_size x grid_size)
        :param window_width: Ignored - uses fixed SCREEN_WIDTH instead
        :param window_height: Ignored - uses fixed SCREEN_HEIGHT instead
        """
        pygame.init()
        self.clock = pygame.time.Clock()
        
        self.grid_size = grid_size
        # Use fixed screen dimensions
        self.win = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_WIDTH])
        # Scale cell sizes based on fixed screen width and grid size
        self.cell_width = self.SCREEN_WIDTH // grid_size
        self.cell_height = self.SCREEN_WIDTH // grid_size
        
        # Colors
        self.ALIVE_COLOR = pygame.Color("yellow")
        self.DEAD_COLOR = pygame.Color("white")
        self.GRID_COLOR = pygame.Color("gray")
        
        pygame.display.set_caption("Conway's Game of Life")
    
    def display_grid(self, grid: np.ndarray, center_row: int = None, center_col: int = None):
        """
        Displays a single grid state.
        
        :param grid: 2D numpy array where 1 = alive, 0 = dead
        :param center_row: Row to center the view on (if grid is larger than display)
        :param center_col: Column to center the view on (if grid is larger than display)
        """
        # Fill background
        self.win.fill(self.DEAD_COLOR)
        
        # Determine viewport if grid is larger than display
        if grid.shape[0] > self.grid_size or grid.shape[1] > self.grid_size:
            if center_row is None:
                center_row = grid.shape[0] // 2
            if center_col is None:
                center_col = grid.shape[1] // 2
            
            # Calculate viewport bounds
            start_row = max(0, center_row - self.grid_size // 2)
            end_row = min(grid.shape[0], start_row + self.grid_size)
            start_col = max(0, center_col - self.grid_size // 2)
            end_col = min(grid.shape[1], start_col + self.grid_size)
            
            # Adjust if we hit boundaries
            if end_row - start_row < self.grid_size:
                start_row = max(0, end_row - self.grid_size)
            if end_col - start_col < self.grid_size:
                start_col = max(0, end_col - self.grid_size)
        else:
            # Grid fits in display, show it all
            start_row = 0
            end_row = grid.shape[0]
            start_col = 0
            end_col = grid.shape[1]
        
        # Draw cells
        display_rows = min(self.grid_size, end_row - start_row)
        display_cols = min(self.grid_size, end_col - start_col)
        
        for i in range(display_rows):
            for j in range(display_cols):
                grid_row = start_row + i
                grid_col = start_col + j
                
                if grid_row < grid.shape[0] and grid_col < grid.shape[1] and grid[grid_row, grid_col] == 1:
                    x = j * self.cell_width
                    y = i * self.cell_height
                    pygame.draw.rect(
                        self.win,
                        self.ALIVE_COLOR,
                        pygame.Rect(x, y, self.cell_width, self.cell_height)
                    )
        
        # Draw grid lines
        for i in range(display_rows + 1):
            y = i * self.cell_height
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
    
    def display_sequence(self, grids: List[np.ndarray], delay_ms: int = 100, auto_advance: bool = True):
        """
        Displays a sequence of grids (e.g., from simulate() log).
        
        :param grids: List of 2D numpy arrays representing grid states
        :param delay_ms: Delay between frames in milliseconds
        :param auto_advance: If True, automatically advances frames. If False, waits for keypress.
        """
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
                        # Pause/unpause
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
                pygame.time.delay(delay_ms)
    
    def update_display(self, fps: int = 60):
        """
        Updates the display and handles timing.
        
        :param fps: Target frames per second
        """
        pygame.display.flip()
        self.clock.tick(fps)
    
    def get_event(self):
        """
        Gets pygame events. Returns "exit" if user wants to quit.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "exit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return "exit"
        return None
    
    def quit(self):
        pygame.quit()
