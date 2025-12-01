from __future__ import annotations  # needed in order to reference a Class within itself
from random import randint, random, choice, choices
from typing import Generic, TypeVar, Callable
from abc import ABC, abstractmethod
import numpy as np
import copy




STEPS = 10_000
INCLUDE_BETWEEN = True
EXPANSION = -1


# makes the state both immutable and hashable
T = TypeVar("T", bound=tuple)

class Problem(ABC, Generic[T]):
    """
    Interface representing a generic problem formulation.

    This class defines the operations needed for search algorithms.
    """



    @abstractmethod
    def value(self, curr_state: T) -> float:
        """
        Performance measure of the current state, used for local search.

        Args:
            curr_state (T): Current state.

        Returns:
            float: Fitness or score of the current state.
        """
        pass


class CGOL_Problem(Problem[T]):
    """
    Version of Problem used for Local Search algorithms Hill Climbing, Genetic Algorithm, and Novelty Search.

    Implements for Conway's Game of Life, with a list of values representing the 20x20 space.
    """
    def __init__(self, state_generator: Callable[[], T]):
        super().__init__(state_generator())
        self.state_generator = state_generator


    @staticmethod
    def simulate(
        initial: np.ndarray,
        steps: int=STEPS,
        include_between: bool=INCLUDE_BETWEEN,
        expansion: int=EXPANSION
    ) -> list[np.ndarray]:
        """
        Simulates Conway's Game of Life efficiently.
        
        Args:
            initial: 1D array representing the initial grid (will be reshaped to square)
            steps: Number of generations to simulate
            include_between: Whether to include intermediate states in the log
            size: Maximum grid size. If -1, allows infinite expansion from original grid.
                  Otherwise, only expands outwards as much as grid specifies.
        
        Returns:
            List of grid states (or just final state if include_between is False)
        """
        log = []
        
        # s is dimension of the intial grid
        s = int(len(initial) ** 0.5)

        size = s + steps if expansion == -1 else s + expansion

        
        M = np.zeros((size, size), dtype=initial.dtype)
        # (middle - half, middle + half) is where the initial grid goes in the new one
        middle = size // 2
        half = s // 2
        
        bounds = (middle - half, middle + half + (s%2))

        M[bounds[0]:bounds[1], bounds[0]:bounds[1]] = initial.reshape((s, s))

        if include_between:
            log.append(M.copy())
        
        for _ in range(steps):

            else:
                # For fixed size, ensure bounds don't exceed grid limits
                min_r, max_r, min_c, max_c = bounds
                bounds = (max(0, min_r), min(M.shape[0], max_r),
                         max(0, min_c), min(M.shape[1], max_c))
            
            # Extract region of interest with padding for neighbor counting
            min_r, max_r, min_c, max_c = bounds
            # Expand bounds by 1 for neighbor counting
            pad_min_r = max(0, min_r - 1)
            pad_max_r = min(M.shape[0], max_r + 1)
            pad_min_c = max(0, min_c - 1)
            pad_max_c = min(M.shape[1], max_c + 1)
            
            padded_region = M[pad_min_r:pad_max_r, pad_min_c:pad_max_c]
            region = M[min_r:max_r, min_c:max_c]
            
            # Count neighbors efficiently using numpy operations
            # Sum all 8 neighbors by shifting and summing
            neighbors = np.zeros(region.shape, dtype=np.uint8)
            
            # Calculate relative offsets within padded region
            r_rel = min_r - pad_min_r
            c_rel = min_c - pad_min_c
            
            # Count neighbors by summing all 8 shifted views
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r1 = r_rel + dr
                    r2 = r_rel + dr + region.shape[0]
                    c1 = c_rel + dc
                    c2 = c_rel + dc + region.shape[1]
                    
                    # Only add if the slice is valid
                    if r1 >= 0 and r2 <= padded_region.shape[0] and \
                       c1 >= 0 and c2 <= padded_region.shape[1]:
                        neighbors += padded_region[r1:r2, c1:c2]
            
            # Apply Game of Life rules
            new_region = np.zeros_like(region)
            # Live cells with 2-3 neighbors survive
            new_region[(region == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
            # Dead cells with exactly 3 neighbors become alive
            new_region[(region == 0) & (neighbors == 3)] = 1
            
            # Update grid
            M[min_r:max_r, min_c:max_c] = new_region
            
            # Update bounds to include new active cells (with 1-cell padding)
            # Find the bounding box of live cells
            live_cells = np.where(new_region == 1)
            if len(live_cells[0]) > 0:
                new_min_r = max(0, min_r + live_cells[0].min() - 1)
                new_max_r = min(M.shape[0], min_r + live_cells[0].max() + 2)
                new_min_c = max(0, min_c + live_cells[1].min() - 1)
                new_max_c = min(M.shape[1], min_c + live_cells[1].max() + 2)
                bounds = (new_min_r, new_max_r, new_min_c, new_max_c)
            else:
                # All cells dead, simulation can stop early
                break
            
            if include_between:
                log.append(M.copy())
        
        if not include_between:
            log.append(M.copy())
        
        return log


    def actions(self, curr_state: np.array[int,...]) -> list[int]:
        """
        Actions that can be applied to a CGoL board.
        We can technically set multiple cells at a time, but we will limit to one per epoch.

        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.

        Returns: List of cells that can be set.

        """
        return [i for i in range(len(curr_state))]

    def result(self, curr_state: np.array[int,...], action:int) -> np.array[int,...]:
        """
        Sets corresponding cell in the state.

        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.
            action: the index to set.

        Returns: the next state representation after the given action is performed.

        """
        next_state = curr_state.copy()
        next_state[action] ^= 1     # XOR toggle the cell

        return next_state


    def is_dead(self, curr_state: np.array[int,...]) -> bool:
        """
        Checks if the board has any cells alive.
        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.

        Returns: True if there are no cells alive, false otherwise.
        
        """
        return np.sum(curr_state) < 1

    def action_cost(self, curr_state: np.array[int,...], action: str, next_state: np.array[int,...]) -> float:
        return 1

    def quality(self, curr_state: np.array[int,...]) -> float:
        """
        Quality of the board.
        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.

        Returns: the quality value of the board.
        """

        # TODO figure this out lol
    
    def novelty(self, curr_state: np.array[int,...]) -> float:
        """
        Novelty of the board.
        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.

        Returns: the novelty value of the board

        """

    def fitness_proportionate_selection(self, fitness_values:list[float]) -> int:
        """
        Performs fitness proportionate selection.

        The probability of returning the index of an individual state in the population is
        proportional to its fitness value when compared to the fitness value of the entire population.
        If the fitness values are [2, 3, 1, 3, 1] then the sum is 10, which makes the probability of
        choosing each index to be [2/10, 3/10, 1/10, 3/10, 1/10] -> [0.2, 0.3, 0.1, 0.3, 0.1]

        :param fitness_values: List of fitness values for each individual in the population.
        :return: Index of individual to choose
        """
        probs = [val / sum(fitness_values) for val in fitness_values]
        indeces = [i for i in range(len(fitness_values))]
        return choices(indeces, weights=probs, k=1)[0]


    def selection(self, n:int, population: list[T],
                  weights: list[float]) -> [T,...]:
        """
        Selects n individuals from a population based on fitness proportionate selection method defined above.
        Args:
            n: Number of individuals to select.
            population: List of the current individuals (states) in the population.
            weights: List of the fitness values for each state.

        Returns: N selected individuals.
        """
        indeces = []
        while len(indeces) < n:
            index = self.fitness_proportionate_selection(weights)
            if index not in indeces:    # ensure we are *not* sampling with replacement
                indeces.append(index)
        return [population[i] for i in indeces]


    def crossover(self, parent1:T, parent2:T) -> T:
        """
        Creates a new state by implementing single point crossover.

        Args:
            parent1: Parent state to crossover. Should not be modified.
            parent2: Parent state to crossover. Should not be modified.

        Returns: New state created by using single-point crossover.
        """
        c = randint(1,len(parent1)-1)
        return parent1[0:c] + parent2[c:]


class GrowthProblem(CGOL_Problem[T]):
    def objective(self, curr_state: np.array[int,...]) -> float:
        # TODO take end state and do mathy maths to get how many cells are alive (very complex calculations)
        pass


class MigrationProblem(CGOL_Problem[T]):
    def objective(self, curr_state: np.array[int,...]) -> float:
        # TODO take end state and do mathy maths to get how far the pop migrated
        pass

