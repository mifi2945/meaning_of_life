from __future__ import annotations  # needed in order to reference a Class within itself
from random import randint, random, choice, choices
from typing import Generic, TypeVar, Callable
from abc import ABC, abstractmethod
import numpy as np
import copy
from scipy.signal import convolve2d




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
        log = []
        
        # Dimension of the intial grid
        s = int(len(initial) ** 0.5)

        # How far we can expand the grid
        size = s + (steps//2) + 1 if expansion == -1 else s + expansion
        M = np.zeros((size, size), dtype=initial.dtype)

        # (middle - half, middle + half) is where the initial grid goes in the new one
        middle = size // 2
        half = s // 2

        # Plop initial state in center of new grid
        bounds = (middle - half, middle + half + (s%2))
        M[bounds[0]:bounds[1], bounds[0]:bounds[1]] = initial.reshape((s, s))

        if include_between:
            log.append(M.copy())
        
        # Kernel for counting neighbors (8 neighbors around each cell)
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        
        for _ in range(steps):
            # Find bounding box of living cells for speed
            living = np.where(M == 1)
            if len(living[0]) == 0:
                # Everyone died and your model is bad :(
                break
            
            # Calculate bounds with 1 cell padding (2 is because [) indexing)
            min_r = max(0, living[0].min() - 1)
            max_r = min(M.shape[0], living[0].max() + 2)
            min_c = max(0, living[1].min() - 1)
            max_c = min(M.shape[1], living[1].max() + 2)            
            
            region = M[min_r:max_r, min_c:max_c]
            
            # pad with zeroes but return same dimensions
            neighbors = convolve2d(region, kernel, mode='same', boundary='fill', fillvalue=0)
            
            # Rules, John
            new_region = ((region == 1) & ((neighbors == 2) | (neighbors == 3))) | ((region == 0) & (neighbors == 3))
            new_region = new_region.astype(initial.dtype)
            
            # Update M with the new region
            M[min_r:max_r, min_c:max_c] = new_region
            
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

