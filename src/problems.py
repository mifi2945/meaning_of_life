from __future__ import annotations  # needed in order to reference a Class within itself
from random import randint, random, choice, choices
from selectors import SelectorKey
from typing import Callable, List
from abc import ABC, abstractmethod
import numpy as np
import copy
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist



STEPS = 100
INCLUDE_BETWEEN = True
EXPANSION = -1


class Parameters:
    steps: int = STEPS
    include_between: bool = INCLUDE_BETWEEN
    expansion: int = EXPANSION
    def __init__(self, steps: int = STEPS, include_between: bool = INCLUDE_BETWEEN, expansion: int = EXPANSION):
        self.steps = steps
        self.include_between = include_between
        self.expansion = expansion

class Problem(ABC):
    """
    Interface representing a generic problem formulation.

    This class defines the operations needed for search algorithms.
    """



    @abstractmethod
    def value(self, curr_state: np.ndarray) -> float:
        """
        Performance measure of the current state, used for local search.

        Args:
            curr_state: Current state as a numpy array.

        Returns:
            float: Fitness or score of the current state.
        """
        pass


class CGOL_Problem(Problem):
    """
    Version of Problem used for Local Search algorithms Hill Climbing, Genetic Algorithm, and Novelty Search.

    Implements for Conway's Game of Life, with a list of values representing the 20x20 space.
    """
    # Pattern definitions
    BLINKER = np.array([
        0, 0, 0,
        1, 1, 1,
        0, 0, 0
    ], dtype=np.uint8)
    
    GLIDER = np.array([
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 1, 1, 1, 0,
        0, 0, 0, 0, 0
    ], dtype=np.uint8)
    
    TOAD = np.array([
        0, 0, 0, 0,
        0, 1, 1, 1,
        1, 1, 1, 0,
        0, 0, 0, 0
    ], dtype=np.uint8)

    SUSTAIN = np.array([
        1, 1, 0,
        1, 0, 1,
        0, 1, 1
    ], dtype=np.uint8)
    
    BEEHIVE = np.array([
        0, 1, 0,
        1, 0, 1,
        1, 0, 1,
        0, 1, 0
    ], dtype=np.uint8)

    
    def __init__(self, state_generator: Callable[[], np.ndarray], type: str = "random", enable_presets: bool = True):
        self.initial_state = state_generator(type)
        self.state_generator = state_generator
        self.type = type
        # Store patterns with their dimensions (only if presets are enabled)
        if enable_presets:
            self.patterns = {
                'blinker': (self.BLINKER, 3, 3),      # 3x3
                'glider': (self.GLIDER, 5, 5),        # 5x5
                'toad': (self.TOAD, 4, 4),            # 4x4
                'sustain_1': (self.SUSTAIN, 3, 3),  # 3x3
                'sustain_2': (self.BEEHIVE, 4, 3)   # 4x3 (rectangular)
            }
        else:
            self.patterns = {}


    @staticmethod
    def simulate(
        initial: np.ndarray,
        parameters: Parameters
    ) -> list[np.ndarray]:
        steps = parameters.steps
        include_between = parameters.include_between
        expansion = parameters.expansion

        log = []
        
        # Dimension of the intial grid
        s = int(len(initial) ** 0.5)

        # How far we can expand the grid
        size = (s + (steps//2) + 1 if expansion == -1 else s + expansion) * 2
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
        
        # Track bounding box
        min_r, max_r = bounds[0], bounds[1]
        min_c, max_c = bounds[0], bounds[1]
        
        for _ in range(steps):
            # Add padding of 1 and account for exclusive end bounds
            scan_min_r = max(0, min_r - 1)
            scan_max_r = min(M.shape[0], max_r + 1)
            scan_min_c = max(0, min_c - 1)
            scan_max_c = min(M.shape[1], max_c + 1)
            
            # Only look for living cells in the region of interest
            region_to_scan = M[scan_min_r:scan_max_r, scan_min_c:scan_max_c]
            living = np.where(region_to_scan == 1)
            
            if len(living[0]) == 0:
                # TODO need to move this print somewhere else cuz otherwise it prints too much
                # print("Everyone died and your model is bad :(")
                break
            
            living_r = living[0] + scan_min_r
            living_c = living[1] + scan_min_c
            
            min_r = max(0, living_r.min() - 1)
            max_r = min(M.shape[0], living_r.max() + 2)
            min_c = max(0, living_c.min() - 1)
            max_c = min(M.shape[1], living_c.max() + 2)
            
            region = M[min_r:max_r, min_c:max_c]
            
            # pad with zeroes but return same dimensions
            neighbors = convolve2d(region, kernel, mode='same', boundary='fill', fillvalue=0)
            
            # Rules, John
            # Live cells with 2-3 neighbors survive
            survives = (region == 1) & ((neighbors == 2) | (neighbors == 3))
            # Dead cells with exactly 3 neighbors become alive
            born = (region == 0) & (neighbors == 3)
            new_region = (survives | born).astype(initial.dtype)
            
            # Update M with the new region
            M[min_r:max_r, min_c:max_c] = new_region
            
            if include_between:
                log.append(M.copy())
        
        if not include_between:
            log.append(M.copy())
        
        return log


    def actions(self, curr_state: np.ndarray) -> list:
        """
        Actions that can be applied to a CGoL board.
        Returns both single bit flips and pattern placements.
        
        Action format:
        - For bit flips: int (cell index)
        - For patterns: tuple ('pattern_name', row, col) where row/col is top-left position

        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.

        Returns: List of actions (bit flips + pattern placements).

        """
        actions = []
        # Single bit flips (most granular)
        actions.extend([i for i in range(len(curr_state))])
        
        # Pattern placements
        length = int(np.sqrt(len(curr_state)))
        for pattern_name, (pattern, pattern_h, pattern_w) in self.patterns.items():
            max_row = length - pattern_h + 1
            max_col = length - pattern_w + 1
            for row in range(max_row):
                for col in range(max_col):
                    actions.append((pattern_name, row, col))
        
        return actions

    def result(self, curr_state: np.ndarray, action) -> np.ndarray:
        """
        Applies an action to the state.
        
        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.
            action: Either:
                   - int: index to flip (single bit flip)
                   - tuple: ('pattern_name', row, col) to place a pattern

        Returns: the next state representation after the given action is performed.

        """
        next_state = curr_state.copy()
        length = int(np.sqrt(len(curr_state)))
        next_state_2d = next_state.reshape((length, length))
        
        if isinstance(action, int):
            # Single bit flip (most granular)
            next_state[action] ^= 1
        elif isinstance(action, tuple):
            # Pattern placement
            pattern_name, row, col = action
            pattern, pattern_h, pattern_w = self.patterns[pattern_name]
            pattern_2d = pattern.reshape((pattern_h, pattern_w))
            
            # Place pattern using XOR (so overlapping patterns toggle)
            next_state_2d[row:row+pattern_h, col:col+pattern_w] ^= pattern_2d
        
        return next_state_2d.flatten()
    
    def behavior_descriptor(self, final_state: np.ndarray, parameters: Parameters) -> np.ndarray:
        """
        Features:
        1. Population Count (normalized)
        2. Bounding Box Width (normalized)
        3. Bounding Box Height (normalized)
        4. Density (Population / Bounding Box Area)
        """
        rows, cols = np.where(final_state == 1)
        
        if len(rows) == 0:
            return np.array([0.0, 0.0, 0.0, 0.0])

        # Calculate Bounding Box
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()
        
        height = rmax - rmin + 1
        width = cmax - cmin + 1
        population = len(rows)
        area = height * width
        density = population / area if area > 0 else 0

        # Normalize large values to keep distances balanced
        # Assuming typical grid sizes, dividing helps keep Euclidean distance meaningful
        return np.array([
            population / 100.0,
            width / 20.0,
            height / 20.0,
            density
        ])


    def novelty(self, descriptor: np.ndarray,
            archive: List[np.ndarray],
            population_descriptors: List[np.ndarray],
            k: int = 10) -> float:
        """
        Calculates the novelty using KNN Euclidean metric
        
        Args:
            descriptor: The behavior vector of the individual being evaluated.
            archive: List of behavior vectors from the archive.
            population_descriptors: List of behavior vectors from the current population.
            k: Number of nearest neighbors to consider.
            
        Returns:
            The average Euclidean distance to the k-nearest neighbors.
        """
        pool = np.array(archive + population_descriptors)
        
        if len(pool) == 0:
            return 0.0
        
        dists = cdist(descriptor.reshape(1, -1), pool, metric='euclidean')[0]
        dists.sort()
        
        # remove exact duplicate (distance 0) if it is the individual itself
        if dists[0] == 0.0 and len(dists) > 1:
            dists = dists[1:]
            
        k_eff = min(k, len(dists))
        if k_eff == 0:
            return 0.0
            
        return float(np.mean(dists[:k_eff]))


    @abstractmethod
    def value(self, curr_state: np.ndarray, parameters: Parameters) -> float:
        pass


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
        total_fit = sum(fitness_values)
        if total_fit == 0:
            return randint(0, len(fitness_values) - 1)
        
        probs = [val / total_fit for val in fitness_values]
        indeces = [i for i in range(len(fitness_values))]
        return choices(indeces, weights=probs, k=1)[0]


    def selection(self, n:int, population: list[np.ndarray],
                  weights: list[float]) -> list[np.ndarray]:
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


    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Creates a new state by implementing single point crossover.

        Args:
            parent1: Parent state to crossover. Should not be modified.
            parent2: Parent state to crossover. Should not be modified.

        Returns: New state created by using single-point crossover.
        """
        c = randint(1, len(parent1) - 1)
        # Concatenate arrays instead of tuple slicing
        return np.concatenate([parent1[:c], parent2[c:]])
    
    def mutate(self, child: np.ndarray, pattern_prob: float = 0.3) -> np.ndarray:
        """
        Mutate by either:
        - Flipping a random bit (most granular, default 70% chance)
        - Placing a random pattern (30% chance by default)
        
        Args:
            child: State to mutate
            pattern_prob: Probability of using pattern mutation instead of bit flip (default 0.3)
        
        Returns: Mutated state
        """
        mutation = child.copy()
        length = int(np.sqrt(len(child)))
        
        if np.random.rand() < pattern_prob and len(self.patterns) > 0:
            # Pattern mutation: place a random pattern at a random position
            pattern_name = np.random.choice(list(self.patterns.keys()))
            pattern, pattern_h, pattern_w = self.patterns[pattern_name]
            
            # Random position (top-left corner of pattern)
            max_row = max(0, length - pattern_h + 1)
            max_col = max(0, length - pattern_w + 1)
            if max_row > 0 and max_col > 0:
                row = randint(0, max_row - 1)
                col = randint(0, max_col - 1)
                
                # Place pattern using XOR
                mutation_2d = mutation.reshape((length, length))
                pattern_2d = pattern.reshape((pattern_h, pattern_w))
                mutation_2d[row:row+pattern_h, col:col+pattern_w] ^= pattern_2d
                mutation = mutation_2d.flatten()
        else:
            # Bit flip mutation (most granular)
            index = randint(0, len(child) - 1)
            mutation[index] ^= 1
        
        return mutation


class GrowthProblem(CGOL_Problem):
    def value(self, state: np.ndarray, parameters: Parameters) -> float:
        """
        Sum of all alive cells on the grid
        """
        alive = np.sum(state)
        return float(alive)


class MigrationProblem(CGOL_Problem):
    def value(self, state: np.ndarray, parameters: Parameters) -> float:
        """
        Quality Metric: Distance of Center of Mass from Grid Center.
        Higher (further) is better.
        """
        if len(state.shape) < 2:
            length = int(np.sqrt(len(state)))
            state = state.reshape((length,length))

        rows, cols = np.where(state == 1)
        if len(rows) == 0:
            return 0.0
            
        # object center
        avg_r = np.mean(rows)
        avg_c = np.mean(cols)
        
        # grid center
        center_r, center_c = state.shape[0] / 2, state.shape[1] / 2
        
        # distance from center
        dist = np.sqrt((avg_r - center_r)**2 + (avg_c - center_c)**2)
        return float(dist)