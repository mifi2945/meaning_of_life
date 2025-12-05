from __future__ import annotations  # needed in order to reference a Class within itself
from random import randint, random, choice, choices
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
    def __init__(self, state_generator: Callable[[], np.ndarray]):
        self.initial_state = state_generator()
        self.state_generator = state_generator


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


    def actions(self, curr_state: np.ndarray) -> list[int]:
        """
        Actions that can be applied to a CGoL board.
        We can technically set multiple cells at a time, but we will limit to one per epoch.

        Args:
            curr_state: List values that correspond to the values of the alive/dead cells.

        Returns: List of cells that can be set.

        """
        return [i for i in range(len(curr_state))]

    def result(self, curr_state: np.ndarray, action:int) -> np.ndarray:
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
    
    def behavior_descriptor(self, final, parameters):
        # final = CGOL_Problem.simulate(state, parameters)[0]

        # 1. Extract bounding box of live cells
        rows, cols = np.where(final == 1)
        if len(rows) == 0:
            return np.zeros((1,), dtype=np.uint8)  # empty pattern

        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()

        shape = final[rmin:rmax+1, cmin:cmax+1]  # cropped pattern

        # 2. Generate all 8 isometries (rotations + reflections)
        transforms = [
            shape,
            np.rot90(shape, 1),
            np.rot90(shape, 2),
            np.rot90(shape, 3),
            np.fliplr(shape),
            np.flipud(shape),
            np.rot90(np.fliplr(shape), 1),
            np.rot90(np.flipud(shape), 1),
        ]

        # 3. Pick the canonical representation (lexicographically smallest)
        canonical = min(t.flatten().tobytes() for t in transforms)

        # return as numeric vector
        return np.frombuffer(canonical, dtype=np.uint8)


    def _to_index_set(self, arr: np.ndarray) -> set:  
        a = np.asarray(arr)
        if a.size == 0:
            return set()
        if a.ndim != 1:
            a = a.ravel()
        # nonzero returns tuple; take first element
        nz = np.nonzero(a)[0]
        return set(map(int, nz))


    def _jaccard_set(self, a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return 1.0 - (inter / union)


    def novelty(self, descriptor: np.ndarray,
            archive: List[np.ndarray],
            population_descriptors: List[np.ndarray],
            k: int = 10) -> float:
        # Convert primary descriptor -> index set
        desc_set = self._to_index_set(descriptor)

        # Build pool of sets (population followed by archive)
        pool_arrays = list(population_descriptors) + list(archive)
        pool_sets = [self._to_index_set(p) for p in pool_arrays]

        # Compute distances but exclude exactly one self-match (if present).
        dists = []
        self_skipped = False
        for other in pool_sets:
            if (other == desc_set) and (not self_skipped):
                # skip the first exact-equal match (treating that as self)
                self_skipped = True
                continue
            d = self._jaccard_set(desc_set, other)
            dists.append(d)

        if len(dists) == 0:
            # No neighbors to compare to
            return 0.0

        # Use up to k nearest neighbors
        k_eff = min(k, len(dists))
        dists.sort()
        return float(sum(dists[:k_eff]) / k_eff)


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
        probs = [val / sum(fitness_values) for val in fitness_values]
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
    
    def mutate(self, child: np.ndarray) -> np.ndarray:
        """
        Mutate randomly by flipping any random bit
        """
        index = randint(0,len(child)-1)
        mutation = child.copy()
        mutation[index] ^= 1    # XOR flips the bit
        return mutation


class GrowthProblem(CGOL_Problem):
    def value(self, state: np.ndarray, parameters: Parameters) -> float:
        """
        Ratio of total alive over total cells on board
        We want to reduce spread of cells as well (one larger clump preferred)
        This is evaluated AFTER the board is simulated
        """

        # get final simulated state
        # state = CGOL_Problem.simulate(curr_state, parameters)[0]

        alive = np.sum(state)
        # total = state.size

        # if alive == 0:
        #     return 0.0

        # n = int(np.sqrt(total))
        # grid = state.reshape((n, n))

        # layer = np.array([
        #     [1, 1, 1],
        #     [1, 0, 1],
        #     [1, 1, 1]
        # ])
        # # live neighbors
        # neighbor_counts = convolve2d(grid, layer, mode='same', boundary='fill', fillvalue=0)

        # # clump_score = avg neighbors normalized to [0, 1]
        # clump_score = np.sum(neighbor_counts * grid) / (alive * 8)

        # density = alive / total
        # weighted density by clump
        #* (0.5 + 0.5 * clump_score)
        return alive 


class MigrationProblem(CGOL_Problem):
    def value(self, curr_state: np.ndarray) -> float:
        return 1