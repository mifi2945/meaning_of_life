import numpy as np
from typing import List
from problems import CGOL_Problem


def hill_climbing(initial: np.ndarray, steps: int = 100, expansion: int = -1) -> List[np.ndarray]:
    """
    Hill climbing algorithm for Conway's Game of Life.
    
    Args:
        initial: Initial state as a 1D numpy array
        steps: Number of steps to run
        expansion: Grid expansion size (-1 for infinite)
    
    Returns:
        List of grid states (log)
    """
    # TODO: Implement hill climbing algorithm
    # For now, just return the result of simulate
    return CGOL_Problem.simulate(
        initial=initial,
        steps=steps,
        include_between=True,
        expansion=expansion
    )

