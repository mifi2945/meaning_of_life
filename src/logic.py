from typing import List
import copy
import numpy as np

STEPS = 10_000
INCLUDE_BETWEEN = True
EXPANSION = -1


def step(M, bounds):
    for i in range(bounds[0], bounds[1])

    



def simulate(
    initial: np.array,
    steps: int=STEPS,
    include_between: bool=INCLUDE_BETWEEN,
    size: int=EXPANSION
):
    log = np.array()

    s = len(initial) ** 0.5
    size = steps if size == -1 else size
    M = copy.deepcopy(initial)
    M = np.zeroes((size, size))

    middle = size/2
    half = s/2

    bounds = (middle-half, middle+half+(s%2))


    M[bounds[0]:bounds[1], bounds[0]:bounds[1]] = initial.reshape((s, s))
    if include_between:
        log.append(M)
    
    for i in range(steps):
        M, bounds = step(M, bounds)
        if include_between:
            log.append(M)










