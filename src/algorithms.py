from problems import CGOL_Problem, Parameters
import numpy as np
import copy

from numba import njit, prange

def hill_climbing(problem: CGOL_Problem, parameters: Parameters) -> np.ndarray:
    """
    Hill Climbing Search.

    :param problem: Local Search Problem Object.
    :return: Returns a list of the best states (as evaluated using value()) at each epoch.

    *notes:
        - Each state represents the starting state of that field (ie it needs to be simulated)
    """

    # don't need between steps... for now
    new_params = copy.deepcopy(parameters)
    new_params.include_between = False

    best_state = problem.initial_state
    while True:
        neighbors = [problem.result(best_state, n) for n in problem.actions(best_state)]
        neighbor = neighbors[np.argmax([problem.value(n, new_params) for n in neighbors])]      # get max valued neighbor

        if problem.value(neighbor, new_params) <= problem.value(best_state, new_params):
            break
        best_state = neighbor

    return best_state

def genetic_algorithm(problem: CGOL_Problem, parameters: Parameters, pop_size:int = 100, num_epochs:int = 100) -> list[np.ndarray]:
    """
    Implements a Genetic Algorithm.

    :param problem: CGOL_Problem Object.
    :param pop_size: Size of the population.
    :param num_epochs: Number of epochs.
    :return: The best state from the population at each epoch.
    """

    # don't need between steps... for now
    new_params = copy.deepcopy(parameters)
    new_params.include_between = False

    mutation_prob = 0.5
    best_states = []

    population = [problem.state_generator() for _ in range(pop_size)]
    # solved = False
    epoch = 0
    while epoch < num_epochs:
        epoch += 1
        weights = [problem.value(state, new_params) for state in population]
        # weights = np.array([state for state in population])
        # parallel_ga_helper(problem, new_params, weights)

        elite_index = np.argmax(weights)
        weights.pop(elite_index)
        elite = population.pop(elite_index)
        best_states.append(elite)

        population2 = [elite]
        for _ in range(len(population)):
            parents = problem.selection(2, population, weights)
            child = problem.crossover(parents[0], parents[1])
            if np.random.choice(a=[True, False], p=[mutation_prob, 1-mutation_prob]):
                child = problem.mutate(child)
            population2.append(child)
        population = population2

    return best_states

@njit(parallel=True)
def parallel_ga_helper(problem: CGOL_Problem, params: Parameters, weights: np.ndarray):
    """
    TODO: ignore for now
    """
    for i in prange(len(weights)):
            weights[i] = problem.value(weights[i], params)