from problems import CGOL_Problem, Parameters
import numpy as np

def hill_climbing(problem: CGOL_Problem, parameters: Parameters) -> list[np.ndarray]:
    """
    Hill Climbing Search.

    :param problem: Local Search Problem Object.
    :return: Returns a list of the best states (as evaluated using value()) at each epoch.

    *notes:
        - Each state represents the starting state of that field (ie it needs to be simulated)
    """
    curr = problem.initial_state
    best_states = []
    while True:
        best_states.append(curr)    # store step
        neighbors = [problem.result(curr, n) for n in problem.actions(curr)]
        neighbor = neighbors[np.argmax([problem.value(n) for n in neighbors])]      # get max valued neighbor

        if problem.value(neighbor) <= problem.value(curr):
            best_states.append(curr)
            break
        curr = neighbor

    return best_states

def genetic_algorithm(problem: CGOL_Problem, pop_size:int = 100, num_epochs:int = 1000) -> list[np.ndarray]:
    """
    Implements a Genetic Algorithm.

    :param problem: LSProblem Object.
    :param pop_size: Size of the population.
    :param num_epochs: Number of epochs.
    :return: The best state from the population at each epoch.
    """
    mutation_prob = 0.5
    best_states = []

    population = [problem.state_generator() for _ in range(pop_size)]
    solved = False
    epoch = 0
    while not solved and epoch < num_epochs:
        epoch += 1
        weights = [problem.value(state) for state in population]

        elite_index = np.argmax(weights)
        weights.pop(elite_index)
        elite = population.pop(elite_index)
        best_states.append(elite)
        if problem.is_goal(elite):
            solved = True

        population2 = [elite]
        for _ in range(len(population)):
            parents = problem.selection(2, population, weights)
            child = problem.crossover(parents[0], parents[1])
            if np.random.choice(a=[True, False], p=[mutation_prob, 1-mutation_prob]):
                child = problem.mutate(child)
            population2.append(child)
        population = population2

    return best_states