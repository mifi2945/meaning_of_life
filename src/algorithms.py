from problems import CGOL_Problem, Parameters
import numpy as np
import copy

# from concurrent.futures import ProcessPoolExecutor

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
        # weights = eval_batch(problem, new_params, population)

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


def novelty_search_with_quality(problem:CGOL_Problem,
                                parameters:Parameters,
                                pop_size:int=100,
                                num_epochs:int=100,
                                k:int=10,
                                novelty_threshold:float=0.5,
                                quality_threshold:float=0.2,
                                archive_max:int=500,
                                novelty_weight:int=0.5):
    """
    NS-Q: novelty + quality
    """

    new_params = copy.deepcopy(parameters)
    new_params.include_between = False

    population = [problem.state_generator() for _ in range(pop_size)]
    archive = []

    # initial eval
    descriptors = [problem.behavior_descriptor(ind, new_params)
                   for ind in population]
    novelties = [
        problem.novelty(desc, archive, descriptors, k)
        for desc in descriptors
    ]
    qualities = [problem.value(ind, new_params) for ind in population]

    # initial archive update
    for desc, n, q in zip(descriptors, novelties, qualities):
        if n >= novelty_threshold and q >= quality_threshold:
            archive.append(desc)

    for epoch in range(num_epochs):

        # selection
        combined = novelty_weight * np.array(novelties) + \
                   (1 - novelty_weight) * np.array(qualities)
        probs = combined + 1e-6
        probs /= probs.sum()

        indices = np.random.choice(len(population), size=pop_size, p=probs)
        parents = [population[i] for i in indices]

        # offspring
        offspring = []
        for _ in range(pop_size):
            # p1, p2 = np.random.choice(parents, 2, replace=True)
            i1, i2 = np.random.choice(len(parents), 2, replace=True)
            p1, p2 = population[i1], population[i2]
            child = problem.crossover(p1, p2)
            if np.random.rand() < 0.5:
                child = problem.mutate(child)
            offspring.append(child)

        # eval offspring
        offspring_desc = [problem.behavior_descriptor(ind, new_params)
                            for ind in offspring]

        offspring_novel = [
        problem.novelty(desc, archive, offspring_desc, k)
        for desc in offspring_desc
        ]
        

        offspring_quality = [
            problem.value(ind, new_params) for ind in offspring
        ]

        # ----------------------------
        # Archive update
        # ----------------------------
        for desc, n, q in zip(offspring_desc, offspring_novel, offspring_quality):
            if n >= novelty_threshold and q >= quality_threshold:
                archive.append(desc)

        if len(archive) > archive_max:
            # remove least novel
            idx = np.argsort(offspring_novel)[::-1]
            archive = [archive[i] for i in idx[:archive_max]]

        # ----------------------------
        # Replace population
        # Replace worst NS-Q individuals with offspring
        # ----------------------------
        old_comb = combined
        new_comb = novelty_weight * np.array(offspring_novel) + \
                   (1 - novelty_weight) * np.array(offspring_quality)

        # choose best pop_size individuals from old+new
        full_pop = population + offspring
        full_desc = descriptors + offspring_desc
        full_nov = novelties + offspring_novel
        full_qual = qualities + offspring_quality

        full_comb = np.concatenate([old_comb, new_comb])

        best_idx = np.argsort(full_comb)[-pop_size:]
        population = [full_pop[i] for i in best_idx]
        descriptors = [full_desc[i] for i in best_idx]
        novelties = [full_nov[i] for i in best_idx]
        qualities = [full_qual[i] for i in best_idx]

    return population



# def eval_batch(problem: CGOL_Problem, params: Parameters, population: list[np.ndarray]):
#     with ProcessPoolExecutor() as ex:
#         return list(ex.map(lambda s: problem.value(s, params), population))