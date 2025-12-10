from problems import CGOL_Problem, Parameters
import numpy as np
import copy

# from concurrent.futures import ProcessPoolExecutor

def hill_climbing(problem: CGOL_Problem, parameters: Parameters) -> list[np.ndarray]:
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

    length = int(np.sqrt(len(problem.initial_state)))
    best_state = problem.initial_state.reshape((length, length))
    best_states = []
    while True:
        best_states.append(best_state.flatten())
        neighbors = [problem.result(best_state.flatten(), n) for n in problem.actions(best_state.flatten())]
        final_states = [CGOL_Problem.simulate(ind, new_params)[-1] for ind in neighbors]
        # get max valued neighbor
        neighbor = neighbors[np.argmax([problem.value(n, new_params) for n in final_states])].reshape((length,length))

        if problem.value(neighbor, new_params) <= problem.value(best_state, new_params):
            best_states.append(best_state.flatten())
            break
        best_state = neighbor

    return best_states

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

    population = [problem.state_generator(problem.type) for _ in range(pop_size)]
    # solved = False
    epoch = 0
    while epoch < num_epochs:
        epoch += 1
        final_states = [CGOL_Problem.simulate(ind, new_params)[-1] for ind in population]
        weights = [problem.value(state, new_params) for state in final_states]
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

    population = [problem.state_generator(problem.type) for _ in range(pop_size)]
    archive = []

    final_states = [CGOL_Problem.simulate(ind, new_params)[-1] for ind in population]

    # initial eval
    descriptors = [problem.behavior_descriptor(ind, new_params)
                   for ind in final_states]
    novelties = [
        problem.novelty(desc, archive, descriptors, k)
        for desc in descriptors
    ]
    qualities = [problem.value(ind, new_params) for ind in final_states]

    best_states = [population[np.argmax(qualities)]]

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

        final_offs = [CGOL_Problem.simulate(ind, new_params)[-1] for ind in offspring]
        # eval offspring
        offspring_desc = [problem.behavior_descriptor(ind, new_params)
                            for ind in final_offs]

        offspring_novel = [
            problem.novelty(desc, archive, offspring_desc, k)
            for desc in offspring_desc
        ]
        

        offspring_quality = [
            problem.value(ind, new_params) for ind in final_offs
        ]

        best_states.append(offspring[np.argmax(offspring_quality)])

        # ----------------------------
        # Archive update
        # ----------------------------
        for desc, n, q in zip(offspring_desc, offspring_novel, offspring_quality):
            if n >= novelty_threshold and q >= quality_threshold:
                archive.append(desc)

        # TODO update based on BOTH novelty and quality
        if len(archive) > archive_max:
            # remove least novel
            combined_ofs = novelty_weight * np.array(offspring_novel) + \
               (1 - novelty_weight) * np.array(offspring_quality)

            idx = np.argsort(combined_ofs)[::-1]  # highest combined score
            archive = [archive[i] for i in idx[:archive_max]]
            # idx = np.argsort(offspring_novel)[::-1]
            # archive = [archive[i] for i in idx[:archive_max]]

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

    # return 
    return best_states