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
                                k:int=5,
                                novelty_threshold:float=1,
                                quality_threshold:float=10,
                                archive_max:int=300,
                                novelty_weight:float=0.3):
    """
    NS-Q: Novelty Search with Quality (elitism implemented).

    This algorithm uses a weighted combination of Novelty and Quality (Fitness)
    for selection and population replacement.
    """

    new_params = copy.deepcopy(parameters)
    new_params.include_between = False

    population = [problem.state_generator(problem.type) for _ in range(pop_size)]
    archive = []
    best_states = []

    # --- Initial Evaluation ---
    final_states = [CGOL_Problem.simulate(ind, new_params)[-1] for ind in population]

    descriptors = [problem.behavior_descriptor(ind, new_params)
                   for ind in final_states]
    novelties = [
        problem.novelty(desc, archive, descriptors, k)
        for desc in descriptors
    ]
    qualities = [problem.value(ind, new_params) for ind in final_states]

    # Calculate combined scores for initial population
    combined_scores = novelty_weight * np.array(novelties) + \
                      (1 - novelty_weight) * np.array(qualities)
    
    # Identify the initial elite
    elite_idx = np.argmax(combined_scores)
    elite = population[elite_idx]
    best_states.append(elite)


    # --- Initial Archive Update ---
    for desc, n, q in zip(descriptors, novelties, qualities):
        if n >= novelty_threshold and q >= quality_threshold:
            archive.append(desc)

    # --- Main Evolutionary Loop ---
    for epoch in range(num_epochs):
        # ----------------------------
        # Selection and Elitism
        # ----------------------------
        
        # Calculate selection probabilities based on combined score
        probs = combined_scores + 1e-6 # Add epsilon to avoid division by zero
        probs /= probs.sum()

        # Select parents (not including the elite in the selection pool)
        indices = np.random.choice(len(population), size=pop_size - 1, p=probs, replace=True)
        parents_for_offspring = [population[i] for i in indices]


        # ----------------------------
        # Offspring Generation
        # ----------------------------
        offspring = []
        for _ in range(pop_size - 1): # Generate pop_size-1 offspring
            # Select parents from the selected pool (can be the same individual)
            i1, i2 = np.random.choice(len(parents_for_offspring), 2, replace=True)
            p1, p2 = parents_for_offspring[i1], parents_for_offspring[i2]
            
            child = problem.crossover(p1, p2)
            if np.random.rand() < 0.5: # Mutation probability check
                child = problem.mutate(child)
            offspring.append(child)

        # ----------------------------
        # Offspring Evaluation
        # ----------------------------
        final_offs = [CGOL_Problem.simulate(ind, new_params)[-1] for ind in offspring]
        
        offspring_desc = [problem.behavior_descriptor(ind, new_params)
                            for ind in final_offs]
        
        # Calculate offspring novelty (Archive is updated AFTER this calculation)
        offspring_novel = [
            problem.novelty(desc, archive, offspring_desc, k)
            for desc in offspring_desc
        ]
        
        offspring_quality = [
            problem.value(ind, new_params) for ind in final_offs
        ]

        # ----------------------------
        # Archive Update
        # ----------------------------
        for desc, n, q in zip(offspring_desc, offspring_novel, offspring_quality):
            # Only add to archive if both novelty and quality pass their thresholds
            if n >= novelty_threshold and q >= quality_threshold:
                archive.append(desc)

        # Archive size maintenance
        if len(archive) > archive_max:
            num_to_remove = len(archive) - archive_max
            
            internal_novelties = [problem.novelty(d, archive, archive, k) for d in archive]
            remove_indices = np.argsort(internal_novelties)[:num_to_remove]
            archive = [archive[i] for i in range(len(archive)) if i not in remove_indices]
            
            # archive = archive[num_to_remove:] # Remove the oldest items

        # ----------------------------
        # New Population Formation (Selection + Elitism)
        # ----------------------------
        
        # 1. Combine Old Population + Offspring
        full_pop = population + offspring
        full_desc = descriptors + offspring_desc
        full_nov = novelties + offspring_novel
        full_qual = qualities + offspring_quality

        # 2. Calculate Combined Score for the Full Pool
        old_comb = combined_scores
        new_comb = novelty_weight * np.array(offspring_novel) + \
                   (1 - novelty_weight) * np.array(offspring_quality)
        full_comb = np.concatenate([old_comb, new_comb])
        
        # 3. Select the Elite (highest combined score in the entire pool)
        elite_idx = np.argmax(full_comb)
        elite = full_pop[elite_idx]
        best_states.append(elite)
        
        # 4. Select the rest of the new population (pop_size - 1) based on combined score
        # Find indices of the top pop_size individuals (excluding the elite index if necessary)
        # We sort all scores and take the top `pop_size` individuals.
        best_idx = np.argsort(full_comb)[-pop_size:]
        
        # New population is formed by the top `pop_size` individuals based on combined score
        population = [full_pop[i] for i in best_idx]
        descriptors = [full_desc[i] for i in best_idx]
        novelties = [full_nov[i] for i in best_idx]
        qualities = [full_qual[i] for i in best_idx]
        combined_scores = np.array([full_comb[i] for i in best_idx])        
        
    return best_states