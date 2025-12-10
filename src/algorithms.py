from problems import CGOL_Problem, Parameters
import numpy as np
import copy
import torch
from pytorch_parallel import (
    get_device, simulate_batch, value_batch, 
    novelty_batch, behavior_descriptor_batch
)

# from concurrent.futures import ProcessPoolExecutor

def hill_climbing(problem: CGOL_Problem, parameters: Parameters) -> list[np.ndarray]:
    """
    Hill Climbing Search using PyTorch/CUDA for batched simulations.
    Automatically uses GPU if available, falls back to CPU otherwise.

    :param problem: Local Search Problem Object.
    :param parameters: Simulation parameters.
    :param use_cuda: Whether to use CUDA if available (default: True).
    :return: Returns a list of the best states (as evaluated using value()) at each epoch.

    *notes:
        - Each state represents the starting state of that field (ie it needs to be simulated)
    """

    # don't need between steps... for now
    new_params = copy.deepcopy(parameters)
    new_params.include_between = False

    device = get_device() if use_cuda else torch.device("cpu")
    
    # Determine problem type for fitness evaluation
    problem_type = problem.type
    
    length = int(np.sqrt(len(problem.initial_state)))
    best_state = problem.initial_state.reshape((length, length))
    best_states = []
    
    while True:
        best_states.append(best_state.flatten())
        neighbors = [problem.result(best_state.flatten(), n) for n in problem.actions(best_state.flatten())]
        
        # Batch simulate all neighbors
        neighbors_array = np.array(neighbors)
        neighbors_tensor = torch.from_numpy(neighbors_array).to(device)
        final_states = simulate_batch(neighbors_tensor, new_params, device)
        
        # Batch evaluate fitness
        fitness_tensor = value_batch(final_states, problem_type)
        fitness_values = fitness_tensor.cpu().numpy()
        
        # Find best neighbor
        best_neighbor_idx = np.argmax(fitness_values)
        neighbor = neighbors[best_neighbor_idx].reshape((length, length))
        neighbor_fitness = fitness_values[best_neighbor_idx]
        
        # Evaluate current best state for comparison
        best_state_tensor = torch.from_numpy(best_state.flatten().reshape(1, -1)).to(device)
        best_final = simulate_batch(best_state_tensor, new_params, device)
        best_fitness = value_batch(best_final, problem_type).cpu().numpy()[0]

        if neighbor_fitness <= best_fitness:
            best_states.append(best_state.flatten())
            break
        best_state = neighbor

    return best_states

def genetic_algorithm(
    problem: CGOL_Problem, 
    parameters: Parameters, 
    pop_size: int = 100, 
    num_epochs: int = 100,
    use_cuda: bool = True
) -> list[np.ndarray]:
    """
    Genetic Algorithm using PyTorch/CUDA for batched simulations.
    Automatically uses GPU if available, falls back to CPU otherwise.

    :param problem: CGOL_Problem Object.
    :param parameters: Simulation parameters.
    :param pop_size: Size of the population.
    :param num_epochs: Number of epochs.
    :param use_cuda: Whether to use CUDA if available (default: True).
    :return: The best state from the population at each epoch.
    """

    # don't need between steps... for now
    new_params = copy.deepcopy(parameters)
    new_params.include_between = False

    device = get_device() if use_cuda else torch.device("cpu")
    
    # Determine problem type for fitness evaluation
    problem_type = problem.type
    
    mutation_prob = 0.5
    best_states = []

    # Initialize population
    population = np.array([problem.state_generator(problem.type) for _ in range(pop_size)])
    
    epoch = 0
    while epoch < num_epochs:
        epoch += 1
        
        # Batch simulate all individuals
        population_tensor = torch.from_numpy(population).to(device)
        final_states = simulate_batch(population_tensor, new_params, device)
        
        # Batch evaluate fitness
        weights_tensor = value_batch(final_states, problem_type)
        weights = weights_tensor.cpu().numpy().tolist()

        # Find elite
        elite_index = np.argmax(weights)
        elite = population[elite_index].copy()
        best_states.append(elite)

        # Remove elite from population for selection
        population_without_elite = np.delete(population, elite_index, axis=0)
        weights_without_elite = weights.copy()
        weights_without_elite.pop(elite_index)

        # Create new population
        population2 = [elite]
        for _ in range(len(population_without_elite)):
            parents = problem.selection(2, population_without_elite.tolist(), weights_without_elite)
            child = problem.crossover(parents[0], parents[1])
            if np.random.choice(a=[True, False], p=[mutation_prob, 1-mutation_prob]):
                child = problem.mutate(child)
            population2.append(child)
        
        population = np.array(population2)

    return best_states


def novelty_search_with_quality(
    problem: CGOL_Problem,
    parameters: Parameters,
    pop_size: int = 100,
    num_epochs: int = 100,
    k: int = 10,
    novelty_threshold: float = 0.5,
    quality_threshold: float = 0.2,
    archive_max: int = 500,
    novelty_weight: float = 0.5,
    use_cuda: bool = True
):
    """
    NS-Q: novelty + quality using PyTorch/CUDA for batched simulations.
    Automatically uses GPU if available, falls back to CPU otherwise.
    """
    new_params = copy.deepcopy(parameters)
    new_params.include_between = False

    device = get_device() if use_cuda else torch.device("cpu")
    
    # Determine problem type for fitness evaluation
    problem_type = problem.type

    # Initialize population
    population = np.array([problem.state_generator(problem.type) for _ in range(pop_size)])
    archive = []

    # Initial evaluation - batch simulate
    population_tensor = torch.from_numpy(population).to(device)
    final_states = simulate_batch(population_tensor, new_params, device)
    
    # Batch compute fitness
    qualities_tensor = value_batch(final_states, problem_type)
    qualities = qualities_tensor.cpu().numpy().tolist()
    
    # Compute descriptors using batch function
    descriptors = behavior_descriptor_batch(final_states, device)
    
    # Compute novelties
    novelties = novelty_batch(descriptors, archive, k, device).tolist()

    best_states = [population[np.argmax(qualities)]]

    # Initial archive update
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
        parents = population[indices]

        # Generate offspring
        offspring = []
        for _ in range(pop_size):
            i1, i2 = np.random.choice(len(parents), 2, replace=True)
            p1, p2 = parents[i1], parents[i2]
            child = problem.crossover(p1, p2)
            if np.random.rand() < 0.5:
                child = problem.mutate(child)
            offspring.append(child)

        # Batch simulate offspring
        offspring_array = np.array(offspring)
        offspring_tensor = torch.from_numpy(offspring_array).to(device)
        final_offs = simulate_batch(offspring_tensor, new_params, device)
        
        # Batch compute fitness
        offspring_quality_tensor = value_batch(final_offs, problem_type)
        offspring_quality = offspring_quality_tensor.cpu().numpy().tolist()
        
        # Batch compute descriptors
        offspring_desc = behavior_descriptor_batch(final_offs, device)
        
        # Compute novelties
        offspring_novel = novelty_batch(offspring_desc, archive, k, device).tolist()

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

        # Combine populations (convert to lists to handle variable shapes)
        full_pop = population.tolist() + offspring
        full_desc = descriptors + offspring_desc
        full_nov = novelties + offspring_novel
        full_qual = qualities + offspring_quality

        full_comb = np.concatenate([old_comb, new_comb])

        best_idx = np.argsort(full_comb)[-pop_size:]
        population_list = [full_pop[i] for i in best_idx]
        population = np.array(population_list)
        descriptors = [full_desc[i] for i in best_idx]
        novelties = [full_nov[i] for i in best_idx]
        qualities = [full_qual[i] for i in best_idx]

    # return 
    return best_states