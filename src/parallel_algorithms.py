"""
PyTorch/CUDA-accelerated parallel algorithms for evolutionary search.
This module provides batch-processed versions of genetic algorithm and novelty search.
"""

import torch
import numpy as np
from typing import Callable, Optional, Tuple, List
import copy

from problems import CGOL_Problem, Parameters
from parallel_sim import (
    get_device, numpy_to_torch, torch_to_numpy,
    batch_simulate_optimized, batch_value_growth
)


def batch_genetic_algorithm(
    problem: CGOL_Problem,
    parameters: Parameters,
    pop_size: int = 100,
    num_epochs: int = 100,
    mutation_prob: float = 0.5,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None
) -> List[np.ndarray]:
    """
    GPU-accelerated Genetic Algorithm using batch processing.
    
    Args:
        problem: CGOL_Problem instance
        parameters: Simulation parameters
        pop_size: Population size
        num_epochs: Number of generations
        mutation_prob: Probability of mutation
        device: Device to use (defaults to best available)
        batch_size: Batch size for processing (defaults to pop_size)
        
    Returns:
        List of best states from each epoch
    """
    if device is None:
        device = get_device()
    
    if batch_size is None:
        batch_size = pop_size
    
    # Don't need intermediate states for evaluation
    new_params = copy.deepcopy(parameters)
    new_params.include_between = False
    
    best_states = []
    
    # Initialize population
    population_np = np.array([problem.state_generator() for _ in range(pop_size)])
    state_size = population_np.shape[1]
    
    # Convert to torch
    population = numpy_to_torch(population_np, device)
    
    print(f"Using device: {device}")
    print(f"Population size: {pop_size}, Batch size: {batch_size}")
    
    for epoch in range(num_epochs):
        # Batch evaluate fitness
        fitness_scores = batch_evaluate_fitness(
            population, problem, new_params, device, batch_size
        )
        
        # Get elite
        elite_idx = fitness_scores.argmax().item()
        elite = population[elite_idx:elite_idx+1].clone()
        best_states.append(torch_to_numpy(elite.squeeze(0)))
        
        # Selection and reproduction
        # Convert fitness to probabilities
        fitness_np = fitness_scores.cpu().numpy()
        fitness_np = np.maximum(fitness_np, 1e-8)  # Avoid division by zero
        probs = fitness_np / fitness_np.sum()
        
        # Create new population
        new_population = [elite]
        
        # Generate children in batches
        for _ in range(pop_size - 1):
            # Select parents
            parent_indices = np.random.choice(pop_size, size=2, p=probs, replace=False)
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]
            
            # Crossover
            c = np.random.randint(1, state_size)
            child = torch.cat([parent1[:c], parent2[c:]])
            
            # Mutation
            if np.random.random() < mutation_prob:
                mut_idx = np.random.randint(0, state_size)
                child[mut_idx] = 1.0 - child[mut_idx]  # Flip bit
            
            new_population.append(child)
        
        population = torch.stack(new_population)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Best fitness: {fitness_scores.max().item():.4f}")
    
    return best_states


def batch_evaluate_fitness(
    population: torch.Tensor,
    problem: CGOL_Problem,
    parameters: Parameters,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None
) -> torch.Tensor:
    """
    Batch evaluate fitness for a population of states.
    
    Args:
        population: Tensor of shape (pop_size, state_size)
        problem: Problem instance
        parameters: Simulation parameters
        device: Device to use
        batch_size: Batch size for processing
        
    Returns:
        Fitness scores tensor of shape (pop_size,)
    """
    if device is None:
        device = get_device()
    
    if batch_size is None:
        batch_size = population.shape[0]
    
    pop_size = population.shape[0]
    all_scores = []
    
    # Process in batches
    for i in range(0, pop_size, batch_size):
        batch = population[i:i+batch_size]
        
        # Simulate batch
        final_states = batch_simulate_optimized(
            batch,
            steps=parameters.steps,
            expansion=parameters.expansion,
            include_between=False,
            device=device
        )
        
        # Evaluate fitness
        # Check if it's a GrowthProblem
        from problems import GrowthProblem
        if isinstance(problem, GrowthProblem):
            scores = batch_value_growth(final_states, device)
        else:
            # Fallback to individual evaluation for other problem types
            scores = torch.tensor([
                problem.value(torch_to_numpy(batch[j]), parameters)
                for j in range(batch.shape[0])
            ], device=device)
        
        all_scores.append(scores)
    
    return torch.cat(all_scores)


def batch_novelty_search(
    problem: CGOL_Problem,
    parameters: Parameters,
    pop_size: int = 100,
    num_epochs: int = 100,
    mutation_prob: float = 0.5,
    novelty_threshold: float = 0.1,
    archive_size: int = 1000,
    k: int = 15,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None
) -> List[np.ndarray]:
    """
    GPU-accelerated Novelty Search algorithm.
    
    Novelty Search selects individuals based on how different they are from
    previously seen behaviors, rather than just fitness.
    
    Args:
        problem: CGOL_Problem instance
        parameters: Simulation parameters
        pop_size: Population size
        num_epochs: Number of generations
        mutation_prob: Probability of mutation
        novelty_threshold: Minimum novelty to add to archive
        archive_size: Maximum archive size
        k: Number of nearest neighbors for novelty calculation
        device: Device to use
        batch_size: Batch size for processing
        
    Returns:
        List of most novel states from each epoch
    """
    if device is None:
        device = get_device()
    
    if batch_size is None:
        batch_size = pop_size
    
    new_params = copy.deepcopy(parameters)
    new_params.include_between = False
    
    novel_states = []
    archive = []  # Archive of behaviors for novelty calculation
    
    # Initialize population
    population_np = np.array([problem.state_generator() for _ in range(pop_size)])
    state_size = population_np.shape[1]
    population = numpy_to_torch(population_np, device)
    
    print(f"Using device: {device}")
    print(f"Novelty Search - Population: {pop_size}, Archive size: {archive_size}, k={k}")
    
    for epoch in range(num_epochs):
        # Get behaviors (final states) for entire population
        final_states = batch_simulate_optimized(
            population,
            steps=parameters.steps,
            expansion=parameters.expansion,
            include_between=False,
            device=device
        )
        
        # Flatten behaviors for distance calculation
        behaviors = final_states.view(pop_size, -1).cpu().numpy()
        
        # Calculate novelty scores
        novelty_scores = calculate_novelty_batch(
            behaviors, archive, k, device
        )
        
        # Add novel individuals to archive
        for i, score in enumerate(novelty_scores):
            if score > novelty_threshold and len(archive) < archive_size:
                archive.append(behaviors[i])
        
        # Get most novel individual
        most_novel_idx = novelty_scores.argmax().item()
        most_novel = population[most_novel_idx:most_novel_idx+1].clone()
        novel_states.append(torch_to_numpy(most_novel.squeeze(0)))
        
        # Selection based on novelty
        novelty_np = novelty_scores.cpu().numpy()
        novelty_np = np.maximum(novelty_np, 1e-8)
        probs = novelty_np / novelty_np.sum()
        
        # Create new population
        new_population = [most_novel]
        
        for _ in range(pop_size - 1):
            # Select parents based on novelty
            parent_indices = np.random.choice(pop_size, size=2, p=probs, replace=False)
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]
            
            # Crossover
            c = np.random.randint(1, state_size)
            child = torch.cat([parent1[:c], parent2[c:]])
            
            # Mutation
            if np.random.random() < mutation_prob:
                mut_idx = np.random.randint(0, state_size)
                child[mut_idx] = 1.0 - child[mut_idx]
            
            new_population.append(child)
        
        population = torch.stack(new_population)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Max novelty: {novelty_scores.max().item():.4f}, "
                  f"Archive size: {len(archive)}")
    
    return novel_states


def calculate_novelty_batch(
    behaviors: np.ndarray,
    archive: List[np.ndarray],
    k: int = 15,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Calculate novelty scores for a batch of behaviors.
    
    Novelty is the average distance to the k nearest neighbors in the archive.
    
    Args:
        behaviors: Array of shape (batch_size, behavior_dim)
        archive: List of archived behaviors
        k: Number of nearest neighbors
        device: Device to use
        
    Returns:
        Novelty scores tensor of shape (batch_size,)
    """
    if device is None:
        device = get_device()
    
    batch_size = behaviors.shape[0]
    novelty_scores = torch.zeros(batch_size, device=device)
    
    if len(archive) == 0:
        # If archive is empty, all behaviors are novel
        return torch.ones(batch_size, device=device) * 100.0
    
    # Convert to torch
    behaviors_torch = torch.from_numpy(behaviors).float().to(device)
    archive_torch = torch.from_numpy(np.array(archive)).float().to(device)
    
    # Calculate distances to all archive members
    # behaviors_torch: (batch_size, dim)
    # archive_torch: (archive_size, dim)
    # distances: (batch_size, archive_size)
    distances = torch.cdist(behaviors_torch, archive_torch, p=2)
    
    # Get k nearest neighbors for each behavior
    k_actual = min(k, len(archive))
    k_nearest_distances, _ = torch.topk(distances, k_actual, dim=1, largest=False)
    
    # Novelty is average distance to k nearest neighbors
    novelty_scores = k_nearest_distances.mean(dim=1)
    
    return novelty_scores

