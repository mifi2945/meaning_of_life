"""
Example script demonstrating how to use the GPU-accelerated parallel algorithms.
This shows how to run novelty search and parallel genetic algorithm.
"""

import numpy as np
from problems import Parameters, GrowthProblem
from algorithms import genetic_algorithm_parallel, novelty_search
from parallel_sim import get_device


def create_initial_state() -> np.ndarray:
    """Create a random initial state for the simulation."""
    return np.random.randint(0, 2, size=100, dtype=np.uint8)


def main():
    """Run parallel algorithms example."""
    import torch
    # Check device availability
    device = get_device()
    print(f"Using device: {device}")
    print(f"CUDA available: {device.type == 'cuda'}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set up parameters
    parameters = Parameters(
        steps=100,
        include_between=False,  # Don't need intermediate states for search
        expansion=-1  # Infinite expansion
    )
    
    # Create problem
    problem = GrowthProblem(state_generator=create_initial_state)
    
    print("\n" + "="*60)
    print("Running GPU-accelerated Genetic Algorithm...")
    print("="*60)
    
    # Run parallel genetic algorithm
    ga_results = genetic_algorithm_parallel(
        problem=problem,
        parameters=parameters,
        pop_size=100,
        num_epochs=50,
        mutation_prob=0.5,
        batch_size=100  # Process entire population at once
    )
    
    print(f"\nGA completed. Best state from final epoch:")
    print(f"  Live cells: {np.sum(ga_results[-1])}")
    
    print("\n" + "="*60)
    print("Running GPU-accelerated Novelty Search...")
    print("="*60)
    
    # Run novelty search
    novelty_results = novelty_search(
        problem=problem,
        parameters=parameters,
        pop_size=100,
        num_epochs=50,
        mutation_prob=0.5,
        novelty_threshold=0.1,
        archive_size=1000,
        k=15,
        batch_size=100
    )
    
    print(f"\nNovelty Search completed. Most novel state from final epoch:")
    print(f"  Live cells: {np.sum(novelty_results[-1])}")
    
    print("\n" + "="*60)
    print("Performance Tips:")
    print("="*60)
    print("1. Use larger batch_size (up to pop_size) for better GPU utilization")
    print("2. Increase pop_size for better exploration (if GPU memory allows)")
    print("3. Novelty search typically finds more diverse solutions")
    print("4. For very large populations, you may need to reduce batch_size")
    print("   to fit in GPU memory")


if __name__ == '__main__':
    main()

