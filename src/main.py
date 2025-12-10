import argparse
import numpy as np
import torch
import time
from visualizer import GameVisualizer
from algorithms import (
    hill_climbing, genetic_algorithm, novelty_search_with_quality
)
from pytorch_parallel import get_device
from problems import Parameters
from problems import CGOL_Problem, GrowthProblem, MigrationProblem
from problems import STEPS, INCLUDE_BETWEEN, EXPANSION
from visualizer import DELAY, GRID_WIDTH


def create_initial_state(type:str = 'empty') -> np.ndarray:

    death = np.array([
        0, 0, 0,
        0, 1, 1,
        0, 0, 0,
    ], dtype=np.uint8)

    glider = np.array([
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 1, 1, 1, 0,
        0, 0, 0, 0, 0
    ], dtype=np.uint8)
    
    blinker = np.array([
        0, 0, 0,
        1, 1, 1,
        0, 0, 0
    ], dtype=np.uint8)
    
    toad = np.array([
        0, 0, 0, 0,
        0, 1, 1, 1,
        1, 1, 1, 0,
        0, 0, 0, 0
    ], dtype=np.uint8)
    
    rando = np.random.randint(0, 2, size=100, dtype=np.uint8)
    empty = np.zeros(100, dtype=np.uint8)

    if type == 'empty':
        return empty
    if type == 'random':
        return rando

def main():
    parser = argparse.ArgumentParser(
        description="Conway's Game of Life Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Simulation parameters
    parser.add_argument(
        "-s", "--steps",
        type=int,
        default = STEPS,
        help="Number of simulation steps (default: 100)"
    )
    
    parser.add_argument(
        "-e", "--expansion",
        type=int,
        default=EXPANSION,
        help="Grid expansion size. -1 for infinite expansion (default: -1)"
    )
    
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        # default=INCLUDE_BETWEEN,
        help="Visualize the simulation (default: True)"
    )
    
    # Visualization parameters
    parser.add_argument(
        "-d", "--delay",
        type=int,
        default=DELAY,
        help="Delay between frames in milliseconds (default: 100)"
    )
    
    parser.add_argument(
        "--grid-width",
        type=int,
        default=GRID_WIDTH,
        help="Display grid width (default: 100)"
    )
    
    parser.add_argument(
        "-a", "--search-type",
        type=str,
        default="default",
        choices=["default", "hill_climbing", "GA", "NS_Q"],
        help="Search algorithm type: 'default' for direct simulation, 'hill_climbing' for hill climbing (default: default), 'GA' for Genetic Algorithm, " \
        "'NS_Q' for Novelty Search (all use CUDA/GPU acceleration when available)"
    )
    
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available (use CPU instead)"
    )
    
    parser.add_argument(
        "-p", "--problem-type",
        type=str,
        default="growth",
        choices=["growth", "migration"],
        help="Problem type to solve: 'growth' for growth problem, 'migration' for migration problem (default: growth)"
    )

    parser.add_argument(
        "-i", "--initial-state",
        type=str,
        default="random",
        choices=["random", "empty"],
        help="Initial state generation: 'random' for random start, 'empty' for an empty initial state"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save simulation state to file (saved_state.npz)"
    )

    args = parser.parse_args()

    # Print all args
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    initial = create_initial_state(args.initial_state)
    print(f"Initial live cells: {np.sum(initial)}")

    print(f"Running {args.search_type}, {args.steps} steps")
    print(f"Expansion: {'infinite' if args.expansion == -1 else args.expansion}")

    parameters = Parameters(
        steps=args.steps,
        include_between=True if args.visualize else False,
        expansion=args.expansion
    )

    # Create the appropriate problem instance based on problem type
    if args.problem_type == "growth":
        problem = GrowthProblem(state_generator=create_initial_state, type=args.initial_state)
    elif args.problem_type == "migration":
        problem = MigrationProblem(state_generator=create_initial_state)
    else:
        problem = CGOL_Problem(state_generator=create_initial_state)

    use_cuda = not args.no_cuda
    
    if use_cuda and torch.cuda.is_available():
        device = get_device()
        print(f"Using device: {device}")
    else:
        device = None
        print("Using CPU")
    
    # Start timing here
    start_time = time.perf_counter()

    if args.search_type == "hill_climbing":
        initial = hill_climbing(
            problem=problem,
            parameters=parameters,
        )[-1]
    elif args.search_type == "GA":
        initial = genetic_algorithm(
            problem=problem,
            parameters=parameters,
            use_cuda=use_cuda,
        )[-1]
    elif args.search_type == "NS_Q":
        initial = novelty_search_with_quality(
            problem=problem,
            parameters=parameters,
            use_cuda=use_cuda,
        )[-1]

    # For visualization, we need intermediate states, so use the non-batch simulation
    # Ensure initial is a numpy array (search algorithms may return lists)
    if isinstance(initial, list):
        initial = np.array(initial, dtype=np.uint8)
    
    # Use CGOL_Problem.simulate to get the full sequence for visualization
    log = CGOL_Problem.simulate(initial, parameters)
    
    # End timing here
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Simulation time: {elapsed:.3f} seconds")

    # Save state if requested
    if args.save:
        import os
        filename = "saved_state.npz"
        # Convert log to numpy array for saving
        states_array = np.array(log) if len(log) > 1 else np.array([log[0]])
        np.savez_compressed(
            filename,
            states=states_array,
            initial_state=initial,
            steps=parameters.steps,
            expansion=parameters.expansion,
            search_type=args.search_type,
            problem_type=args.problem_type
        )
        print(f"State saved to {filename}")

    if args.visualize:
        print("Visualizing...")
        viz = GameVisualizer(grid_width=args.grid_width, delay=args.delay)
        
        try:
            viz.display_sequence(log)
        finally:
            viz.quit()

    print(f"Final state: {log[-1].shape[0]}x{log[-1].shape[1]} grid")
    print(f"Final live cells: {np.sum(log[-1])}")


if __name__ == '__main__':
    main()
