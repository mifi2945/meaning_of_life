import argparse
import numpy as np
from visualizer import GameVisualizer
from algorithms import hill_climbing, genetic_algorithm, novelty_search_with_quality
from problems import Parameters
from problems import CGOL_Problem, GrowthProblem, MigrationProblem
from problems import STEPS, INCLUDE_BETWEEN, EXPANSION
from visualizer import DELAY, GRID_WIDTH


def create_initial_state(type:str = 'empty', size:int = 100) -> np.ndarray:

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
    
    # Create random grid with 31% probability of being alive
    rando = (np.random.rand(size) < 0.5).astype(np.uint8)
    empty = np.zeros(size, dtype=np.uint8)

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
        help="Number of cells to display in visualization (default: 100). Larger values show more detail but smaller cells."
    )
    
    parser.add_argument(
        "-a", "--search-type",
        type=str,
        default="default",
        choices=["default", "hill_climbing", "GA", "NS_Q"],
        help="Search algorithm type: 'default' for direct simulation, 'hill_climbing' for hill climbing (default: default), 'GA' for Genetic Algorithm, " \
        "'NS_Q' for Novelty Search"
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
        "--grid-size",
        type=int,
        default=10,
        help="Initial grid size (NxN). Default: 10 (10x10 = 100 cells)"
    )
    
    parser.add_argument(
        "--no-presets",
        dest="presets",
        action="store_false",
        default=True,
        help="Disable pattern presets (only use single bit flips). By default, presets are enabled."
    )
    
    args = parser.parse_args()
    
    grid_size = args.grid_size * args.grid_size  # Convert to total number of cells
    initial = create_initial_state(args.initial_state, size=grid_size)
    print(f"Initial live cells: {np.sum(initial)}")
    
    print(f"Running {args.search_type}, {args.steps} steps")
    print(f"Expansion: {'infinite' if args.expansion == -1 else args.expansion}")

    parameters = Parameters(
        steps=args.steps,
        include_between=True if args.visualize else False,
        expansion=args.expansion
    )

    quality_threshold = 0
    # Create a wrapper function that uses the correct grid size
    def state_generator_with_size(type: str) -> np.ndarray:
        return create_initial_state(type, size=grid_size)
    
    # Create the appropriate problem instance based on problem type
    if args.problem_type == "growth":
        problem = GrowthProblem(state_generator=state_generator_with_size, type=args.initial_state, enable_presets=args.presets)
        quality_threshold = 200
    elif args.problem_type == "migration":
        problem = MigrationProblem(state_generator=state_generator_with_size, type=args.initial_state, enable_presets=args.presets)
        quality_threshold = 20
    else:
        problem = CGOL_Problem(state_generator=state_generator_with_size, type=args.initial_state, enable_presets=args.presets)

    if args.search_type == "hill_climbing":
        initial = hill_climbing(
            problem=problem,
            parameters=parameters,
        )[-1]
    elif args.search_type == "GA":
        initial = genetic_algorithm(
            problem=problem,
            parameters=parameters,
            quality_threshold=quality_threshold
        )[-1]
    elif args.search_type == "NS_Q":
        initial = novelty_search_with_quality(
            problem=problem,
            parameters=parameters,
            quality_threshold=quality_threshold
        )[-1]

    log = CGOL_Problem.simulate(
        initial=initial,
        parameters=parameters
    )

    if args.visualize:
        viz = GameVisualizer(grid_width=args.grid_width, delay=args.delay)
        
        try:
            viz.display_sequence(log)
        finally:
            viz.quit()
    else:
        print("\nVisualization disabled (--no-display)")

    print(f"Final state: {log[-1].shape[0]}x{log[-1].shape[1]} grid")
    
    # Print problem-specific value
    final_value = problem.value(log[-1], parameters)
    if args.problem_type == "growth":
        print(f"Final value (alive cells): {final_value:.1f}")
    elif args.problem_type == "migration":
        print(f"Final value (distance from center): {final_value:.2f}")
    else:
        print(f"Final value: {final_value:.2f}")


if __name__ == '__main__':
    main()
