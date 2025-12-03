import argparse
import numpy as np
from visualizer import GameVisualizer
from algorithms import hill_climbing, genetic_algorithm, novelty_search_with_quality
from problems import Parameters
from problems import CGOL_Problem, GrowthProblem, MigrationProblem
from problems import STEPS, INCLUDE_BETWEEN, EXPANSION
from visualizer import DELAY, GRID_WIDTH


def create_initial_state() -> np.ndarray:

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
        "'NS_Q' for Novelty Search"
    )
    
    parser.add_argument(
        "-p", "--problem-type",
        type=str,
        default="growth",
        choices=["growth", "migration"],
        help="Problem type to solve: 'growth' for growth problem, 'migration' for migration problem (default: growth)"
    )
    
    args = parser.parse_args()
    
    initial = create_initial_state()
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
        problem = GrowthProblem(state_generator=create_initial_state)
    elif args.problem_type == "migration":
        problem = MigrationProblem(state_generator=create_initial_state)
    else:
        problem = CGOL_Problem(state_generator=create_initial_state)

    if args.search_type == "hill_climbing":
        initial = hill_climbing(
            problem=problem,
            parameters=parameters,
        )
    elif args.search_type == "GA":
        initial = genetic_algorithm(
            problem=problem,
            parameters=parameters,
        )[-1]
    elif args.search_type == "NS_Q":
        initial = novelty_search_with_quality(
            problem=problem,
            parameters=parameters,
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
    print(f"Final live cells: {np.sum(log[-1])}")


if __name__ == '__main__':
    main()
