import argparse
import numpy as np
from problems import CGOL_Problem
from visualizer import GameVisualizer
from algorithms import hill_climbing


def create_initial_state() -> np.ndarray:
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
    
    return glider

def main():
    parser = argparse.ArgumentParser(
        description="Conway's Game of Life Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Simulation parameters
    parser.add_argument(
        "-s", "--steps",
        type=int,
        default = 100,
        help="Number of simulation steps (default: 100)"
    )
    
    parser.add_argument(
        "-e", "--expansion",
        type=int,
        default=-1,
        help="Grid expansion size. -1 for infinite expansion (default: -1)"
    )
    
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        default=True,
        help="Visualize the simulation (default: True)"
    )
    
    # Visualization parameters
    parser.add_argument(
        "-d", "--delay",
        type=int,
        default=100,
        help="Delay between frames in milliseconds (default: 100)"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=50,
        help="Display grid size (default: 50)"
    )
    
    parser.add_argument(
        "--search-type",
        type=str,
        default="default",
        choices=["default", "hill_climbing"],
        help="Search algorithm type: 'default' for direct simulation, 'hill_climbing' for hill climbing (default: default)"
    )
    
    args = parser.parse_args()
    
    initial = create_initial_state()
    print(f"Initial live cells: {np.sum(initial)}")
    
    print(f"Running {args.search_type}, {args.steps} steps")
    print(f"Expansion: {'infinite' if args.expansion == -1 else args.expansion}")
    
    if args.search_type == "hill_climbing":
        log = hill_climbing(
            initial=initial,
            steps=args.steps,
            expansion=args.expansion
        )
    else:
        log = CGOL_Problem.simulate(
            initial=initial,
            steps=args.steps,
            include_between=True if args.visualize else False,
            expansion=args.expansion
        )

    if args.visualize:
        display_size = min(args.grid_size, log[0].shape[0])
        viz = GameVisualizer(grid_size=display_size)
        
        try:
            viz.display_sequence(log, delay_ms=args.delay, auto_advance=True)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            viz.quit()
    else:
        print("\nVisualization disabled (--no-display)")
        print(f"Final state: {log[-1].shape[0]}x{log[-1].shape[1]} grid")
        print(f"Final live cells: {np.sum(log[-1])}")


if __name__ == '__main__':
    main()
