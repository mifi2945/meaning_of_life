#!/usr/bin/env python3
"""
Visualize a previously saved simulation state.
"""

import argparse
import numpy as np
from visualizer import GameVisualizer, DELAY, GRID_WIDTH
import os


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a saved Conway's Game of Life simulation state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "-f", "--file",
        type=str,
        default="saved_state.npz",
        help="Path to saved state file (default: saved_state.npz)"
    )
    
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
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        print(f"Looking for saved state file. Make sure to run the main script with --save flag first.")
        return 1
    
    # Load saved state
    try:
        data = np.load(args.file, allow_pickle=True)
        states = data['states']
        
        # Handle both single state and list of states
        if isinstance(states, np.ndarray) and states.ndim == 3:
            # If it's a 3D array, convert to list
            states = [states[i] for i in range(states.shape[0])]
        elif isinstance(states, np.ndarray) and states.ndim == 2:
            # Single state
            states = [states]
        elif isinstance(states, np.ndarray) and states.ndim == 1:
            # List of states stored as object array
            states = states.tolist()
        
        # Print info if available
        if 'initial_state' in data:
            initial = data['initial_state']
            print(f"Initial state: {initial.shape if hasattr(initial, 'shape') else 'N/A'}")
            if hasattr(initial, 'shape'):
                print(f"Initial live cells: {np.sum(initial)}")
        
        if 'steps' in data:
            print(f"Simulation steps: {data['steps']}")
        
        if 'search_type' in data:
            print(f"Search algorithm: {data['search_type']}")
        
        if 'problem_type' in data:
            print(f"Problem type: {data['problem_type']}")
        
        print(f"Loaded {len(states)} state(s)")
        print(f"State shape: {states[0].shape}")
        print(f"Final live cells: {np.sum(states[-1])}")
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1
    
    # Visualize
    viz = GameVisualizer(grid_width=args.grid_width, delay=args.delay)
    
    try:
        viz.display_sequence(states)
    finally:
        viz.quit()
    
    return 0


if __name__ == '__main__':
    exit(main())

