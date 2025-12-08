import numpy as np
import matplotlib.pyplot as plt
from algorithms import hill_climbing
from problems import Parameters, GrowthProblem, CGOL_Problem
from problems import STEPS, EXPANSION


def create_initial_state() -> np.ndarray:
    """Create a random initial state for the simulation."""
    return np.random.randint(0, 2, size=100, dtype=np.uint8)


def run_simulation(search_type: str, num_runs: int = 50) -> list[int]:
    """
    Run the simulation multiple times and collect final living cells.
    
    Args:
        search_type: Either "default" or "hill_climbing"
        num_runs: Number of times to run the simulation
        
    Returns:
        List of final living cells from each run
    """
    final_living_cells = []
    
    parameters = Parameters(
        steps=STEPS,
        include_between=False,  # Don't need intermediate states
        expansion=EXPANSION
    )
    
    for i in range(num_runs):
        print(f"Running {search_type} iteration {i+1}/{num_runs}...")
        
        # Create problem instance
        problem = GrowthProblem(state_generator=create_initial_state)
        initial = problem.initial_state
        
        # Apply search algorithm if specified
        if search_type == "hill_climbing":
            initial = hill_climbing(
                problem=problem,
                parameters=parameters,
            )
        # For "default", just use the initial state as-is
        
        # Simulate the game
        log = CGOL_Problem.simulate(
            initial=initial,
            parameters=parameters
        )
        
        # Get final living cells
        final_state = log[-1]
        living_cells = int(np.sum(final_state))
        final_living_cells.append(living_cells)
        
        print(f"  Final living cells: {living_cells}")
    
    return final_living_cells


def main():
    """Run both algorithms and create comparison graph."""
    print("Running hill_climbing 20 times...")
    hill_climbing_results = run_simulation("hill_climbing", num_runs=20)
    
    print("\nRunning default 20 times...")
    default_results = run_simulation("default", num_runs=20)
    
    # Sort both results in ascending order
    hill_climbing_sorted = sorted(hill_climbing_results)
    default_sorted = sorted(default_results)
    
    # Create the plot
    plt.figure(figsize=(4, 3))
    plt.plot(hill_climbing_sorted, label='Hill Climbing', marker='o', linestyle='-')
    plt.plot(default_sorted, label='Default', marker='s', linestyle='-')
    
    plt.xlabel('Run Index (sorted)')
    plt.ylabel('Final Living Cells')
    plt.title('Comparison of Final Living Cells: Hill Climbing vs Default')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("\nGraph saved as 'algorithm_comparison.png'")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Hill Climbing - Mean: {np.mean(hill_climbing_results):.2f}, "
          f"Median: {np.median(hill_climbing_results):.2f}, "
          f"Min: {min(hill_climbing_results)}, Max: {max(hill_climbing_results)}")
    print(f"Default - Mean: {np.mean(default_results):.2f}, "
          f"Median: {np.median(default_results):.2f}, "
          f"Min: {min(default_results)}, Max: {max(default_results)}")


if __name__ == '__main__':
    main()

