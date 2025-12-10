#!/usr/bin/env python3
"""
Performance testing script for NS-Q algorithm.
Compares GPU vs CPU performance with timing information.
"""

import time
import numpy as np
import argparse
from problems import CGOL_Problem, GrowthProblem, Parameters
from algorithms import novelty_search_with_quality
from main import create_initial_state
import torch

def test_nsq_performance(
    runs: int = 5,
    steps: int = 100,
    pop_size: int = 100,
    num_epochs: int = 50,
    use_cuda: bool = True,
    compare_sequential: bool = True
):
    """
    Test NS-Q performance and compare GPU vs CPU.
    
    Args:
        runs: Number of runs to perform
        steps: Number of simulation steps
        pop_size: Population size
        num_epochs: Number of epochs
        use_cuda: Whether to use CUDA for GPU runs
        compare_sequential: Whether to also run CPU version for comparison
    """
    
    parameters = Parameters(
        steps=steps,
        include_between=False,
        expansion=-1
    )
    
    print("=" * 60)
    print("NS-Q Performance Test (GPU vs CPU)")
    print("=" * 60)
    print(f"Runs: {runs}")
    print(f"Steps: {steps}")
    print(f"Population Size: {pop_size}")
    print(f"Epochs: {num_epochs}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Use CUDA: {use_cuda}")
    print("=" * 60)
    
    parallel_times = []
    parallel_results = []
    sequential_times = []
    sequential_results = []
    
    for run in range(runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{runs}")
        print(f"{'='*60}")
        
        problem = GrowthProblem(state_generator=create_initial_state)
        
        # Test GPU version
        print("\n--- Testing NS-Q on GPU ---")
        start_time = time.perf_counter()
        
        try:
            initial = novelty_search_with_quality(
                problem=problem,
                parameters=parameters,
                pop_size=pop_size,
                num_epochs=num_epochs,
                use_cuda=use_cuda
            )[-1]
            
            # Ensure initial is a numpy array (function returns list)
            if isinstance(initial, list):
                initial = np.array(initial, dtype=np.uint8)
            elif not isinstance(initial, np.ndarray):
                initial = np.asarray(initial, dtype=np.uint8)
            
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            parallel_times.append(elapsed)
            
            # Evaluate result using batch simulation
            from pytorch_parallel import simulate_batch
            import torch
            device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
            initial_tensor = torch.from_numpy(initial.reshape(1, -1) if initial.ndim == 1 else initial.reshape(1, -1)).to(device)
            final_state_tensor = simulate_batch(initial_tensor, parameters, device)
            final_state = final_state_tensor[0].cpu().numpy()
            result = np.sum(final_state)
            parallel_results.append(result)
            
            print(f"GPU time: {elapsed:.2f} seconds")
            print(f"GPU result (final live cells): {result}")
            
        except Exception as e:
            print(f"Error in parallel version: {e}")
            import traceback
            traceback.print_exc()
        
        # Test CPU version for comparison
        if compare_sequential:
            print("\n--- Testing NS-Q on CPU ---")
            start_time = time.perf_counter()
            
            try:
                initial = novelty_search_with_quality(
                    problem=problem,
                    parameters=parameters,
                    pop_size=pop_size,
                    num_epochs=num_epochs,
                    use_cuda=False  # Force CPU
                )[-1]
                
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                sequential_times.append(elapsed)
                
                # Evaluate result using batch simulation
                from pytorch_parallel import simulate_batch
                import torch
                device = torch.device("cpu")  # Force CPU for comparison
                initial_tensor = torch.from_numpy(initial.reshape(1, -1) if initial.ndim == 1 else initial.reshape(1, -1)).to(device)
                final_state_tensor = simulate_batch(initial_tensor, parameters, device)
                final_state = final_state_tensor[0].cpu().numpy()
                result = np.sum(final_state)
                sequential_results.append(result)
                
                print(f"CPU time: {elapsed:.2f} seconds")
                print(f"CPU result (final live cells): {result}")
                
                if parallel_times and sequential_times:
                    speedup = sequential_times[-1] / parallel_times[-1]
                    print(f"GPU Speedup: {speedup:.2f}x")
                    
            except Exception as e:
                print(f"Error in sequential version: {e}")
                import traceback
                traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if parallel_times:
        print(f"\nNS-Q (GPU):")
        print(f"  Average time: {np.mean(parallel_times):.2f} ± {np.std(parallel_times):.2f} seconds")
        print(f"  Min time: {np.min(parallel_times):.2f} seconds")
        print(f"  Max time: {np.max(parallel_times):.2f} seconds")
        print(f"  Average result: {np.mean(parallel_results):.2f} ± {np.std(parallel_results):.2f}")
        print(f"  Best result: {np.max(parallel_results)}")
    
    if sequential_times:
        print(f"\nNS-Q (CPU):")
        print(f"  Average time: {np.mean(sequential_times):.2f} ± {np.std(sequential_times):.2f} seconds")
        print(f"  Min time: {np.min(sequential_times):.2f} seconds")
        print(f"  Max time: {np.max(sequential_times):.2f} seconds")
        print(f"  Average result: {np.mean(sequential_results):.2f} ± {np.std(sequential_results):.2f}")
        print(f"  Best result: {np.max(sequential_results)}")
    
    if parallel_times and sequential_times:
        avg_speedup = np.mean([s/p for s, p in zip(sequential_times, parallel_times)])
        print(f"\nAverage GPU Speedup: {avg_speedup:.2f}x")
        
        if avg_speedup > 1:
            print(f"✓ GPU version is {avg_speedup:.2f}x faster on average")
        else:
            print(f"✗ GPU version is {1/avg_speedup:.2f}x slower on average")
    
    print("=" * 60)
    
    return {
        'parallel_times': parallel_times,
        'parallel_results': parallel_results,
        'sequential_times': sequential_times,
        'sequential_results': sequential_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NS-Q parallelization performance")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs (default: 5)")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps (default: 100)")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size (default: 100)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--no-compare", action="store_true", help="Skip CPU comparison")
    
    args = parser.parse_args()
    
    test_nsq_performance(
        runs=args.runs,
        steps=args.steps,
        pop_size=args.pop_size,
        num_epochs=args.epochs,
        use_cuda=not args.no_cuda,
        compare_sequential=not args.no_compare
    )

