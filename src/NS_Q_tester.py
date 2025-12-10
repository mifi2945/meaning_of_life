from problems import CGOL_Problem, GrowthProblem, Parameters
from algorithms import novelty_search_with_quality, hill_climbing, genetic_algorithm
from main import create_initial_state
from pytorch_parallel import simulate_batch
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

RUNS = 4
PARS = Parameters(
        steps=100,
        include_between=False,
        expansion=-1
    )

def main():
    d_results = []
    hc_results = []
    ga_results = []
    nsq_results = []
    for i in range(RUNS):
        print(f"Run {i+1}/{RUNS}")
        problem = GrowthProblem(state_generator=create_initial_state)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # random
        random_initial = create_initial_state()
        random_tensor = torch.from_numpy(random_initial.reshape(1, -1)).to(device)
        d_log = simulate_batch(random_tensor, PARS, device)[0].cpu().numpy()
        d_results.append(np.sum(d_log))

        # Hillclimbing
        initial = hill_climbing(
            problem=problem,
            parameters=PARS,
            use_cuda=True,
        )
        hc_tensor = torch.from_numpy(initial.reshape(1, -1)).to(device)
        hc_log = simulate_batch(hc_tensor, PARS, device)[0].cpu().numpy()
        hc_results.append(np.sum(hc_log))

        # GA
        initial = genetic_algorithm(
            problem=problem,
            parameters=PARS,
            use_cuda=True,
        )[-1]
        # Ensure initial is a numpy array
        if isinstance(initial, list):
            initial = np.array(initial, dtype=np.uint8)
        ga_tensor = torch.from_numpy(initial.reshape(1, -1)).to(device)
        ga_log = simulate_batch(ga_tensor, PARS, device)[0].cpu().numpy()
        ga_results.append(np.sum(ga_log))

        # NS-Q
        initial = novelty_search_with_quality(
            problem=problem,
            parameters=PARS,
            use_cuda=True,
        )[-1]
        # Ensure initial is a numpy array
        if isinstance(initial, list):
            initial = np.array(initial, dtype=np.uint8)
        nsq_tensor = torch.from_numpy(initial.reshape(1, -1)).to(device)
        nsq_log = simulate_batch(nsq_tensor, PARS, device)[0].cpu().numpy()
        nsq_results.append(np.sum(nsq_log))
    
    # plot and save based on sorted final living cell count...
    os.makedirs("plots", exist_ok=True)

    d_results.sort()
    hc_results.sort()
    ga_results.sort()
    nsq_results.sort()

    plt.figure(figsize=(10,6))
    plt.plot(d_results, label="Direct Random", marker="o")
    plt.plot(hc_results, label="HC", marker="o")
    plt.plot(ga_results, label="GA", marker="o")
    plt.plot(nsq_results, label="NS-Q", marker="o")

    plt.title("Final Live Cells After 100 Steps (20 Runs)")
    plt.xlabel("Run (sorted)")
    plt.ylabel("Final Live Cell Count")
    plt.grid(True)
    plt.legend()

    out_path = "plots/all3.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"\nPlot saved to {out_path}")
    print(f"""Avg Direct: {np.mean(d_results):.2f}, 
          Avg Hillclimbing: {np.mean(hc_results):.2f}, 
          Avg GA: {np.mean(ga_results):.2f}, 
          Avg NS-Q: {np.mean(nsq_results):.2f}""")

if __name__ == "__main__":
    main()