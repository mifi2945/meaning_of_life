from problems import CGOL_Problem, GrowthProblem, MigrationProblem, Parameters
from algorithms import novelty_search_with_quality, hill_climbing, genetic_algorithm
from main import create_initial_state
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

RUNS = 10
PARS = Parameters(
        steps=100,
        include_between=False,
        expansion=-1
    )

def growth_over_runs():
    d_results = []
    hc_results = []
    ga_results = []
    nsq_results = []
    for i in range(RUNS):
        print(f"Run {i+1}/{RUNS}")
        problem = GrowthProblem(state_generator=create_initial_state)

        # random
        d_log = CGOL_Problem.simulate(initial=create_initial_state(), parameters=PARS)[-1]
        d_results.append(np.sum(d_log))

        # Hillclimbing
        initial = hill_climbing(
            problem=problem,
            parameters=PARS,
        )[-1]
        hc_log = CGOL_Problem.simulate(initial=initial, parameters=PARS)[-1]
        hc_results.append(np.sum(hc_log))

        # GA
        initial = genetic_algorithm(
            problem=problem,
            parameters=PARS,
        )[-1]
        ga_log = CGOL_Problem.simulate(initial=initial, parameters=PARS)[-1]
        ga_results.append(np.sum(ga_log))

        # NS-Q
        initial = novelty_search_with_quality(
            problem=problem,
            parameters=PARS,
            quality_threshold=200
        )[-1]
        nsq_log = CGOL_Problem.simulate(initial=initial, parameters=PARS)[-1]
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

    plt.title("Final Live Cells After 100 Steps")
    plt.xlabel("Run (sorted)")
    plt.ylabel("Final Live Cell Count")
    plt.grid(True)
    plt.legend()

    out_path = "plots/all.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"\nPlot saved to {out_path}")
    print(f"""Avg Direct: {np.mean(d_results):.2f}, 
          Avg Hillclimbing: {np.mean(hc_results):.2f}, 
          Avg GA: {np.mean(ga_results):.2f}, 
          Avg NS-Q: {np.mean(nsq_results):.2f}""")

 
def migration():
    problem = MigrationProblem(state_generator=create_initial_state, type="random")
    hc_states = hill_climbing(
            problem=problem,
            parameters=PARS,
        )
    hc_results = [problem.value(CGOL_Problem.simulate(s, PARS)[-1], PARS) for s in hc_states]

    ga_states = genetic_algorithm(
            problem=problem,
            parameters=PARS,
        )
    ga_results = [problem.value(CGOL_Problem.simulate(s, PARS)[-1], PARS) for s in ga_states]

    nsq_states = novelty_search_with_quality(
            problem=problem,
            parameters=PARS,
            quality_threshold=20
        )
    nsq_results = [problem.value(CGOL_Problem.simulate(s, PARS)[-1], PARS) for s in nsq_states]
    
    # # plot and save based on sorted final living cell count...
    os.makedirs("migration_plots", exist_ok=True)

    plt.figure(figsize=(10,6))
    plt.plot(hc_results, label="HC", marker="o")
    plt.plot(ga_results, label="GA", marker="o")
    plt.plot(nsq_results, label="NS-Q", marker="o")

    plt.title(f"Migration Quality over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Quality")
    plt.grid(True)
    plt.legend()

    out_path = f"migration_plots/all.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"\nPlot saved to {out_path}")

def growth_over_epochs():
    problem = GrowthProblem(state_generator=create_initial_state, type="random")
    hc_states = hill_climbing(
            problem=problem,
            parameters=PARS,
        )
    hc_results = [problem.value(CGOL_Problem.simulate(s, PARS)[-1], PARS) for s in hc_states]

    ga_states = genetic_algorithm(
            problem=problem,
            parameters=PARS,
        )
    ga_results = [problem.value(CGOL_Problem.simulate(s, PARS)[-1], PARS) for s in ga_states]

    nsq_states = novelty_search_with_quality(
            problem=problem,
            parameters=PARS,
            quality_threshold=200
        )
    nsq_results = [problem.value(CGOL_Problem.simulate(s, PARS)[-1], PARS) for s in nsq_states]
    
    # # plot and save based on sorted final living cell count...
    os.makedirs("growth_plots", exist_ok=True)

    plt.figure(figsize=(10,6))
    plt.plot(hc_results, label="HC", marker="o")
    plt.plot(ga_results, label="GA", marker="o")
    plt.plot(nsq_results, label="NS-Q", marker="o")

    plt.title(f"Growth Quality over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Quality")
    plt.grid(True)
    plt.legend()

    out_path = f"growth_plots/all.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    #growth_over_runs()
    growth_over_epochs()
    migration()