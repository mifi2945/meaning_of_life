"""
Visualize the archive from Novelty Search with Quality (NS-Q).

This script runs NS-Q and creates visualizations showing:
1. Archive size over epochs
2. 2D scatter plots of archive members in behavior space
3. 3D visualization of archive members
"""

from problems import CGOL_Problem, GrowthProblem, MigrationProblem, Parameters
from algorithms import novelty_search_with_quality
from main import create_initial_state
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import os


def novelty_search_with_quality_track_archive(problem: CGOL_Problem,
                                              parameters: Parameters,
                                              pop_size: int = 100,
                                              num_epochs: int = 100,
                                              k: int = 5,
                                              novelty_threshold: float = 1,
                                              quality_threshold: float = 10,
                                              archive_max: int = 300,
                                              novelty_weight: float = 0.3):
    """
    NS-Q with archive tracking. Returns best states and archive history.
    
    Returns:
        best_states: List of best states at each epoch
        archive_history: List of archive states at each epoch (list of descriptors)
    """
    import copy
    
    new_params = copy.deepcopy(parameters)
    new_params.include_between = False

    population = [problem.state_generator(problem.type) for _ in range(pop_size)]
    archive = []
    best_states = []
    archive_history = []  # Track archive at each epoch

    # --- Initial Evaluation ---
    final_states = [CGOL_Problem.simulate(ind, new_params)[-1] for ind in population]

    descriptors = [problem.behavior_descriptor(ind, new_params)
                   for ind in final_states]
    novelties = [
        problem.novelty(desc, archive, descriptors, k)
        for desc in descriptors
    ]
    qualities = [problem.value(ind, new_params) for ind in final_states]

    # Calculate combined scores for initial population
    combined_scores = novelty_weight * np.array(novelties) + \
                      (1 - novelty_weight) * np.array(qualities)
    
    # Identify the initial elite
    elite_idx = np.argmax(combined_scores)
    elite = population[elite_idx]
    best_states.append(elite)

    # --- Initial Archive Update ---
    for desc, n, q in zip(descriptors, novelties, qualities):
        if n >= novelty_threshold and q >= quality_threshold:
            archive.append(desc.copy())
    
    archive_history.append([desc.copy() for desc in archive])

    # --- Main Evolutionary Loop ---
    for epoch in range(num_epochs):
        # ----------------------------
        # Selection and Elitism
        # ----------------------------
        
        # Calculate selection probabilities based on combined score
        probs = combined_scores + 1e-6
        probs /= probs.sum()

        # Select parents (not including the elite in the selection pool)
        indices = np.random.choice(len(population), size=pop_size - 1, p=probs)
        parents_for_offspring = [population[i] for i in indices]

        # ----------------------------
        # Offspring Generation
        # ----------------------------
        offspring = []
        for _ in range(pop_size - 1):
            i1, i2 = np.random.choice(len(parents_for_offspring), 2, replace=True)
            p1, p2 = parents_for_offspring[i1], parents_for_offspring[i2]
            
            child = problem.crossover(p1, p2)
            if np.random.rand() < 0.5:
                child = problem.mutate(child)
            offspring.append(child)

        # ----------------------------
        # Offspring Evaluation
        # ----------------------------
        final_offs = [CGOL_Problem.simulate(ind, new_params)[-1] for ind in offspring]
        
        offspring_desc = [problem.behavior_descriptor(ind, new_params)
                            for ind in final_offs]
        
        offspring_novel = [
            problem.novelty(desc, archive, offspring_desc, k)
            for desc in offspring_desc
        ]
        
        offspring_quality = [
            problem.value(ind, new_params) for ind in final_offs
        ]

        # ----------------------------
        # Archive Update
        # ----------------------------
        for desc, n, q in zip(offspring_desc, offspring_novel, offspring_quality):
            if n >= novelty_threshold and q >= quality_threshold:
                archive.append(desc.copy())

        # Archive size maintenance
        if len(archive) > archive_max:
            num_to_remove = len(archive) - archive_max
            
            internal_novelties = [problem.novelty(d, archive, archive, k) for d in archive]
            remove_indices = np.argsort(internal_novelties)[:num_to_remove]
            archive = [archive[i] for i in range(len(archive)) if i not in remove_indices]

        archive_history.append([desc.copy() for desc in archive])

        # ----------------------------
        # New Population Formation
        # ----------------------------
        full_pop = population + offspring
        full_desc = descriptors + offspring_desc
        full_nov = novelties + offspring_novel
        full_qual = qualities + offspring_quality

        old_comb = combined_scores
        new_comb = novelty_weight * np.array(offspring_novel) + \
                   (1 - novelty_weight) * np.array(offspring_quality)
        full_comb = np.concatenate([old_comb, new_comb])
        
        elite_idx = np.argmax(full_comb)
        elite = full_pop[elite_idx]
        best_states.append(elite)
        
        best_idx = np.argsort(full_comb)[-pop_size:]
        
        population = [full_pop[i] for i in best_idx]
        descriptors = [full_desc[i] for i in best_idx]
        novelties = [full_nov[i] for i in best_idx]
        qualities = [full_qual[i] for i in best_idx]
        combined_scores = np.array([full_comb[i] for i in best_idx])
        
    return best_states, archive_history


def visualize_archive(archive_history, output_dir="plots"):
    """
    Create visualizations of the archive.
    
    Args:
        archive_history: List of archive states at each epoch
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Archive size over epochs
    archive_sizes = [len(archive) for archive in archive_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(archive_sizes, linewidth=2, color='steelblue')
    plt.title("Archive Size Over Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Archive Size", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/archive_size.png", dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/archive_size.png")
    
    # 2. Get final archive for detailed visualization
    final_archive = archive_history[-1]
    if len(final_archive) == 0:
        print("Warning: Final archive is empty!")
        return
    
    archive_array = np.array(final_archive)
    
    # Behavior descriptor dimensions:
    # [0] = population/100 (normalized)
    # [1] = width/20 (normalized)
    # [2] = height/20 (normalized)
    # [3] = density
    
    # 3. 2D scatter plots - different dimension pairs
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Population vs Width
    axes[0, 0].scatter(archive_array[:, 0], archive_array[:, 1], 
                       alpha=0.6, s=50, c=archive_array[:, 3], 
                       cmap='viridis', edgecolors='black', linewidth=0.5)
    axes[0, 0].set_xlabel("Population (normalized)", fontsize=11)
    axes[0, 0].set_ylabel("Width (normalized)", fontsize=11)
    axes[0, 0].set_title("Population vs Width\n(colored by density)", fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='Density')
    
    # Population vs Height
    axes[0, 1].scatter(archive_array[:, 0], archive_array[:, 2], 
                       alpha=0.6, s=50, c=archive_array[:, 3], 
                       cmap='viridis', edgecolors='black', linewidth=0.5)
    axes[0, 1].set_xlabel("Population (normalized)", fontsize=11)
    axes[0, 1].set_ylabel("Height (normalized)", fontsize=11)
    axes[0, 1].set_title("Population vs Height\n(colored by density)", fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Density')
    
    # Width vs Height
    axes[1, 0].scatter(archive_array[:, 1], archive_array[:, 2], 
                       alpha=0.6, s=50, c=archive_array[:, 0], 
                       cmap='plasma', edgecolors='black', linewidth=0.5)
    axes[1, 0].set_xlabel("Width (normalized)", fontsize=11)
    axes[1, 0].set_ylabel("Height (normalized)", fontsize=11)
    axes[1, 0].set_title("Width vs Height\n(colored by population)", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Population (norm)')
    
    # Population vs Density
    axes[1, 1].scatter(archive_array[:, 0], archive_array[:, 3], 
                       alpha=0.6, s=50, c=archive_array[:, 1] + archive_array[:, 2], 
                       cmap='coolwarm', edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel("Population (normalized)", fontsize=11)
    axes[1, 1].set_ylabel("Density", fontsize=11)
    axes[1, 1].set_title("Population vs Density\n(colored by size)", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Width+Height')
    
    plt.suptitle(f"Archive Visualization (Final Archive: {len(final_archive)} members)", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/archive_2d_scatter.png", dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/archive_2d_scatter.png")
    
    # 4. 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(archive_array[:, 0],  # Population
                        archive_array[:, 1],  # Width
                        archive_array[:, 2],  # Height
                        c=archive_array[:, 3],  # Density (color)
                        s=100,
                        alpha=0.7,
                        cmap='viridis',
                        edgecolors='black',
                        linewidth=0.5)
    
    ax.set_xlabel("Population (normalized)", fontsize=11)
    ax.set_ylabel("Width (normalized)", fontsize=11)
    ax.set_zlabel("Height (normalized)", fontsize=11)
    ax.set_title("3D Archive Visualization\n(colored by density)", fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Density', shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/archive_3d.png", dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/archive_3d.png")
    
    # 5. Archive evolution over time (showing how archive members change)
    # Sample a few epochs to show evolution
    sample_epochs = np.linspace(0, len(archive_history) - 1, min(5, len(archive_history)), dtype=int)
    
    fig, axes = plt.subplots(1, len(sample_epochs), figsize=(5*len(sample_epochs), 5))
    if len(sample_epochs) == 1:
        axes = [axes]
    
    for idx, epoch in enumerate(sample_epochs):
        archive_at_epoch = archive_history[epoch]
        if len(archive_at_epoch) == 0:
            axes[idx].text(0.5, 0.5, "Empty\nArchive", 
                          ha='center', va='center', fontsize=14)
            axes[idx].set_title(f"Epoch {epoch}\n(Empty)", fontsize=11)
            continue
        
        archive_epoch_array = np.array(archive_at_epoch)
        axes[idx].scatter(archive_epoch_array[:, 0], archive_epoch_array[:, 1],
                         alpha=0.6, s=50, c=archive_epoch_array[:, 3],
                         cmap='viridis', edgecolors='black', linewidth=0.5)
        axes[idx].set_xlabel("Population (norm)", fontsize=9)
        axes[idx].set_ylabel("Width (norm)", fontsize=9)
        axes[idx].set_title(f"Epoch {epoch}\n({len(archive_at_epoch)} members)", fontsize=11)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle("Archive Evolution Over Time", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/archive_evolution.png", dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/archive_evolution.png")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ARCHIVE SUMMARY STATISTICS")
    print("="*60)
    print(f"Final archive size: {len(final_archive)}")
    print(f"Max archive size: {max(archive_sizes)}")
    print(f"Min archive size: {min(archive_sizes)}")
    print(f"Average archive size: {np.mean(archive_sizes):.2f}")
    print("\nFinal Archive Statistics (behavior descriptors):")
    print(f"  Population (norm): mean={archive_array[:, 0].mean():.3f}, std={archive_array[:, 0].std():.3f}")
    print(f"  Width (norm):      mean={archive_array[:, 1].mean():.3f}, std={archive_array[:, 1].std():.3f}")
    print(f"  Height (norm):     mean={archive_array[:, 2].mean():.3f}, std={archive_array[:, 2].std():.3f}")
    print(f"  Density:           mean={archive_array[:, 3].mean():.3f}, std={archive_array[:, 3].std():.3f}")
    print("="*60)


def main():
    """Main function to run NS-Q and visualize archive."""
    print("Running Novelty Search with Quality (NS-Q) with Archive Visualization")
    print("="*60)
    
    # Configuration
    parameters = Parameters(
        steps=100,
        include_between=False,
        expansion=-1
    )
    
    # Create problem
    problem = GrowthProblem(
        state_generator=create_initial_state,
        type="random",
        enable_presets=True
    )
    
    # Run NS-Q with archive tracking
    print("\nRunning NS-Q algorithm...")
    best_states, archive_history = novelty_search_with_quality_track_archive(
        problem=problem,
        parameters=parameters,
        pop_size=100,
        num_epochs=50,
        k=5,
        novelty_threshold=1.0,
        quality_threshold=100.0,
        archive_max=300,
        novelty_weight=0.3
    )
    
    print(f"Completed {len(archive_history)} epochs")
    print(f"Final archive size: {len(archive_history[-1])}")
    
    # Visualize archive
    print("\nCreating visualizations...")
    visualize_archive(archive_history, output_dir="plots")
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

