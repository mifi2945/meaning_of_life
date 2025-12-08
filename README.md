# The Meaning of Life
## CSC 4631 Final Project
#### Authors: Alex Ewart, Mikhail Filippov

The goal of this project is to implement the Novelty Search with Quality (NS-Q) on Conway's Game of Life (CGoL, or alternatevely, Seagull).

## GPU Acceleration with PyTorch/CUDA

This project now includes GPU-accelerated parallel implementations using PyTorch and CUDA for significantly faster novelty search and genetic algorithm execution.

### Features

- **Batch Processing**: Simulate multiple Game of Life states simultaneously on GPU
- **Parallel Genetic Algorithm**: Process entire populations in parallel
- **Novelty Search**: GPU-accelerated novelty search for exploring diverse solutions
- **Automatic Device Selection**: Automatically uses CUDA if available, falls back to CPU

### Installation

Make sure PyTorch is installed (it's included in `pyproject.toml`):

```bash
# Install dependencies (using uv or pip)
uv sync
# or
pip install -e .
```

### Usage

#### Command Line

Run novelty search with GPU acceleration:

```bash
python src/main.py -a novelty -p growth --steps 100
```

Run parallel genetic algorithm:

```bash
python src/main.py -a GA_parallel -p growth --steps 100
```

#### Python API

```python
from algorithms import novelty_search, genetic_algorithm_parallel
from problems import GrowthProblem, Parameters

# Create problem
problem = GrowthProblem(state_generator=create_initial_state)
parameters = Parameters(steps=100, include_between=False, expansion=-1)

# Run novelty search
results = novelty_search(
    problem=problem,
    parameters=parameters,
    pop_size=100,
    num_epochs=100,
    novelty_threshold=0.1,
    archive_size=1000,
    k=15
)

# Run parallel genetic algorithm
results = genetic_algorithm_parallel(
    problem=problem,
    parameters=parameters,
    pop_size=100,
    num_epochs=100
)
```

### Performance Tips

1. **Batch Size**: Use `batch_size` equal to `pop_size` for maximum GPU utilization
2. **Population Size**: Larger populations benefit more from parallelization
3. **GPU Memory**: If you run out of memory, reduce `batch_size` or `pop_size`
4. **Novelty Search**: Typically finds more diverse solutions than fitness-based search

### Architecture

- `src/parallel_sim.py`: PyTorch-based batch Game of Life simulation
- `src/parallel_algorithms.py`: GPU-accelerated genetic algorithm and novelty search
- `src/algorithms.py`: Wrapper functions for easy integration