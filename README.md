# The Meaning of Life
## CSC 4631 Final Project
#### Authors: Alex Ewart, Mikhail Filippov

The goal of this project is to implement the Novelty Search with Quality (NS-Q) on Conway's Game of Life (CGoL, or alternatively, Seagull).

```bash
# Run a simple simulation (default: 100 steps, no visualization)
python -m src.main

# Run with visualization
python -m src.main --visualize

# Run with a specific search algorithm
python -m src.main --search-type NS_Q --visualize
```

#### Command-Line Arguments

**Simulation Parameters:**
- `-s, --steps <int>`: Number of simulation steps (default: 100)
- `-e, --expansion <int>`: Grid expansion size. Use `-1` for infinite expansion (default: -1)
- `-a, --search-type <str>`: Search algorithm type
  - `default`: Direct simulation (no search)
  - `hill_climbing`: Hill Climbing search
  - `GA`: Genetic Algorithm
  - `NS_Q`: Novelty Search with Quality
- `-p, --problem-type <str>`: Problem type
  - `growth`: Growth problem (maximize living cells)
  - `migration`: Migration problem
- `--no-cuda`: Disable CUDA/GPU acceleration (use CPU instead)

**Visualization Parameters:**
- `-v, --visualize`: Enable visualization (opens pygame window)
- `-d, --delay <int>`: Delay between frames in milliseconds (default: 100)
- `--grid-width <int>`: Display grid width in pixels (default: 100)

**State Management:**
- `--save`: Save simulation state to `saved_state.npz` file

#### Examples

```bash
# Run NS-Q with 200 steps and visualization
python -m src.main --search-type NS_Q --steps 200 --visualize

# Run Genetic Algorithm with custom delay
python -m src.main --search-type GA --visualize --delay 50

# Run Hill Climbing and save the result
python -m src.main --search-type hill_climbing --save

# Run on CPU (no GPU)
python -m src.main --search-type NS_Q --no-cuda

# Run migration problem
python -m src.main --search-type NS_Q --problem-type migration --visualize
```
#### Viewing Saved States

To visualize a previously saved simulation state:

```bash
# View default saved state
python -m src.visualize_saved

# View a specific saved file
python -m src.visualize_saved -f saved_state.npz

# Customize visualization settings
python -m src.visualize_saved -f saved_state.npz --delay 200 --grid-width 150
```


