# The Meaning of Life
## CSC 4631 Final Project
##### Authors: Alex Ewart, Mikhail Filippov

### Project Goal: 
Implement **Novelty Search with Quality (NS-Q)** on **Conway's Game of Life** (CGoL, or alternatevely, Seagull).

### Tasks: 
- Solve Growth Problem: Find states which expand to the largest final cell counts
- Solve Migration Problem: Find states which travel the furthest from the center

### Why: 
CGoL is Turing Complete, meaning the game itself is like a programming language (an inefficient one at that). Furthermore, the state-space of CGoL makes it a perfect candidate for NS-Q, as the large and volatile search-space makes even the smallest change propagate to the end of the simulation; NS-Q's nature to search through novel solutions makes it a prime candidate to search through enormous state-spaces, looking for solutions that are not only valuable, but also different from previous ones.


### Project Structure:
- `src/`  ............-> the source folder for all program `.py` files
    - `algorithms.py` ............-> contains the 3 algorithm functions (`HC`, `GA`, `NS-Q`) and their implementations
    - `main.py` ............-> main thread to start search and program, utilizing the CLI for options
    - `NS_Q_tester.py` ............-> evaluation tester used to generate 3 graphs for evaluation of NS-Q
    - `problems.py` ............-> `CGoL` environment/simulation logic and implemenations of `Problem` classes for `Growth` and `Migration`
    - `visualizer.py` ............-> `TODO`

### Program:
The program utilizes the `uv` package to sync all required packages:

1. Install `uv`: `pip install uv`
2. From the root directory of the project, call: `uv sync`
    - This should install and setup the needed packages; **if you alternatively would like to install the required packages in a different manner, the packages can be found in the `pyproject.toml`**
3. To run the program: `uv run src/main.py`
4. Options (which can also be listed by the `-h` or `--help` flag):
    - `-s, --steps <STEPS>`: Number of simulation steps (default: 100)
    - `-v, --visualize`: Visualize the simulation (default: True)
    - `-d, --delay <DELAY>`: Delay between frames for visualization in milliseconds (default: 100)
    - `--grid-width <GRID_WIDTH>`: Display grid width (default: 100)
    - `-a, --search-type {default,hill_climbing,GA,NS_Q}`: Search algorithm types; `default` for direct simulation, `hill_climbing` for hill climbing, `GA` for Genetic Algorithm, `NS_Q` for Novelty Search (default: default)
    - `-p, --problem-type {growth,migration}`: Problem type to solve; `growth` for growth problem, `migration` for migration problem (default: growth)
    - `-i, --initial-state {random,empty}`: Initial state generation; `random` for random start, `empty` for an empty initial state
    - `--grid-size <GRID_SIZE>`: Initial grid size (NxN). Default: 10 (10x10 = 100 cells)
    - `--no-presets`: Disable pattern presets (only use single bit flips). By default, presets are enabled.
5. At the end, `main.py` will: 
    - Return the final results of each simulation based on the `value()` metric of each problem in the CLI: distance from the center for `Migration`, total live cell count for `Growth`.
    - Display the simulation if `-v` was selected. This PyGame can be stepped through with `left` and `right` keys; or `space` starts the animation, playing through to the end.
6. `NS_Q_tester.py` can be ran with `uv run src/NS_Q_tester.py`
    - Saved plots (as shown in the final presentation) will be saved to `growth_plots/`, `migration_plots/`, and `plots/`