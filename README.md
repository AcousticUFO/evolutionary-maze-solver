# Maze Runner: Evolutionary AI & Deterministic Oracle Maze Solver

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-PEP8-green.svg)](https://pep8.org/)

An end-to-end framework for **generating perfect mazes**, computing **ground-truth optimal trajectories via a Wavefront Dijkstra Oracle**, and training **autonomous pathfinding agents using an Evolutionary Genetic Algorithm** equipped with **Pivot Mutation**, **Stigmergic Pheromones**, and an adaptive **Deluge Mechanism**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Pipeline](#architecture--pipeline)
- [Algorithmic Foundations & Design Choices](#algorithmic-foundations--design-choices)
  - [Part 1: Environment Generation (Iterative DFS)](#part-1-environment-generation-iterative-dfs)
  - [Part 2: The Reference Oracle (Wavefront Dijkstra)](#part-2-the-reference-oracle-wavefront-dijkstra)
  - [Part 3: The Evolutionary AI (Genetic Algorithm)](#part-3-the-evolutionary-ai-genetic-algorithm)
- [Performance Benchmarks](#performance-benchmarks)
- [Project Structure](#project-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Command-Line Interface](#command-line-interface)
- [Running Unit Tests](#running-unit-tests)
- [Author & License](#author--license)

---

## Overview

Traditional search algorithms (A*, BFS, Dijkstra) solve static environments with mathematical perfection. However, biological organisms solve spatial navigation problems through trial, error, adaptation, and memory.

This project bridges both paradigms:
1. **The Deterministic Oracle (Wavefront Dijkstra)** computes the exact topological distance field and optimal path, establishing the mathematical ground truth.
2. **The Evolutionary AI (Genetic Algorithm)** learns to navigate complex, tortuous mazes up to 100 × 100 without prior coordinate knowledge, utilizing the Oracle solely as a fitness gradient.

```
+------------------------+      +---------------------------+      +--------------------------+
|  Iterative DFS Engine  | ---> |  Wavefront Dijkstra BFS   | ---> | Evolutionary AI Solver   |
|  - 1D Bytearray Grid   |      |  - O(1) Deque Propagation |      | - Pivot Mutation         |
|  - Border Padding      |      |  - Vector Flow Field      |      | - Stigmergic Pheromones  |
|  - O(N²) Complexity    |      |  - True Distance Matrix   |      | - Adaptive Deluge Surges |
+------------------------+      +---------------------------+      +--------------------------+
```

---

## Architecture & Pipeline

The repository follows a clean modular architecture separating computation, configuration, testing, and visualization:

```
├── maze_runner/                 # Core Python package
│   ├── __init__.py              # Package public exports
│   ├── config.py                # Hyperparameter dataclass (GeneticConfig)
│   ├── generator.py             # Iterative DFS 1D bytearray maze generator
│   ├── oracle.py                # Wavefront Dijkstra, distance field & vector flow field
│   ├── genetic.py               # Genetic Algorithm: Pivot mutation, stigmergy, deluge
│   └── visualization.py         # Decoupled visualization routines & analytical dashboard
├── tests/                       # Automated test suite (unittest / pytest compatible)
│   ├── __init__.py
│   ├── test_generator.py        # Generator shape, bounds, and BFS tests
│   ├── test_oracle.py           # Dijkstra distances, Moore neighborhood, vector fields
│   └── test_genetic.py          # Genome simulation, fitness, and evolutionary cycles
├── main.py                      # CLI entrypoint with argparse and profiling
├── requirements.txt             # Minimal dependencies (numpy, matplotlib)
├── .gitignore                   # Comprehensive Python & asset ignore file
└── README.md                    # Technical documentation
```

---

## Algorithmic Foundations & Design Choices

### Part 1: Environment Generation (Iterative DFS)

The generator constructs **perfect mazes** (spanning trees with no loops, no isolated islands, and exactly one path between any two cells).

#### Key Engineering Decisions:
1. **Iterative Stack over Recursion**:
   Standard recursive DFS in Python rapidly crashes on grids larger than N = 35 due to `RecursionError` (Python's default call stack limit of 1000). The generator employs an explicit list stack (`stack.append()`, `stack.pop()`), allowing seamless generation of mazes beyond N = 1000.
2. **1D Array Flattening with 1-Cell Border Padding**:
   Rather than nested lists `grid[x][y]`, the grid is stored as a 1D sequence of size (N + 2) × (N + 2). The outer boundary is pre-marked with sentinel values (`2 = border`). This eliminates conditional boundary checks (`0 <= x < N`) inside the hot exploration loop.
3. **`bytearray` Memory Optimization**:
   Each cell occupies exactly 1 byte (`0 = wall`, `1 = passage`, `2 = border`). A 500 × 500 maze (250,000 cells) consumes only **~244 KB of RAM**, compared to > 7.5 MB for standard Python list-of-lists representations.
4. **Empirical Complexity Proof (T / N² Ratio)**:
   Empirical benchmarking demonstrates strict O(N²) time complexity:
   - For N = 50: `T / N² ≈ 1.33 × 10⁻⁶ s/cell`
   - For N = 300: `T / N² ≈ 1.34 × 10⁻⁶ s/cell`
   - For N = 500: `T / N² ≈ 1.35 × 10⁻⁶ s/cell`

   Average processing cost per cell:
   ```text
   T / N² ≈ 1.34 × 10⁻⁶ s/cell
   ```

---

### Part 2: The Reference Oracle (Wavefront Dijkstra)

The Oracle calculates the exact shortest corridor distance from every walkable cell to the goal coordinate.

#### Key Engineering Decisions:
1. **Inverse Wavefront Propagation**:
   Rather than running point-to-point Dijkstra searches for each agent, the algorithm propagates outward from the destination (`(0, 0)`), saturating the entire grid in a single pass.
2. **O(1) Queue Expansion via `collections.deque`**:
   Because all edge weights are uniform (1 step), the algorithm simplifies from a priority queue O(E log V) to a Breadth-First Wavefront O(V + E) = O(N²). Utilizing `collections.deque.popleft()` ensures true O(1) pop operations.
3. **8-Connected Moore Neighborhood**:
   Movements permit 8 discrete directions (orthogonal + diagonal). A cell `(i, j)` connects to `(k, l)` if:

   ```text
   |i - k| ≤ 1,   |j - l| ≤ 1,   and   1 ≤ |i - k| + |j - l| ≤ 2
   ```

4. **Empirical Quadratic Path Growth Law**:
   By analyzing hundreds of generated perfect mazes, the length of the optimal path `L(N)` was empirically modeled as:

   ```text
   L(N) ≈ 0.035 · N² + 4.64 · N + 89
   ```

   This law is fundamental for sizing agent genomes in Part 3 (genome length `L = 25 × N`).
5. **Vector Flow Field Generation**:
   A vector field (dx, dy) is derived from the distance map. Each cell stores a direction vector pointing to the neighbor with the minimum remaining distance, enabling instantaneous deterministic path reconstruction.

---

### Part 3: The Evolutionary AI (Genetic Algorithm)

The AI navigates the maze without coordinate or map knowledge. Movements emerge purely through natural selection.

#### 1. Genome Representation & Physics Simulation
- **Genome**: Sequence of integers `g_k ∈ {0, 1, 2, 3, 4, 5, 6, 7}` encoding 8-directional steps.
- **Search Space**: With genome length `L = 25 × N`, the combinatorial space has size `8ᴸ` (e.g., `8²⁵⁰⁰` combinations for `N = 100`), rendering brute-force approaches impossible.
- **Simulation Engine (`simulate_genome`)**: An agent steps through its genome until it collides with a physical wall or virtual pheromone wall. To maximize CPU efficiency, the simulation terminates immediately upon collision.

#### 2. Multi-Objective Fitness Function ('ness' Score)
A common pitfall in maze navigation is using Euclidean distance to the goal, which leads agents into dead-end traps near the target separated by a wall. Our fitness function uses the **Dijkstra Oracle** as a true corridor gradient:

```text
Fitness = Dijkstra_Distance(end, goal) + 1.2 × (L_max - steps) + Immobility_Penalty
```

- **Dijkstra Distance (`Dijkstra_Distance(end, goal)`)**: True topological step distance through walkable corridors to reach the goal (from the Oracle).
- **Exploration Bonus (`1.2 × (L_max - steps)`)**: Rewards agents that explore deeper into corridors before colliding.
- **Immobility Penalty (+5000)**: Eliminates static agents that make ≤ 1 step.

#### 3. Three Evolutionary Innovations:
1. **Pivot Mutation (Targeted Local Search)**:
   Standard random gene mutations in sequential paths break the entire trajectory downstream. Pivot mutation:
   - Identifies the exact index where the champion collided (the **Pivot**).
   - Keeps the valid trajectory prefix up to the pivot intact.
   - Regenerates exploratory movement genes strictly *after* the collision point.
2. **Stigmergic Pheromones (Spatial Memory)**:
   Inspired by ant colony foraging, if the champion's best distance stagnates for ≥ 25 generations, its landing coordinate is marked as a **virtual wall** (`virtual_pheromone_walls[x, y] = 1`). Future generations treat this coordinate as impassable, forcing the population to abandon dead-end corridors.
3. **The Deluge Mechanism (Anti-Inbreeding Reset)**:
   When population diversity collapses (< 5 unique landing spots) or prolonged stagnation occurs (> 15 generations), the mutation probability surges from 0.15 to 0.70. This massive genetic perturbation breaks local minima plateaus.

---

## Performance Benchmarks

All benchmarks performed on Linux x86_64:

### 1. Maze Generation (Iterative DFS + 1D Bytearray)
| Size (N × N) | Total Cells | Execution Time (s) | RAM Consumption |
|:------------:|:-----------:|:------------------:|:---------------:|
| 10 × 10      | 100         | 0.0001 s           | ~10 KB          |
| 50 × 50      | 2,500       | 0.0033 s           | ~25 KB          |
| 100 × 100    | 10,000      | 0.0135 s           | ~48 KB          |
| 300 × 300    | 90,000      | 0.1210 s           | ~140 KB         |
| 500 × 500    | 250,000     | 0.3380 s           | ~244 KB         |

### 2. Wavefront Dijkstra vs Naive Iterative Scan
| Size (N) | Wavefront Deque (Our Implementation) | Naive Matrix Scan | Speedup Factor |
|:--------:|:------------------------------------:|:-----------------:|:--------------:|
| 10       | 0.000095 s                           | 0.000319 s        | **3.3x**       |
| 50       | 0.001139 s                           | 0.007265 s        | **6.3x**       |
| 100      | 0.004416 s                           | 0.046831 s        | **10.6x**      |
| 1000     | 0.470000 s                           | > 45.0000 s       | **> 95x**      |

---

## Installation & Quickstart

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/AcousticUFO/maze-runner.git
cd maze-runner
pip install -r requirements.txt
```

### 2. Run the Simulation
```bash
# Run standard 50x50 maze simulation
python main.py

# Run smaller fast demonstration (15x15) without displaying plot window
python main.py --size 15 --no-plot

# Run 40x40 simulation and save custom dashboard image
python main.py --size 40 --save-plot my_dashboard.png
```

---

## Command-Line Interface

```
usage: main.py [-h] [-s SIZE] [--selection-rate SELECTION_RATE]
               [--mutation-rate MUTATION_RATE] [-p POPULATION]
               [-l GENOME_LENGTH] [-g MAX_GENERATIONS] [--seed SEED]
               [--no-plot] [--save-plot SAVE_PLOT] [--show-plot]

Maze Runner: Evolutionary Genetic Algorithm Maze Solver

options:
  -h, --help            show this help message and exit
  -s, --size SIZE       Side length of the square maze (N x N) (default: 50)
  --selection-rate RATE Proportion of top-ranking individuals selected for breeding (default: 0.15)
  --mutation-rate RATE  Base mutation probability during crossover (default: 0.15)
  -p, --population POP  Population size (defaults to 10 * size if omitted)
  -l, --genome-length L Length of movement genome (defaults to 25 * size if omitted)
  -g, --max-generations Maximum generation limit before termination
  --seed SEED           Random seed for reproducible maze and initial population
  --no-plot             Disable generating the Matplotlib visual dashboard
  --save-plot PATH      File path to save the resulting dashboard image
  --show-plot           Display interactive plot window after simulation completion
```

---

## Analytical Dashboard

When a simulation completes, a 4-panel analytical dashboard is generated:

1. **Top-Left (Trajectory Comparison)**: Overlays the ground-truth Dijkstra Oracle path (translucent blue) against the trajectory navigated by the AI agent (red).
2. **Top-Right (Metrics Summary)**: Summary card displaying grid size, population, genome length, execution time, outcome, and closest distance achieved.
3. **Bottom-Left (Fitness Convergence)**: Tracks the best agent's remaining distance to the goal across generations, highlighting step-down convergence plateaus.
4. **Bottom-Right (Adaptive Mutation History)**: Visualizes mutation rate fluctuations, clearly identifying Deluge spikes (mutation rate = 0.70) triggered during stagnation.

---

## Running Unit Tests

The test suite covers all components (generator, oracle, and genetic solver) using Python's built-in `unittest`:

```bash
python3 -m unittest discover -s tests -v
```

Output:
```
test_find_nearest_free_cell_already_free (test_generator.TestMazeGenerator) ... ok
test_find_nearest_free_cell_on_wall (test_generator.TestMazeGenerator) ... ok
test_generate_maze_dimensions_and_binary_values (test_generator.TestMazeGenerator) ... ok
test_generate_maze_invalid_size (test_generator.TestMazeGenerator) ... ok
test_generate_maze_seed_reproducibility (test_generator.TestMazeGenerator) ... ok
test_calculate_fitness (test_genetic.TestGeneticSolver) ... ok
test_evolution_short_integration_run (test_genetic.TestGeneticSolver) ... ok
test_reconstruct_optimal_path (test_genetic.TestGeneticSolver) ... ok
test_simulate_genome_empty_and_collision (test_genetic.TestGeneticSolver) ... ok
test_solver_initialization_and_config (test_genetic.TestGeneticSolver) ... ok
test_build_vector_field (test_oracle.TestDijkstraOracle) ... ok
test_compute_dijkstra_distances_diagonal_and_propagation (test_oracle.TestDijkstraOracle) ... ok
test_compute_dijkstra_walls_marked (test_oracle.TestDijkstraOracle) ... ok
test_solve_maze_vector_field (test_oracle.TestDijkstraOracle) ... ok

----------------------------------------------------------------------
Ran 14 tests in 0.032s

OK
```

---

## Author & License

- **Author**: Camil Hery
- **License**: Released under the [MIT License](LICENSE).
