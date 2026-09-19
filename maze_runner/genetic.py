"""Evolutionary Artificial Intelligence for autonomous maze traversal.

This module implements GeneticMazeSolver, which utilizes a genetic algorithm
tailored for maze navigation. Key evolutionary mechanisms include:
  - Pivot Mutation: Local search targeting the collision point to preserve valid
    path prefixes while exploring alternative routes.
  - Stigmergic Pheromones: Spatial memory marking unproductive dead-ends as virtual
    walls when progress stagnates.
  - Deluge Mechanism: Dynamic mutation rate surge (0.15 -> 0.70) triggered when
    spatial diversity collapses, escaping deep local minima.
"""

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from maze_runner.config import GeneticConfig
from maze_runner.generator import find_nearest_free_cell, generate_maze_dfs
from maze_runner.oracle import compute_dijkstra_distances


# 8-directional movement index to displacement mapping
MOVEMENT_DIRECTIONS: Dict[int, Tuple[int, int]] = {
    0: (0, 1),    # East
    1: (-1, 1),   # North-East
    2: (-1, 0),   # North
    3: (-1, -1),  # North-West
    4: (0, -1),   # West
    5: (1, -1),   # South-West
    6: (1, 0),    # South
    7: (1, 1),    # South-East
}


class GeneticMazeSolver:
    """Genetic Algorithm Solver for navigating complex mazes.

    Attributes:
        config: GeneticConfig instance containing hyperparameters.
        maze: 2D binary numpy array of the environment.
        start: Starting (row, col) coordinates.
        goal: Target (row, col) coordinates.
        dijkstra_oracle: Distance map from every cell to the goal.
        virtual_pheromone_walls: Binary grid of virtual walls representing stigmergic dead-ends.
        population: Current generation of genomes.
        best_genome_history: Overall best genome discovered throughout evolution (Hall of Fame).
        record_distance: Closest distance to the goal achieved so far.
        success_generation: Generation index at which the goal was reached, or -1 if unsuccessful.
        fitness_history: Chronological list of best distance per generation.
        mutation_rate_history: Chronological list of adaptive mutation rates.
    """

    def __init__(
        self,
        config: Optional[GeneticConfig] = None,
        maze: Optional[np.ndarray] = None,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the genetic maze solver.

        Args:
            config: Optional configuration dataclass. Defaults to GeneticConfig().
            maze: Optional pre-generated maze. If None, generated via generate_maze_dfs.
            start: Optional start coordinates. If None, chosen near bottom-right corner.
            goal: Optional goal coordinates. If None, chosen near top-left corner.
            seed: Optional random seed for reproducible initialization.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.config = config if config is not None else GeneticConfig()
        grid_size = self.config.grid_size

        # 1. Environment initialization
        self.maze = (
            maze if maze is not None else generate_maze_dfs(grid_size, seed=seed)
        )
        self.start = (
            start
            if start is not None
            else find_nearest_free_cell(self.maze, (grid_size - 1, grid_size - 1))
        )
        self.goal = (
            goal if goal is not None else find_nearest_free_cell(self.maze, (0, 0))
        )

        # 2. Distance Oracle initialization (provides real corridor distance)
        self.dijkstra_oracle = compute_dijkstra_distances(self.maze, self.goal)

        # 3. Spatial memory (Stigmergic Pheromones)
        self.virtual_pheromone_walls = np.zeros(self.maze.shape, dtype=np.int8)

        # 4. Initial random population
        self.population: List[List[int]] = [
            [random.randint(0, 7) for _ in range(self.config.genome_length)]
            for _ in range(self.config.population_size)
        ]

        # 5. Tracking and telemetry
        self.best_genome_history: Optional[List[int]] = None
        self.record_distance: float = float("inf")
        self.success_generation: int = -1
        self.fitness_history: List[float] = []
        self.mutation_rate_history: List[float] = []

    def reconstruct_optimal_path(self) -> List[Tuple[int, int]]:
        """Reconstruct the ground-truth optimal path from start to goal via Dijkstra Oracle.

        Returns:
            List[Tuple[int, int]]: Sequential coordinates along the optimal path.
        """
        path: List[Tuple[int, int]] = [self.start]
        current = self.start
        grid_size = self.config.grid_size

        while current != self.goal:
            cx, cy = current
            best_next: Optional[Tuple[int, int]] = None
            min_dist = self.dijkstra_oracle[cx, cy]

            for dx, dy in MOVEMENT_DIRECTIONS.values():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    dist_neighbor = self.dijkstra_oracle[nx, ny]
                    if dist_neighbor != -1 and dist_neighbor < min_dist:
                        min_dist = dist_neighbor
                        best_next = (nx, ny)

            if best_next is None or best_next == current:
                break

            current = best_next
            path.append(current)

        return path

    def simulate_genome(
        self, genome: List[int]
    ) -> Tuple[Tuple[int, int], int, List[Tuple[int, int]]]:
        """Simulate agent physics along a sequence of directional genes.

        Movement terminates immediately upon collision with a physical wall or a
        virtual pheromone wall.

        Args:
            genome: List of directional movement integers (0-7).

        Returns:
            Tuple containing:
                - (final_x, final_y): Coordinates where agent halted.
                - valid_steps: Count of successful steps before collision.
                - trajectory: Complete list of visited coordinates.
        """
        x, y = self.start
        trajectory: List[Tuple[int, int]] = [(x, y)]
        grid_size = self.config.grid_size

        for direction in genome:
            dx, dy = MOVEMENT_DIRECTIONS[direction]
            nx, ny = x + dx, y + dy

            # Check boundaries, physical walkable corridors, and pheromone walls
            if (
                0 <= nx < grid_size
                and 0 <= ny < grid_size
                and self.maze[nx, ny] == 1
                and self.virtual_pheromone_walls[nx, ny] == 0
            ):
                x, y = nx, ny
                trajectory.append((x, y))
            else:
                # Collision detected: stop simulation early for computational speed
                break

        return (x, y), len(trajectory) - 1, trajectory

    def calculate_fitness(self, genome: List[int]) -> float:
        """Compute the multi-objective fitness score ('ness' score) to minimize.

        Score components:
          1. True Distance: Remaining corridor steps to the goal (from Dijkstra Oracle).
          2. Exploration Incentive: Penalty proportional to unused genome length to
             discourage lazy early collisions.
          3. Immobility Penalty: Massive penalty if the agent makes <= 1 step.

        Args:
            genome: List of directional movement integers (0-7).

        Returns:
            float: Fitness score (lower is better).
        """
        (ex, ey), steps_taken, _ = self.simulate_genome(genome)

        # 1. Distance through maze corridors
        oracle_val = self.dijkstra_oracle[ex, ey]
        distance_to_goal = (
            float(oracle_val)
            if oracle_val >= 0
            else self.config.unreachable_penalty
        )

        # 2. Exploration bonus (penalize unused genome steps)
        unused_steps = self.config.genome_length - steps_taken
        exploration_penalty = self.config.exploration_bonus_weight * unused_steps

        # 3. Penalty against static agents
        immobility_penalty = (
            self.config.immobility_penalty if steps_taken <= 1 else 0.0
        )

        return distance_to_goal + exploration_penalty + immobility_penalty

    def run_evolution(
        self, verbose: bool = True, log_interval: int = 50
    ) -> List[int]:
        """Execute the main evolutionary loop.

        Args:
            verbose: If True, prints periodic progress logs to console.
            log_interval: Interval of generations between console log outputs.

        Returns:
            List[int]: Best genome discovered during the evolutionary run.
        """
        stagnation_counter = 0
        current_mutation_rate = self.config.base_mutation_rate

        if verbose:
            print(
                f"Starting evolutionary simulation (Max {self.config.max_generations} generations, "
                f"Pop={self.config.population_size}, Genome={self.config.genome_length})..."
            )

        for generation in range(self.config.max_generations):
            evaluation_results: List[Dict[str, Any]] = []
            landing_positions: Set[Tuple[int, int]] = set()

            # 1. Population evaluation
            for individual in self.population:
                (end_x, end_y), steps, _ = self.simulate_genome(individual)
                dist = self.dijkstra_oracle[end_x, end_y]
                fitness = self.calculate_fitness(individual)

                evaluation_results.append({
                    "genome": individual,
                    "dist": dist,
                    "fitness": fitness,
                    "steps": steps,
                    "end_pos": (end_x, end_y),
                })
                landing_positions.add((end_x, end_y))

            # 2. Sort population by fitness (ascending order: lower is better)
            evaluation_results.sort(key=lambda item: item["fitness"])
            gen_champion = evaluation_results[0]

            self.fitness_history.append(gen_champion["dist"])
            self.mutation_rate_history.append(current_mutation_rate)

            # 3. Track historical record (Hall of Fame)
            if gen_champion["dist"] < self.record_distance:
                self.record_distance = gen_champion["dist"]
                self.best_genome_history = list(gen_champion["genome"])
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # 4. Periodic progress reporting
            if verbose and (generation % log_interval == 0 or generation == self.config.max_generations - 1):
                print(
                    f"Gen: {generation:4d} | Best Dist: {gen_champion['dist']:3d} | "
                    f"Steps: {gen_champion['steps']:4d} | Mutation: {current_mutation_rate:.2f} | "
                    f"Diversity: {len(landing_positions):2d}"
                )

            # 5. Victory condition check
            if gen_champion["dist"] == 0:
                if verbose:
                    print(f"\n>>> Goal reached successfully at generation {generation}! <<<")
                self.success_generation = generation
                self.best_genome_history = list(gen_champion["genome"])
                return gen_champion["genome"]

            # 6. Pheromone placement (Stigmergy)
            # If population stagnates, mark the champion's halting position as a virtual wall
            if stagnation_counter >= self.config.pheromone_stagnation_limit:
                self.virtual_pheromone_walls[gen_champion["end_pos"]] = 1
                stagnation_counter = 0

            # 7. Deluge mechanism (adaptive mutation rate surge)
            # Triggered if spatial diversity drops critically low or prolonged stagnation persists
            if (
                len(landing_positions) < self.config.deluge_diversity_limit
                or stagnation_counter > self.config.deluge_stagnation_limit
            ):
                current_mutation_rate = self.config.deluge_mutation_rate
            else:
                current_mutation_rate = self.config.base_mutation_rate

            # 8. Breeding next generation
            # Elitism: Retain generation champion and overall historic best
            next_generation: List[List[int]] = [
                list(gen_champion["genome"]),
                list(self.best_genome_history),
            ]

            # Parent selection pool (top selection_rate %)
            pool_size = max(2, int(self.config.population_size * self.config.selection_rate))
            parent_pool = [entry["genome"] for entry in evaluation_results[:pool_size]]

            half_length = self.config.genome_length // 2

            while len(next_generation) < self.config.population_size:
                # Tournament / Pool random selection
                p1, p2 = random.sample(parent_pool, 2)

                # Single-point crossover
                child = p1[:half_length] + p2[half_length:]

                # PIVOT MUTATION
                # Preserves verified movement up to collision, regenerates exploratory tail
                if random.random() < current_mutation_rate:
                    pivot = max(0, gen_champion["steps"] - random.randint(0, 3))
                    for idx in range(pivot, self.config.genome_length):
                        child[idx] = random.randint(0, 7)

                next_generation.append(child)

            self.population = next_generation

        if verbose:
            print(f"\nEvolution reached maximum generations ({self.config.max_generations}).")
            print(f"Closest distance achieved: {self.record_distance}")

        return self.best_genome_history or evaluation_results[0]["genome"]
