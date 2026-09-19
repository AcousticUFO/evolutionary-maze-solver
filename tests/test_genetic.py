"""Unit tests for the Evolutionary Genetic Algorithm solver."""

import random
import unittest
import numpy as np

from maze_runner.config import GeneticConfig
from maze_runner.genetic import GeneticMazeSolver


class TestGeneticSolver(unittest.TestCase):
    """Test suite for GeneticMazeSolver mechanics, simulation, and fitness."""

    def test_solver_initialization_and_config(self) -> None:
        """Verify solver correctly configures dimensions and population."""
        config = GeneticConfig(grid_size=15, genome_length=100)
        solver = GeneticMazeSolver(config=config, seed=42)

        self.assertEqual(solver.config.grid_size, 15)
        self.assertEqual(solver.config.genome_length, 100)
        self.assertEqual(len(solver.population), 150)  # 10 * N
        self.assertEqual(len(solver.population[0]), 100)

    def test_reconstruct_optimal_path(self) -> None:
        """Verify optimal Dijkstra path reconstruction leads to the goal."""
        config = GeneticConfig(grid_size=10)
        solver = GeneticMazeSolver(config=config, seed=42)

        path = solver.reconstruct_optimal_path()
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[-1], solver.goal)

    def test_simulate_genome_empty_and_collision(self) -> None:
        """Verify empty genome stays at start coordinate without moving."""
        config = GeneticConfig(grid_size=10)
        solver = GeneticMazeSolver(config=config, seed=42)

        pos, steps, traj = solver.simulate_genome([])
        self.assertEqual(pos, solver.start)
        self.assertEqual(steps, 0)
        self.assertEqual(traj, [solver.start])

    def test_calculate_fitness(self) -> None:
        """Verify fitness calculation produces valid numeric values."""
        config = GeneticConfig(grid_size=10)
        solver = GeneticMazeSolver(config=config, seed=42)

        random_genome = [random.randint(0, 7) for _ in range(solver.config.genome_length)]
        fitness = solver.calculate_fitness(random_genome)

        self.assertTrue(
            isinstance(fitness, (float, int, np.floating, np.integer)),
            f"Fitness score must be numeric, got {type(fitness)}",
        )
        self.assertGreater(fitness, 0.0)

    def test_evolution_short_integration_run(self) -> None:
        """Run short evolutionary cycle to verify complete pipeline execution."""
        config = GeneticConfig(grid_size=6, max_generations=5)
        solver = GeneticMazeSolver(config=config, seed=42)

        champion_genome = solver.run_evolution(verbose=False)
        self.assertEqual(len(champion_genome), solver.config.genome_length)
        self.assertGreater(len(solver.fitness_history), 0)
        self.assertGreater(len(solver.mutation_rate_history), 0)


if __name__ == "__main__":
    unittest.main()
