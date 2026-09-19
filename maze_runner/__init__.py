"""Maze Runner: Autonomous maze generation, deterministic oracle, and evolutionary AI."""

from maze_runner.config import GeneticConfig
from maze_runner.generator import find_nearest_free_cell, generate_maze_dfs
from maze_runner.genetic import GeneticMazeSolver
from maze_runner.oracle import (
    build_vector_field,
    compute_dijkstra_distances,
    solve_maze_vector_field,
)
from maze_runner.visualization import (
    plot_maze,
    plot_oracle_analysis,
    plot_simulation_dashboard,
)

__all__ = [
    "GeneticConfig",
    "generate_maze_dfs",
    "find_nearest_free_cell",
    "compute_dijkstra_distances",
    "build_vector_field",
    "solve_maze_vector_field",
    "GeneticMazeSolver",
    "plot_maze",
    "plot_oracle_analysis",
    "plot_simulation_dashboard",
]

__version__ = "1.0.0"
