"""Unit tests for Dijkstra distance calculation, vector flow field, and path solver."""

import unittest
import numpy as np

from maze_runner.generator import generate_maze_dfs
from maze_runner.oracle import (
    build_vector_field,
    compute_dijkstra_distances,
    solve_maze_vector_field,
)


class TestDijkstraOracle(unittest.TestCase):
    """Test suite for deterministic Wavefront Dijkstra and flow field solving."""

    def test_compute_dijkstra_distances_diagonal_and_propagation(self) -> None:
        """Verify 8-neighborhood metric (diagonal moves have unit step cost)."""
        # Open 3x3 grid
        grid = np.ones((3, 3), dtype=np.int8)
        goal = (0, 0)
        dist_map = compute_dijkstra_distances(grid, goal)

        # In 8-connectivity, diagonal neighbor (1, 1) is 1 step from (0, 0)
        self.assertEqual(dist_map[0, 0], 0)
        self.assertEqual(dist_map[1, 1], 1, "Diagonal step cost should be 1.")
        self.assertEqual(dist_map[2, 2], 2, "2 steps diagonally from (0, 0).")

    def test_compute_dijkstra_walls_marked(self) -> None:
        """Verify walls are correctly marked as -1 in distance map."""
        grid = np.array([[1, 0], [1, 1]], dtype=np.int8)
        dist_map = compute_dijkstra_distances(grid, goal=(0, 0))

        self.assertEqual(dist_map[0, 1], -1, "Wall cell must have value -1.")
        self.assertEqual(dist_map[0, 0], 0)
        self.assertEqual(dist_map[1, 0], 1)
        self.assertEqual(dist_map[1, 1], 1)

    def test_build_vector_field(self) -> None:
        """Verify vector flow field points towards lowest distance neighbor."""
        grid = np.ones((3, 3), dtype=np.int8)
        dist_map = compute_dijkstra_distances(grid, goal=(0, 0))
        dir_x, dir_y = build_vector_field(dist_map)

        # Cell (1, 1) should point toward (0, 0) -> dx = -1, dy = -1
        self.assertEqual(dir_x[1, 1], -1)
        self.assertEqual(dir_y[1, 1], -1)

    def test_solve_maze_vector_field(self) -> None:
        """Verify following vector flow field correctly reaches the goal."""
        grid = np.array([[1, 1], [0, 1]], dtype=np.int8)
        dist_map = compute_dijkstra_distances(grid, goal=(0, 0))
        dir_x, dir_y = build_vector_field(dist_map)

        path = solve_maze_vector_field(dir_x, dir_y, start=(1, 1))
        self.assertIn((0, 0), path, "Solved path should terminate at goal.")
        self.assertEqual(path[0], (1, 1), "Path should start at requested coordinate.")


if __name__ == "__main__":
    unittest.main()
