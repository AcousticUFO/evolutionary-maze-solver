"""Unit tests for maze generation and spatial utilities."""

import unittest
import numpy as np

from maze_runner.generator import find_nearest_free_cell, generate_maze_dfs


class TestMazeGenerator(unittest.TestCase):
    """Test suite for maze generation via iterative DFS and cell utilities."""

    def test_generate_maze_dimensions_and_binary_values(self) -> None:
        """Verify generated maze has correct shape and only contains binary values (0 or 1)."""
        size = 15
        maze = generate_maze_dfs(size)

        self.assertEqual(maze.shape, (size, size), "Maze dimensions mismatch.")
        unique_vals = set(np.unique(maze))
        self.assertTrue(
            unique_vals.issubset({0, 1}),
            f"Unexpected non-binary cell values detected: {unique_vals}",
        )

    def test_generate_maze_invalid_size(self) -> None:
        """Verify ValueError is raised when size is smaller than 2."""
        with self.assertRaises(ValueError):
            generate_maze_dfs(1)

    def test_generate_maze_seed_reproducibility(self) -> None:
        """Verify identical seeds produce identical maze layouts."""
        maze1 = generate_maze_dfs(size=12, seed=42)
        maze2 = generate_maze_dfs(size=12, seed=42)

        np.testing.assert_array_equal(
            maze1, maze2, "Mazes with identical seed are not equal."
        )

    def test_find_nearest_free_cell_already_free(self) -> None:
        """Verify free cell returns its own coordinate immediately."""
        grid = np.array([[1, 0], [0, 1]])
        coord = find_nearest_free_cell(grid, (0, 0))
        self.assertEqual(coord, (0, 0))

    def test_find_nearest_free_cell_on_wall(self) -> None:
        """Verify locating closest walkable neighbor when target is on a wall."""
        grid = np.array([[0, 0], [1, 0]])
        coord = find_nearest_free_cell(grid, (0, 0))
        self.assertEqual(coord, (1, 0))


if __name__ == "__main__":
    unittest.main()
