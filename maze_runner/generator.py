"""Maze environment generation using iterative Depth-First Search (DFS).

This module implements an optimized iterative DFS algorithm to construct perfect
binary mazes. It leverages 1D array flattening with 1-cell border padding and bytearray
storage to maximize memory efficiency and computational speed, completely avoiding
Python recursion limit issues.
"""

from collections import deque
import random
from typing import Deque, List, Optional, Set, Tuple
import numpy as np


# 8-directional neighborhood offsets (orthogonal and diagonal)
NEIGHBORS_8: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def generate_maze_dfs(size: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a square perfect maze of size N x N using an iterative DFS algorithm.

    A perfect maze contains no loops and guarantees exactly one path between any
    two reachable cells (spanning tree topology).

    Optimization details:
      1. 1D Array with Padding: Adds a 1-cell border around the grid to eliminate
         boundary checks during neighbor exploration.
      2. bytearray Storage: Uses 1 byte per cell (0 = unvisited/wall, 1 = passage,
         2 = border), reducing memory consumption to ~244 KB for N=500.
      3. Iterative Stack: Uses an explicit Python list as a stack to eliminate
         RecursionError for arbitrary grid sizes.

    Args:
        size: Width and height of the maze (N). Must be >= 2.
        seed: Optional random seed for deterministic generation.

    Returns:
        np.ndarray: A 2D binary numpy array of shape (size, size) where:
            - 0 represents a wall
            - 1 represents a walkable corridor
    """
    if size < 2:
        raise ValueError(f"Size must be >= 2, got {size}")

    if seed is not None:
        random.seed(seed)

    # 1D flattened width including a 1-cell boundary padding on all sides
    width = size + 2
    total_cells = width * width

    # Cell states: 0 = unvisited wall candidate, 1 = passage, 2 = forbidden border
    grid = bytearray(total_cells)

    # Pre-mark outer boundary borders (top, bottom, left, right)
    for i in range(width):
        grid[i] = 2                          # Top row
        grid[total_cells - 1 - i] = 2        # Bottom row
        grid[i * width] = 2                  # Left column
        grid[i * width + width - 1] = 2      # Right column

    # Precomputed 1D index offsets for all 8 surrounding neighbors:
    # [top-left, top, top-right, left, right, bottom-left, bottom, bottom-right]
    offsets_1d = [
        -width - 1, -width, -width + 1,
        -1,                  1,
        width - 1,  width,  width + 1,
    ]

    # Select random internal start cell
    start_x = random.randrange(1, size + 1)
    start_y = random.randrange(1, size + 1)
    start_idx = start_x * width + start_y

    grid[start_idx] = 1
    stack = [start_idx]

    # Local method caching for micro-optimization inside the hot loop
    push = stack.append
    pop = stack.pop

    while stack:
        curr = stack[-1]
        eligible_neighbors: List[int] = []

        # Inspect 8 surrounding cells
        for off in offsets_1d:
            neighbor = curr + off

            # If neighbor is an unvisited cell (0)
            if grid[neighbor] == 0:
                # Count visited neighbors of this candidate to prevent open rooms
                # and preserve single-corridor perfect maze topology
                visited_count = 0
                is_valid = True

                for off2 in offsets_1d:
                    if grid[neighbor + off2] == 1:
                        visited_count += 1
                        if visited_count > 1:
                            is_valid = False
                            break

                if is_valid:
                    eligible_neighbors.append(neighbor)

        if eligible_neighbors:
            next_cell = random.choice(eligible_neighbors)
            grid[next_cell] = 1
            push(next_cell)
        else:
            pop()

    # Reconstruct 2D numpy array and strip outer padding borders
    maze_2d = np.frombuffer(grid, dtype=np.int8).reshape((width, width))
    return maze_2d[1:-1, 1:-1].copy()


def find_nearest_free_cell(maze: np.ndarray, target: Tuple[int, int]) -> Tuple[int, int]:
    """Find the closest walkable cell (value 1) to a requested target coordinate.

    Uses Breadth-First Search (BFS) to identify the closest corridor cell if the
    initial target coordinate is placed inside a wall.

    Args:
        maze: 2D binary numpy array representing the maze.
        target: (row, col) coordinate to validate or snap to.

    Returns:
        Tuple[int, int]: Closest accessible (row, col) cell where maze[row, col] == 1.
    """
    rows, cols = maze.shape
    tx, ty = target

    # Bound target within maze dimensions
    tx = max(0, min(rows - 1, tx))
    ty = max(0, min(cols - 1, ty))

    if maze[tx, ty] != 0:
        return tx, ty

    visited: Set[Tuple[int, int]] = {(tx, ty)}
    queue: Deque[Tuple[int, int]] = deque([(tx, ty)])

    while queue:
        cx, cy = queue.popleft()

        if maze[cx, cy] != 0:
            return cx, cy

        for dx, dy in NEIGHBORS_8:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return target
