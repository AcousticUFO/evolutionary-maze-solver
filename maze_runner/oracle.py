"""Deterministic shortest-path Oracle using Wavefront Dijkstra BFS.

This module computes the exact corridor distance from every walkable cell in the maze
to the goal coordinate, constructing a ground-truth distance field and vector flow
field. This serves as an analytical benchmark and provides gradient guidance for the
genetic algorithm.
"""

from collections import deque
from typing import Deque, List, Set, Tuple
import numpy as np

from maze_runner.generator import NEIGHBORS_8, find_nearest_free_cell


def compute_dijkstra_distances(
    maze: np.ndarray, goal: Tuple[int, int]
) -> np.ndarray:
    """Compute the true shortest-path distance map from all maze cells to the goal.

    Uses a reverse wavefront BFS starting at the goal cell with `collections.deque`
    for O(1) pop operations. This produces the exact shortest distance through corridors
    for each walkable cell in O(N^2) time.

    Args:
        maze: 2D binary numpy array (0 = wall, 1 = passage).
        goal: Target (row, col) destination coordinates.

    Returns:
        np.ndarray: 2D int32 array of shape (rows, cols) where:
            - -1 represents walls
            - -2 represents unreachable corridors
            - >= 0 represents shortest step distance to the goal
    """
    rows, cols = maze.shape
    distance_map = np.full((rows, cols), -2, dtype=np.int32)
    distance_map[maze == 0] = -1

    gx, gy = find_nearest_free_cell(maze, goal)

    distance_map[gx, gy] = 0
    exploration_queue: Deque[Tuple[int, int]] = deque([(gx, gy)])

    while exploration_queue:
        curr_x, curr_y = exploration_queue.popleft()
        curr_dist = distance_map[curr_x, curr_y]

        for dx, dy in NEIGHBORS_8:
            nx, ny = curr_x + dx, curr_y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                # Cell is free and unvisited (-2)
                if distance_map[nx, ny] == -2:
                    distance_map[nx, ny] = curr_dist + 1
                    exploration_queue.append((nx, ny))

    return distance_map


def build_vector_field(
    distance_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive a directional vector flow field from the distance map.

    Each accessible cell is assigned a unit vector (dx, dy) pointing to the
    neighbor with the strictly lowest distance value towards the goal.

    Args:
        distance_map: 2D int32 array computed by compute_dijkstra_distances.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (dir_x, dir_y) arrays of dtype int8 storing
            the optimal movement step for every coordinate.
    """
    rows, cols = distance_map.shape
    dir_x = np.zeros((rows, cols), dtype=np.int8)
    dir_y = np.zeros((rows, cols), dtype=np.int8)

    # Process all reachable cells (distance >= 0)
    valid_cells = np.where(distance_map >= 0)

    for i, j in zip(valid_cells[0], valid_cells[1]):
        current_dist = distance_map[i, j]

        # Destination cell has distance 0, no direction needed
        if current_dist == 0:
            continue

        best_dist = current_dist
        best_dx, best_dy = 0, 0

        # Scan 8-directional neighborhood for steepest negative gradient
        for dx, dy in NEIGHBORS_8:
            ni, nj = i + dx, j + dy
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbor_val = distance_map[ni, nj]
                if 0 <= neighbor_val < best_dist:
                    best_dist = neighbor_val
                    best_dx, best_dy = dx, dy

        dir_x[i, j] = best_dx
        dir_y[i, j] = best_dy

    return dir_x, dir_y


def solve_maze_vector_field(
    dir_x: np.ndarray,
    dir_y: np.ndarray,
    start: Tuple[int, int],
    max_steps: int = 10000,
) -> List[Tuple[int, int]]:
    """Reconstruct the optimal trajectory by following the directional vector field.

    Args:
        dir_x: Row offset vector field.
        dir_y: Column offset vector field.
        start: Starting (row, col) coordinates.
        max_steps: Safeguard limit against circular loops.

    Returns:
        List[Tuple[int, int]]: Chronological list of (row, col) coordinates along
            the optimal path.
    """
    path: List[Tuple[int, int]] = [start]
    curr_x, curr_y = start
    rows, cols = dir_x.shape
    visited: Set[Tuple[int, int]] = {(curr_x, curr_y)}

    for _ in range(max_steps):
        move_x = int(dir_x[curr_x, curr_y])
        move_y = int(dir_y[curr_x, curr_y])

        # Goal reached or dead-end encountered
        if move_x == 0 and move_y == 0:
            break

        next_x, next_y = curr_x + move_x, curr_y + move_y

        # Out-of-bounds or loop detected
        if not (0 <= next_x < rows and 0 <= next_y < cols) or (next_x, next_y) in visited:
            break

        visited.add((next_x, next_y))
        path.append((next_x, next_y))
        curr_x, curr_y = next_x, next_y

    return path
