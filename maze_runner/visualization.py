"""Visualization and plotting utilities for Maze Runner.

This module provides clean plotting routines decoupled from the core algorithms,
supporting headless server rendering, image exporting, and interactive displays.
"""

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_maze(
    maze: np.ndarray,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Visualize a 2D binary maze with optional start and goal markers.

    Args:
        maze: 2D binary numpy array (0 = wall, 1 = corridor).
        start: Optional start (row, col) coordinates.
        goal: Optional goal (row, col) coordinates.
        save_path: Optional file path to export the plot image.
        show: If True, displays the interactive plot window.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap="gray", interpolation="nearest")

    if start is not None:
        # Note: scatter expects (x=col, y=row)
        plt.scatter(
            [start[1]], [start[0]], c="lime", s=100, label="Start", edgecolors="black"
        )
    if goal is not None:
        plt.scatter(
            [goal[1]], [goal[0]], c="red", s=100, label="Goal", edgecolors="black"
        )

    plt.legend(loc="upper right")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_oracle_analysis(
    maze: np.ndarray,
    distance_map: np.ndarray,
    optimal_path: Optional[List[Tuple[int, int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Generate side-by-side comparison: Reconstructed path vs Dijkstra distance heatmap.

    Args:
        maze: 2D binary numpy array.
        distance_map: 2D array of shortest-path distances to goal.
        optimal_path: List of (row, col) coordinates along the solved path.
        start: Start coordinates.
        goal: Goal coordinates.
        save_path: Optional file path to export the figure.
        show: If True, displays the interactive plot window.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Left: Maze with reconstructed optimal path
    rgb_img = np.zeros((maze.shape[0], maze.shape[1], 3))
    rgb_img[maze == 1] = [1.0, 1.0, 1.0]

    if optimal_path:
        path_arr = np.array(optimal_path)
        if len(path_arr) > 0:
            rgb_img[path_arr[:, 0], path_arr[:, 1]] = [0.0, 0.5, 1.0]

    if start:
        rgb_img[start[0], start[1]] = [0.0, 1.0, 0.0]
    if goal:
        rgb_img[goal[0], goal[1]] = [1.0, 0.0, 0.0]

    ax1.imshow(rgb_img, interpolation="nearest")
    ax1.set_title("Optimal Path Trajectory")
    ax1.axis("off")

    # Right: Distance Heatmap with masked walls
    dist_vis = distance_map.astype(float)
    dist_vis[distance_map < 0] = np.nan

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color="black")

    im2 = ax2.imshow(dist_vis, cmap=cmap, interpolation="nearest")
    ax2.set_title("Wavefront Dijkstra Distance Heatmap")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_simulation_dashboard(
    solver: Any,
    best_genome: List[int],
    duration: float = 0.0,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Generate a 4-panel analytical dashboard summarizing the evolutionary run.

    Panels:
      1. Top-Left: Oracle Ground Truth (Blue) vs AI Agent Trajectory (Red).
      2. Top-Right: Key Metrics & Performance Summary Card.
      3. Bottom-Left: Fitness Convergence Curve across Generations.
      4. Bottom-Right: Adaptive Mutation Rate History showcasing Deluge Spikes.

    Args:
        solver: An initialized and executed GeneticMazeSolver instance.
        best_genome: Genome sequence of the best-performing agent.
        duration: Total execution wall-clock time in seconds.
        save_path: Optional file path to export the dashboard image.
        show: If True, displays the interactive plot window.
    """
    grid_size = solver.config.grid_size
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3)

    # 1. Trajectory comparison: Oracle vs AI
    axes[0, 0].imshow(solver.maze, cmap="gray", origin="upper")

    # Overlay ground-truth Oracle path in translucent blue
    oracle_mask = np.zeros((grid_size, grid_size, 4))
    for r, c in solver.reconstruct_optimal_path():
        oracle_mask[r, c] = [0.0, 0.8, 1.0, 0.6]
    axes[0, 0].imshow(oracle_mask)

    # Overlay AI agent path in red
    _, _, ai_trajectory = solver.simulate_genome(best_genome)
    ai_points = np.array(ai_trajectory)
    if len(ai_points) > 0:
        axes[0, 0].plot(
            ai_points[:, 1],
            ai_points[:, 0],
            color="red",
            linewidth=2,
            label="AI Agent",
        )

    axes[0, 0].set_title("Path Comparison: Oracle (Blue) vs AI Agent (Red)")
    axes[0, 0].axis("off")
    axes[0, 0].legend(loc="upper right")

    # 2. Summary text metrics card
    axes[0, 1].axis("off")
    success_str = (
        f"Gen {solver.success_generation}"
        if solver.success_generation >= 0
        else "Max Gens Reached"
    )
    summary_text = (
        f"Maze Runner AI Simulation Metrics\n"
        f"-----------------------------------------\n"
        f"Grid Dimension (N)      : {grid_size} x {grid_size}\n"
        f"Population Size         : {solver.config.population_size}\n"
        f"Genome Length           : {solver.config.genome_length}\n"
        f"Execution Time          : {duration:.3f} s\n"
        f"Resolution Outcome      : {success_str}\n"
        f"Closest Distance Record : {solver.record_distance:.0f} steps\n"
        f"Base Mutation Rate      : {solver.config.base_mutation_rate:.2f}\n"
        f"Deluge Mutation Rate    : {solver.config.deluge_mutation_rate:.2f}"
    )
    axes[0, 1].text(
        0.05,
        0.5,
        summary_text,
        fontsize=12,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da"),
    )

    # 3. Fitness convergence curve
    axes[1, 0].plot(
        solver.fitness_history,
        color="#1f77b4",
        linewidth=1.5,
        label="Distance to Goal",
    )
    axes[1, 0].set_title("Convergence (Distance to Goal)")
    axes[1, 0].set_xlabel("Generation")
    axes[1, 0].set_ylabel("Dijkstra Distance (steps)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend()

    # 4. Adaptive mutation rate history (Deluge tracking)
    axes[1, 1].step(
        range(len(solver.mutation_rate_history)),
        solver.mutation_rate_history,
        color="#ff7f0e",
        linewidth=1.5,
    )
    axes[1, 1].set_title("Adaptive Mutation Rate (Deluge Spikes)")
    axes[1, 1].set_xlabel("Generation")
    axes[1, 1].set_ylabel("Mutation Probability (tm)")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
