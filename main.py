"""Command-line interface and entry point for Maze Runner simulation.

Executes the genetic algorithm maze solver, profiles performance, and produces
an analytical visual dashboard.
"""

import argparse
import sys
import time
from typing import Optional

from maze_runner.config import GeneticConfig
from maze_runner.genetic import GeneticMazeSolver
from maze_runner.visualization import plot_simulation_dashboard


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Maze Runner: Evolutionary Genetic Algorithm Maze Solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=50,
        help="Side length of the square maze (N x N)",
    )
    parser.add_argument(
        "--selection-rate",
        type=float,
        default=0.15,
        help="Proportion of top-ranking individuals selected for breeding",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.15,
        help="Base mutation probability during crossover",
    )
    parser.add_argument(
        "-p",
        "--population",
        type=int,
        default=None,
        help="Population size (defaults to 10 * size if omitted)",
    )
    parser.add_argument(
        "-l",
        "--genome-length",
        type=int,
        default=None,
        help="Length of movement genome sequence (defaults to 25 * size if omitted)",
    )
    parser.add_argument(
        "-g",
        "--max-generations",
        type=int,
        default=None,
        help="Maximum generation limit before termination",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible maze and initial population generation",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable generating the Matplotlib visual dashboard",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="File path to save the resulting dashboard image (e.g. dashboard.png)",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display interactive plot window after simulation completion",
    )

    return parser.parse_args()


def run_simulation(args: argparse.Namespace) -> int:
    """Configure and execute the Maze Runner simulation."""
    config = GeneticConfig(
        grid_size=args.size,
        population_size=args.population,
        genome_length=args.genome_length,
        max_generations=args.max_generations,
        selection_rate=args.selection_rate,
        base_mutation_rate=args.mutation_rate,
    )

    print("=" * 60)
    print("           MAZE RUNNER - EVOLUTIONARY AI SOLVER           ")
    print("=" * 60)
    print(f"Grid Size (N)     : {config.grid_size} x {config.grid_size}")
    print(f"Population Size   : {config.population_size}")
    print(f"Genome Length (L) : {config.genome_length}")
    print(f"Max Generations   : {config.max_generations}")
    print(f"Selection Rate    : {config.selection_rate:.2f}")
    print(f"Base Mutation Rate: {config.base_mutation_rate:.2f}")
    if args.seed is not None:
        print(f"Random Seed       : {args.seed}")
    print("-" * 60)

    # Initialize environment, oracle, and population
    init_start = time.perf_counter()
    solver = GeneticMazeSolver(config=config, seed=args.seed)
    init_duration = time.perf_counter() - init_start
    print(f"Environment & Dijkstra Oracle initialized in {init_duration:.4f}s\n")

    # Execute evolutionary algorithm
    start_time = time.perf_counter()
    champion_genome = solver.run_evolution(verbose=True, log_interval=50)
    evolution_duration = time.perf_counter() - start_time

    total_time = init_duration + evolution_duration

    print("\n" + "=" * 60)
    print("                     SIMULATION SUMMARY                    ")
    print("=" * 60)
    print(f"Total Execution Time : {total_time:.4f} seconds")
    print(f"Evolution Loop Time  : {evolution_duration:.4f} seconds")
    print(f"Outcome              : {'GOAL REACHED' if solver.record_distance == 0 else 'TERMINATED'}")
    print(f"Winning Generation   : {solver.success_generation if solver.success_generation >= 0 else 'N/A'}")
    print(f"Closest Distance     : {solver.record_distance:.0f} steps")
    print("=" * 60)

    if not args.no_plot:
        output_image = args.save_plot or f"dashboard_N{config.grid_size}.png"
        print(f"\nGenerating analytical dashboard -> {output_image}...")
        plot_simulation_dashboard(
            solver=solver,
            best_genome=champion_genome,
            duration=evolution_duration,
            save_path=output_image,
            show=args.show_plot,
        )
        print("Dashboard generated successfully.")

    return 0


def main() -> None:
    """Script entrypoint."""
    args = parse_arguments()
    sys.exit(run_simulation(args))


if __name__ == "__main__":
    main()