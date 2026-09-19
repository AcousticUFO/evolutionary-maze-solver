"""Configuration parameters for Maze Runner simulation and genetic algorithm."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GeneticConfig:
    """Hyperparameters for the Genetic Algorithm Maze Solver.

    Attributes:
        grid_size: Side length of the square maze (N x N).
        population_size: Number of individuals in the population. If None, scales as 10 * grid_size.
        genome_length: Total number of movement genes per individual. If None, scales as 25 * grid_size.
        max_generations: Maximum generations allowed. If None, dynamically computed.
        selection_rate: Proportion of top-performing individuals selected for breeding (elitism pool).
        base_mutation_rate: Standard mutation probability applied during crossover.
        deluge_mutation_rate: Elevated mutation probability triggered during stagnation or genetic collapse.
        pheromone_stagnation_limit: Consecutive stagnant generations before placing a virtual pheromone wall.
        deluge_stagnation_limit: Consecutive stagnant generations before triggering a deluge event.
        deluge_diversity_limit: Minimum number of unique landing coordinates required to prevent deluge.
        exploration_bonus_weight: Multiplier rewarding agents that explore further along the corridors.
        immobility_penalty: Heavy fitness penalty applied to agents with <= 1 step.
        unreachable_penalty: Penalty score assigned if the agent ends in an unreachable cell.
    """

    grid_size: int = 50
    population_size: Optional[int] = None
    genome_length: Optional[int] = None
    max_generations: Optional[int] = None
    selection_rate: float = 0.15
    base_mutation_rate: float = 0.15
    deluge_mutation_rate: float = 0.70
    pheromone_stagnation_limit: int = 25
    deluge_stagnation_limit: int = 15
    deluge_diversity_limit: int = 5
    exploration_bonus_weight: float = 1.2
    immobility_penalty: float = 5000.0
    unreachable_penalty: float = 5000.0

    def __post_init__(self) -> None:
        """Resolve dynamic default values if not explicitly provided."""
        if self.population_size is None:
            self.population_size = 10 * self.grid_size

        if self.genome_length is None:
            # Scaled to quadratic corridor length studied in empirical analysis
            self.genome_length = 25 * self.grid_size

        if self.max_generations is None:
            # Dynamic generation threshold based on polynomial difficulty scaling
            self.max_generations = int((0.08 * (self.grid_size**2) + 65 * self.grid_size + 500) * 1.5)
