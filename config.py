"""
Configuration Management
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import json


@dataclass
class AutoencoderConfig:
    """Autoencoder configuration"""
    enabled: bool = True
    latent_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [96, 64, 48])
    learning_rate: float = 0.001
    batch_size: int = 1024
    epochs: int = 100
    contamination: float = 0.05     # P95 threshold — fixed by paper
    device: Optional[str] = None    # None for auto-detect, 'cuda' or 'cpu' for manual

    use_mixed_precision: bool = False
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class GraphConfig:
    """Graph-based detector configuration"""
    enabled: bool = True
    k_neighbors: int = 10
    density_threshold: float = 0.3  # Fixed by paper
    min_cluster_size: int = 10
    contamination: float = 0.05
    use_enhanced: bool = True

    # Larger k -> smoother density estimates -> fewer false-positive outliers.
    # Paper specifies "multi-scale" only; these values are not paper-fixed.
    multi_scale_k: List[int] = field(default_factory=lambda: [30, 50, 85])

    n_jobs: int = -1


@dataclass
class EnsembleConfig:
    """Ensemble learning configuration"""
    voting_strategy: Literal['hard', 'weighted', 'soft'] = 'soft'
    threshold: float = 0.5          # Fixed by paper

    ae_weight: float = 0.6          # Fixed by paper
    graph_weight: float = 0.4       # Fixed by paper

    def __post_init__(self):
        if self.ae_weight + self.graph_weight > 0:
            total = self.ae_weight + self.graph_weight
            self.ae_weight /= total
            self.graph_weight /= total


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    name: str = "default"
    description: str = ""

    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

    verbose: bool = True
    save_intermediate: bool = False

    def to_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'autoencoder': {
                'enabled': self.autoencoder.enabled,
                'latent_dim': self.autoencoder.latent_dim,
                'hidden_dims': self.autoencoder.hidden_dims,
                'epochs': self.autoencoder.epochs,
                'batch_size': self.autoencoder.batch_size,
            },
            'graph': {
                'enabled': self.graph.enabled,
                'k_neighbors': self.graph.k_neighbors,
                'density_threshold': self.graph.density_threshold,
                'use_enhanced': self.graph.use_enhanced,
                'multi_scale_k': self.graph.multi_scale_k,
            },
            'ensemble': {
                'voting_strategy': self.ensemble.voting_strategy,
                'ae_weight': self.ensemble.ae_weight,
                'graph_weight': self.ensemble.graph_weight,
            }
        }

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            name=data.get('name', 'loaded'),
            description=data.get('description', '')
        )


def get_config() -> ExperimentConfig:
    """Return the default baseline configuration."""
    return ExperimentConfig(
        name='baseline',
        description='Ensemble learning: Autoencoder 60% + Graph 40%',
        autoencoder=AutoencoderConfig(enabled=True, epochs=100),
        graph=GraphConfig(enabled=True, use_enhanced=True),
        ensemble=EnsembleConfig(ae_weight=0.6, graph_weight=0.4, voting_strategy='soft')
    )


def print_config_summary(config: ExperimentConfig):
    """Print a summary of the experiment configuration."""
    print(f"\n{'='*70}")
    print(f"Config: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*70}")

    print("\nAlgorithm status:")
    print(f"  Autoencoder: {'enabled' if config.autoencoder.enabled else 'disabled'}")
    if config.autoencoder.enabled:
        print(f"    - Latent dim: {config.autoencoder.latent_dim}")
        print(f"    - Hidden dims: {config.autoencoder.hidden_dims}")
        print(f"    - Epochs: {config.autoencoder.epochs}")
        print(f"    - Batch size: {config.autoencoder.batch_size}")

    print(f"  Graph detector: {'enabled' if config.graph.enabled else 'disabled'}")
    if config.graph.enabled:
        print(f"    - Enhanced: {'yes' if config.graph.use_enhanced else 'no'}")
        print(f"    - K neighbors: {config.graph.k_neighbors}")
        print(f"    - Multi-scale k: {config.graph.multi_scale_k}")
        print(f"    - Density threshold: {config.graph.density_threshold}")
        print(f"    - Min cluster size: {config.graph.min_cluster_size}")

    if config.autoencoder.enabled and config.graph.enabled:
        print(f"\nEnsemble strategy:")
        print(f"  Voting: {config.ensemble.voting_strategy}")
        print(f"  Autoencoder weight: {config.ensemble.ae_weight:.1%}")
        print(f"  Graph weight: {config.ensemble.graph_weight:.1%}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    config = get_config()
    print_config_summary(config)
    config.save('/tmp/test_config.json')
    print("Config saved to /tmp/test_config.json")
