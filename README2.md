# IPv6 Seed Address Preprocessing

Ensemble-based outlier filtering for IPv6 seed sets, used as the preprocessing stage of DeepScan6. Combines a Variational Autoencoder (VAE) and a graph-theoretic density cut detector via soft voting to remove noisy seeds before candidate generation.

## Quick Start

```bash
python run_preprocessing.py -i seeds.txt -o output/
```

Output files are written to `output/`:
- `normal_seeds.txt` — filtered seed set passed to CEX
- `outlier_seeds.txt` — removed addresses
- `statistics.json` — summary counts and timing

## How It Works

Two detectors run concurrently and their scores are fused:

**VAE** — encodes each address to a 32-dim latent space (encoder: 96→64→48) and flags addresses exceeding the 95th-percentile reconstruction error.

**Graph density cut** — builds a multi-scale k-NN graph, marks nodes whose local density falls below 30% of the median density as outliers, using unanimous voting across scales.

**Soft voting** — fused score = 0.6 × score_ae + 0.4 × score_graph; addresses scoring above 0.5 are removed.

## File Overview

| File | Role |
|------|------|
| `run_preprocessing.py` | CLI entry point |
| `config.py` | All hyperparameters |
| `experiment.py` | Orchestrates concurrent AE + Graph execution and result saving |
| `optimized_autoencoder.py` | VAE detector with CUDA/AMP support |
| `optimized_graph.py` | Multi-scale graph density cut detector |

## Key Parameters (`config.py`)

| Parameter | Value | Note |
|-----------|-------|------|
| `contamination` | 0.05 | P95 threshold — fixed by paper |
| `density_threshold` | 0.3 | Fixed by paper |
| `ae_weight / graph_weight` | 0.6 / 0.4 | Fixed by paper |
| `ensemble threshold` | 0.5 | Fixed by paper |
| `multi_scale_k` | [30, 50, 85] | Tunable — larger k reduces false positives |
| `min_cluster_size` | 10 | Tunable |

## Requirements

```
torch, scikit-learn, scipy, numpy, joblib
```

GPU is used automatically when available; CPU fallback is supported.
