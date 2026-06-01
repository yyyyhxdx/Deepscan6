# Deepscan6

Iterative IPv6 active address discovery using **CEX (Continuous Extrapolation)** with optional BGP-guided exploration.

## Requirements

- Python 3.8+
- `masscan` (for scanning)
- `numpy` (optional, faster exploration)
- `psutil` (optional, memory stats)

## Quick Start

```bash
# Pure CEX mode
python main.py --seeds seeds.txt --rounds 10 --num 1000000

# CEX + BGP exploration hybrid (30% exploration)
python main.py --seeds seeds.txt --rounds 10 --num 1000000 \
  --explore-ratio 0.30 --bgp-file bgp_data.json
```

## Key Arguments

| Argument            | Default                  | Description                                     |
| ------------------- | ------------------------ | ----------------------------------------------- |
| `--seeds`           | *(required)*             | Initial active IPv6 seed file                   |
| `--rounds`          | 10                       | Number of scan rounds                           |
| `--num`             | 1,000,000                | Candidates generated per round                  |
| `--explore-ratio`   | 0.0                      | Fraction from BGP exploration (0–1)             |
| `--bgp-file`        | —                        | BGP prefix JSON (required if explore-ratio > 0) |
| `--alias-file`      | `./aliased-prefixes.txt` | Aliased prefix blocklist                        |
| `--no-alias-filter` | —                        | Skip alias filtering                            |
| `--workers`         | CPU-1                    | Parallel worker count                           |
| `--max-steps`       | 30                       | Max extrapolation steps                         |
| `--tolerance`       | 0.5                      | Pattern matching tolerance                      |
| `--state-dir`       | `./cex_state`            | State persistence directory                     |
| `--output-dir`      | `./cex_output`           | Candidate/hit output directory                  |
| `--reset`           | —                        | Clear state and restart from scratch            |

## Candidate Generation

CEX candidates are split into two pools and interleaved:

- **Interpolation** (~95%) — fill gaps between known IIDs within the same /64 prefix
- **Extrapolation** (~5%) — project beyond min/max IIDs using median step size

When `--explore-ratio` is set, BGP-guided exploration adds addresses from adjacent ASes, /48-level neighbors, and /64 neighbor prefixes.

## Output Files

```
cex_state/
  seeds.txt          # Current seed set
  all_active.txt     # Cumulative active addresses
  output_history.txt # All previously scanned candidates
  stats.json         # Per-round statistics

cex_output/
  candidates_r{N}.txt  # Candidates for round N
  hits_r{N}.txt        # Confirmed active addresses for round N
```

## 
