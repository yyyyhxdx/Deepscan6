# Deepscan6

Deepscan6 is an iterative IPv6 active address scanner. It uses Continuation-based Extrapolation (CEX) to generate candidate addresses from a set of known-active seed addresses, scans them with masscan, and feeds confirmed hits back into the model to guide the next round. An optional BGP-guided exploration layer can be enabled to discover addresses in adjacent address space.

## Requirements

- Python 3.8+
- `numpy` (optional, recommended for faster exploration sampling)
- `psutil` (optional, for accurate memory reporting)

```bash
pip install numpy psutil
```

## Configuration

Before running, open `main.py` and fill in your scanner host settings in the `Config.__init__` method:

```python
self.interface = "eno2"               # your outbound network interface, e.g. "eth0"
self.source_ip = "2001:db8::1"        # your scanner's IPv6 source address
self.router_mac = "00:00:00:00:00:00" # MAC address of your IPv6 gateway/router
```

You may also adjust `scan_rate` and `ping_rate` (packets per second) to suit your network capacity.

## Seed File

Prepare a plain-text file with one IPv6 address per line:

```
2001:db8::1
2001:db8::100
2001:db8::200
```

## Usage

### Pure CEX mode (default)

```bash
python main.py --seeds seeds.txt --rounds 10 --num 1000000
```

### CEX + BGP exploration hybrid

```bash
python main.py --seeds seeds.txt --rounds 10 --num 1000000 \
               --explore-ratio 0.30 --bgp-file bgp_data.json
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--seeds` | *(required)* | Seed address file |
| `--rounds` | `10` | Number of scan rounds |
| `--num` | `1000000` | Candidate addresses per round |
| `--explore-ratio` | `0.0` | Fraction of candidates from BGP exploration (0–1) |
| `--bgp-file` | — | BGP prefix JSON file (required when `--explore-ratio > 0`) |
| `--alias-file` | `./aliased-prefixes.txt` | Aliased prefix list to filter out |
| `--no-alias-filter` | — | Disable alias prefix filtering |
| `--max-steps` | `30` | Maximum extrapolation steps |
| `--tolerance` | `0.5` | Pattern matching tolerance |
| `--workers` | CPU count − 1 | Parallel worker processes |
| `--state-dir` | `./cex_state` | Directory for persisted state |
| `--output-dir` | `./cex_output` | Directory for output files |
| `--reset` | — | Clear existing state and restart from scratch |

## Output

Each round writes two files to `--output-dir`:

- `candidates_r<N>.txt` — addresses sent to masscan
- `hits_r<N>.txt` — addresses that responded

Cumulative state (seeds, history, all active addresses) is saved to `--state-dir` and reused across rounds.

## BGP Data Format

The BGP JSON file should follow this structure:

```json
{
  "prefixes": {
    "2001:db8::/32": { "as": 64496, "as_path": [64496, 64497] },
    ...
  }
}
```

