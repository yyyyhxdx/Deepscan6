#!/usr/bin/env python3
"""
IPv6 Seed Preprocessing CLI Tool
"""

import argparse
import json
import os
import sys
import time
from typing import List

from config import get_config, print_config_summary
from experiment import Experiment


def load_addresses(filepath: str) -> List[str]:
    """Load IPv6 addresses from file, skipping blank lines and comments."""
    addresses = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                addresses.append(line)
    return addresses


def run_preprocessing(input_file: str, output_dir: str, verbose: bool = True):
    """Run preprocessing with the default baseline configuration."""

    prog_start = time.perf_counter()

    print(f"\n{'=' * 70}")
    print("IPv6 Seed Address Preprocessing")
    print(f"{'=' * 70}\n")

    # --- Load ---
    t0 = time.perf_counter()
    print(f"Loading addresses: {input_file}")
    addresses = load_addresses(input_file)
    load_time = time.perf_counter() - t0
    print(f"  Loaded {len(addresses)} addresses  [{load_time:.3f}s]\n")

    config = get_config()
    if verbose:
        print_config_summary(config)

    # --- Run ---
    experiment = Experiment(output_dir=output_dir)
    result = experiment.run_single_experiment(config, addresses, verbose=verbose)

    if not result["success"]:
        print(f"\nProcessing failed: {result.get('error')}")
        return 1

    # --- Write top-level summary files (single write, no duplication with _save_result) ---
    import numpy as np

    t0 = time.perf_counter()
    predictions = np.array(result["predictions"])
    normal_addrs = [a for a, p in zip(addresses, predictions) if p == 1]
    outlier_addrs = [a for a, p in zip(addresses, predictions) if p == -1]

    os.makedirs(output_dir, exist_ok=True)
    normal_file = os.path.join(output_dir, "normal_seeds.txt")
    outlier_file = os.path.join(output_dir, "outlier_seeds.txt")
    stats_file = os.path.join(output_dir, "statistics.json")

    with open(normal_file, "w") as f:
        f.writelines(a + "\n" for a in normal_addrs)

    with open(outlier_file, "w") as f:
        f.writelines(a + "\n" for a in outlier_addrs)

    with open(stats_file, "w") as f:
        json.dump(
            {
                "config": config.name,
                "total": len(addresses),
                "normal": len(normal_addrs),
                "outliers": len(outlier_addrs),
                "outlier_ratio": result["outlier_ratio"],
                "execution_time": result["execution_time"],
            },
            f,
            indent=2,
        )

    write_time = time.perf_counter() - t0
    total_time = time.perf_counter() - prog_start

    # --- Final report ---
    print(f"\n{'=' * 70}")
    print("Processing complete")
    print(f"{'=' * 70}")
    print(f"  Normal seeds : {normal_file}")
    print(f"  Outlier seeds: {outlier_file}")
    print(f"  Statistics   : {stats_file}")
    print(f"{'─' * 70}")
    print(f"  Load time    : {load_time:.3f}s")
    print(f"  Process time : {result['execution_time']:.3f}s")
    print(f"  Write time   : {write_time:.3f}s")
    print(f"  Total time   : {total_time:.3f}s")
    print(f"{'=' * 70}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="IPv6 Seed Address Preprocessing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  Basic preprocessing:
    python run_preprocessing.py -i seeds.txt -o output/

  Verbose output (with per-stage timing):
    python run_preprocessing.py -i seeds.txt -o output/ -v

  Quiet mode:
    python run_preprocessing.py -i seeds.txt -o output/ -q
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-stage timing and detailed output",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()
    verbose = args.verbose and not args.quiet

    try:
        return run_preprocessing(
            input_file=args.input,
            output_dir=args.output,
            verbose=verbose,
        )
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
