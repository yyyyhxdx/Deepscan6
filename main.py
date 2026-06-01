# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import ipaddress
import os
import sys
import random
import json
import time
import subprocess
import shutil
import hashlib
import pickle
import resource
import bisect

# Optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not installed, exploration layer will use fallback implementation (slower)")

from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict, Optional, Iterable
from multiprocessing import Pool, cpu_count
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field


# ============================================================
# Module 0: Timing and Resource Statistics
# ============================================================

class Timer:
    """Timer utility"""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time

    def __str__(self):
        return f"{self.elapsed:.2f}s"


class ResourceStats:
    """Resource statistics"""
    def __init__(self):
        self.start_time = time.time()
        self.phase_times: Dict[str, float] = {}
        self.round_stats: List[Dict] = []
        self.peak_memory_mb = 0
        if HAS_PSUTIL:
            self.process = psutil.Process()
        else:
            self.process = None

    def update_memory(self):
        try:
            if HAS_PSUTIL and self.process:
                mem_info = self.process.memory_info()
                current_mb = mem_info.rss / 1024 / 1024
            else:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                current_mb = usage.ru_maxrss / 1024
            self.peak_memory_mb = max(self.peak_memory_mb, current_mb)
        except:
            pass

    def get_current_memory_mb(self) -> float:
        try:
            if HAS_PSUTIL and self.process:
                return self.process.memory_info().rss / 1024 / 1024
            else:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                return usage.ru_maxrss / 1024
        except:
            return 0

    def total_elapsed(self) -> float:
        return time.time() - self.start_time

    def format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            return f"{seconds/3600:.1f}h"


resource_stats = ResourceStats()


# ============================================================
# Module 1: Configuration Management
# ============================================================

class Config:
    """Global configuration"""
    def __init__(self):
        # Directory configuration
        self.state_dir = "./cex_state"
        self.output_dir = "./cex_output"
        self.cache_dir = "./.alias_cache"

        # Parallel configuration
        self.num_workers = max(1, cpu_count() - 1)

        # CEX configuration
        self.max_steps = 30
        self.tolerance = 0.5

        # Alias filter configuration
        self.alias_file = "./aliased-prefixes.txt"
        self.enable_alias_filter = True

        # Exploration configuration
        self.explore_ratio = 0.0  # Default 0, pure CEX mode
        self.bgp_file = None

        # Scan configuration
        self.scan_port = "80,443"
        self.scan_rate = 100000
        self.ping_rate = 100000
        self.interface = ""
        self.source_ip = ""
        self.router_mac = ""

    def setup_dirs(self):
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)


# ============================================================
# Module 2: State Management
# ============================================================

class StateManager:
    """State persistence manager (optimized)"""

    def __init__(self, config: Config):
        self.config = config
        self.seeds_file = os.path.join(config.state_dir, "seeds.txt")
        self.history_file = os.path.join(config.state_dir, "output_history.txt")
        self.all_active_file = os.path.join(config.state_dir, "all_active.txt")
        self.stats_file = os.path.join(config.state_dir, "stats.json")

        self._seeds_cache: Optional[Set[str]] = None
        self._history_cache: Optional[Set[str]] = None
        self._all_active_cache: Optional[Set[str]] = None

    def load_seeds(self) -> Set[str]:
        if self._seeds_cache is not None:
            return self._seeds_cache
        if not os.path.exists(self.seeds_file):
            self._seeds_cache = set()
            return self._seeds_cache
        with open(self.seeds_file, 'r', buffering=1024*1024) as f:
            self._seeds_cache = {line.strip() for line in f if line.strip()}
        return self._seeds_cache

    def save_seeds(self, seeds: Set[str]):
        self._seeds_cache = seeds
        with open(self.seeds_file, 'w', buffering=1024*1024) as f:
            f.write('\n'.join(seeds))
            f.write('\n')

    def load_history(self) -> Set[str]:
        if self._history_cache is not None:
            return self._history_cache
        if not os.path.exists(self.history_file):
            self._history_cache = set()
            return self._history_cache
        with open(self.history_file, 'r', buffering=1024*1024) as f:
            self._history_cache = {line.strip() for line in f if line.strip()}
        return self._history_cache

    def append_history(self, addrs: Set[str]):
        if self._history_cache is not None:
            self._history_cache.update(addrs)
        with open(self.history_file, 'a', buffering=1024*1024) as f:
            f.write('\n'.join(addrs))
            f.write('\n')

    def load_all_active(self) -> Set[str]:
        if self._all_active_cache is not None:
            return self._all_active_cache
        if not os.path.exists(self.all_active_file):
            self._all_active_cache = set()
            return self._all_active_cache
        with open(self.all_active_file, 'r', buffering=1024*1024) as f:
            self._all_active_cache = {line.strip() for line in f if line.strip()}
        return self._all_active_cache

    def save_all_active(self, addrs: Set[str]):
        self._all_active_cache = addrs
        with open(self.all_active_file, 'w', buffering=1024*1024) as f:
            f.write('\n'.join(addrs))
            f.write('\n')

    def load_stats(self) -> Dict:
        if not os.path.exists(self.stats_file):
            return {'rounds': [], 'total_scanned': 0, 'total_hits': 0}
        with open(self.stats_file, 'r') as f:
            return json.load(f)

    def save_stats(self, stats: Dict):
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

    def is_first_run(self) -> bool:
        return not os.path.exists(self.seeds_file)

    def clear_cache(self):
        self._seeds_cache = None
        self._history_cache = None
        self._all_active_cache = None


# ============================================================
# Module 3: BGP Prefix Database (from V18)
# ============================================================

class BGPPrefixDatabase:
    """BGP prefix database"""

    def __init__(self):
        self.prefixes: Dict[str, Dict] = {}
        self.as_graph: Dict[str, List[int]] = {}
        self.as_to_prefixes: Dict[int, List[str]] = defaultdict(list)
        self.ip_ranges: List[Dict] = []

    def load_from_file(self, filepath: str) -> bool:
        """Load BGP data from file"""
        if not filepath or not os.path.exists(filepath):
            print(f"  BGP data file not found: {filepath}")
            return False

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if 'prefixes' in data:
                self.prefixes = data.get('prefixes', {})
                self.as_graph = data.get('as_graph', {})
            else:
                self.prefixes = data

            self.as_to_prefixes.clear()
            for prefix, info in self.prefixes.items():
                as_num = info.get('as')
                if as_num is not None:
                    self.as_to_prefixes[as_num].append(prefix)

            if not self.as_graph:
                self._build_as_graph()

            self._build_ip_index()

            print(f"  BGP data loaded: {len(self.prefixes):,} prefixes, "
                  f"{len(self.as_to_prefixes):,} ASes")

            return True

        except Exception as e:
            print(f"  Failed to load BGP data: {e}")
            return False

    def _build_ip_index(self):
        """Build IP range index for fast lookup"""
        self.ip_ranges = []

        for prefix, info in self.prefixes.items():
            try:
                network = ipaddress.IPv6Network(prefix, strict=False)
                start_int = int(network.network_address)
                end_int = int(network.broadcast_address)
                as_num = info.get('as')

                if as_num is not None:
                    self.ip_ranges.append({
                        'start': start_int,
                        'end': end_int,
                        'as': as_num,
                        'prefix': prefix
                    })
            except:
                continue

        self.ip_ranges.sort(key=lambda x: x['start'])

    def _build_as_graph(self):
        """Build AS adjacency graph from AS paths"""
        as_neighbors = defaultdict(set)

        for prefix, info in self.prefixes.items():
            as_path = info.get('as_path', [])
            if len(as_path) >= 2:
                for i in range(len(as_path) - 1):
                    as1 = as_path[i]
                    as2 = as_path[i + 1]
                    as_neighbors[str(as1)].add(as2)
                    as_neighbors[str(as2)].add(as1)

        self.as_graph = {k: list(v) for k, v in as_neighbors.items()}

    def get_as_for_address(self, addr: str) -> Optional[int]:
        """Lookup AS number for an IPv6 address (binary search)"""
        try:
            addr_int = int(ipaddress.IPv6Address(addr))
        except:
            return None

        left, right = 0, len(self.ip_ranges) - 1
        result_idx = -1

        while left <= right:
            mid = (left + right) // 2
            if self.ip_ranges[mid]['start'] <= addr_int:
                result_idx = mid
                left = mid + 1
            else:
                right = mid - 1

        if result_idx >= 0:
            entry = self.ip_ranges[result_idx]
            if entry['start'] <= addr_int <= entry['end']:
                return entry['as']

        return None

    def get_as_prefixes(self, as_num: int) -> List[str]:
        """Get all prefixes belonging to an AS"""
        return self.as_to_prefixes.get(as_num, [])

    def get_as_neighbors(self, as_num: int) -> List[int]:
        """Get neighboring ASes of an AS"""
        return self.as_graph.get(str(as_num), [])

    def get_all_prefixes(self) -> List[str]:
        """Get all prefixes"""
        return list(self.prefixes.keys())


# ============================================================
# Module 4: Utility Functions
# ============================================================

def ipv6_to_cont(addr: str) -> str:
    """Convert IPv6 address to 32-char continuous hex format"""
    try:
        return ipaddress.IPv6Address(addr).exploded.replace(":", "")
    except:
        return addr

def cont_to_ipv6(cont: str) -> str:
    """Convert 32-char continuous hex format to IPv6 address"""
    if len(cont) == 32 and ':' not in cont:
        return ":".join(cont[i:i+4] for i in range(0, 32, 4))
    return cont

def read_addrs(filename: str) -> List[str]:
    """Read address file"""
    if not os.path.exists(filename):
        return []
    out = []
    with open(filename, 'r') as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'):
                try:
                    out.append(ipaddress.IPv6Address(s).exploded)
                except:
                    continue
    return out


# ============================================================
# Module 5: Alias Address Filter
# ============================================================

@dataclass
class AliasFilterStats:
    """Alias filter statistics"""
    total: int = 0
    aliased: int = 0
    remaining: int = 0

    @property
    def aliased_rate(self) -> float:
        return self.aliased / self.total if self.total > 0 else 0


class AliasFilter:
    """Alias address filter"""

    _MASK_CACHE: Dict[int, int] = {}

    def __init__(self, config: Config):
        self.config = config
        self.prefix_lookup: Dict[int, frozenset] = {}
        self.enabled = False
        self._init_mask_cache()

    def _init_mask_cache(self):
        if not AliasFilter._MASK_CACHE:
            for prefixlen in range(0, 129):
                if prefixlen == 0:
                    AliasFilter._MASK_CACHE[prefixlen] = 0
                elif prefixlen == 128:
                    AliasFilter._MASK_CACHE[prefixlen] = (1 << 128) - 1
                else:
                    AliasFilter._MASK_CACHE[prefixlen] = ((1 << 128) - 1) ^ ((1 << (128 - prefixlen)) - 1)

    def _get_file_hash(self, filepath: str) -> str:
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
        except:
            return ""
        return hash_md5.hexdigest()

    def _get_cache_path(self, prefix_file: str) -> str:
        abs_path = os.path.abspath(prefix_file)
        cache_name = hashlib.md5(abs_path.encode()).hexdigest() + ".pkl"
        return os.path.join(self.config.cache_dir, cache_name)

    def load(self, verbose: bool = False) -> bool:
        if not self.config.enable_alias_filter:
            if verbose:
                print("  Alias filter: disabled")
            return False

        if not os.path.exists(self.config.alias_file):
            if verbose:
                print(f"  Alias filter: file not found ({self.config.alias_file})")
            return False

        if verbose:
            print(f"  Loading alias prefixes: {self.config.alias_file}")

        file_hash = self._get_file_hash(self.config.alias_file)
        cache_path = self._get_cache_path(self.config.alias_file)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if cached.get('hash') == file_hash:
                    self.prefix_lookup = cached['lookup']
                    self.enabled = True
                    total = sum(len(v) for v in self.prefix_lookup.values())
                    if verbose:
                        print(f"    Loaded from cache: {total:,} prefixes")
                    return True
            except:
                pass

        start = time.time()
        self.prefix_lookup = self._load_prefixes_fast(verbose)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'hash': file_hash,
                    'lookup': self.prefix_lookup
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f"    Cache saved")
        except:
            pass

        self.enabled = True
        elapsed = time.time() - start

        if verbose:
            total_prefixes = sum(len(v) for v in self.prefix_lookup.values())
            print(f"    Load complete: {total_prefixes:,} prefixes, {elapsed:.2f}s")

        return True

    def _load_prefixes_fast(self, verbose: bool = False) -> Dict[int, frozenset]:
        lookup: Dict[int, Set[int]] = defaultdict(set)
        line_count = 0
        valid_count = 0

        with open(self.config.alias_file, 'r') as f:
            for line in f:
                line_count += 1
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    if '/' in line:
                        addr_part, prefix_len = line.rsplit('/', 1)
                        prefix_len = int(prefix_len)
                    else:
                        addr_part = line
                        prefix_len = 64

                    addr_int = self._parse_ipv6_fast(addr_part)
                    if addr_int is None:
                        continue

                    if prefix_len < 128:
                        mask = self._MASK_CACHE.get(prefix_len)
                        if mask is None:
                            mask = ((1 << 128) - 1) ^ ((1 << (128 - prefix_len)) - 1)
                        addr_int = addr_int & mask

                    lookup[prefix_len].add(addr_int)
                    valid_count += 1

                    if verbose and valid_count % 100000 == 0:
                        print(f"    Loaded: {valid_count:,}...", end='\r')

                except Exception:
                    continue

        if verbose:
            print(f"    Parsing complete: {valid_count:,} / {line_count:,}    ")

        return {plen: frozenset(addrs) for plen, addrs in lookup.items()}

    def _parse_ipv6_fast(self, addr: str) -> Optional[int]:
        try:
            addr = addr.strip()

            if '::' in addr:
                parts = addr.split('::')
                if len(parts) != 2:
                    return None

                left = parts[0].split(':') if parts[0] else []
                right = parts[1].split(':') if parts[1] else []

                missing = 8 - len(left) - len(right)
                if missing < 0:
                    return None

                groups = left + ['0'] * missing + right
            else:
                groups = addr.split(':')

            if len(groups) != 8:
                return None

            result = 0
            for g in groups:
                if not g:
                    g = '0'
                result = (result << 16) | int(g, 16)

            return result

        except Exception:
            return None

    def is_aliased(self, addr: str) -> bool:
        if not self.enabled:
            return False

        try:
            addr_obj = ipaddress.IPv6Address(addr)
            addr_int = int(addr_obj)
        except:
            return False

        for prefixlen, network_addrs in self.prefix_lookup.items():
            mask = AliasFilter._MASK_CACHE[prefixlen]
            network_int = addr_int & mask
            if network_int in network_addrs:
                return True

        return False

    def filter_addresses(self, addrs: Iterable[str], verbose: bool = False) -> Tuple[List[str], AliasFilterStats]:
        stats = AliasFilterStats()

        if not self.enabled:
            addr_list = list(addrs)
            stats.total = len(addr_list)
            stats.remaining = len(addr_list)
            return addr_list, stats

        remaining = []

        for addr in addrs:
            stats.total += 1
            if self.is_aliased(addr):
                stats.aliased += 1
            else:
                stats.remaining += 1
                remaining.append(addr)

        if verbose and stats.total > 0:
            print(f"    Alias filter: {stats.total:,} -> {stats.remaining:,} "
                  f"(filtered {stats.aliased:,}, {stats.aliased_rate:.1%})")

        return remaining, stats

    def filter_file(self, input_file: str, output_file: str,
                    verbose: bool = False) -> AliasFilterStats:
        """Filter aliased addresses from file"""
        stats = AliasFilterStats()

        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            for line in fin:
                addr = line.strip()
                if not addr:
                    continue

                stats.total += 1
                if self.enabled and self.is_aliased(addr):
                    stats.aliased += 1
                else:
                    stats.remaining += 1
                    fout.write(addr + '\n')

        if verbose:
            print(f"    File alias filter: {stats.total:,} -> {stats.remaining:,}")

        return stats


# ============================================================
# Module 6: CEX Extrapolation Worker
# ============================================================

def _process_prefix_chunk(args: Tuple) -> List[Tuple[str, int, str]]:
    """Process a batch of prefixes for extrapolation (multiprocessing worker)"""
    prefix_data_chunk, existing_set, max_steps = args

    results = []

    for p64, data in prefix_data_chunk:
        iids = data['iids']
        step = data['step']
        max_iid = data['max']
        min_iid = data['min']

        if len(iids) < 2 or step <= 0:
            continue

        # Interpolation
        for i in range(len(iids) - 1):
            lo, hi = iids[i], iids[i + 1]
            gap = hi - lo
            if gap <= 1:
                continue

            num_samples = min(50, gap - 1)
            sampled = set()
            attempts = 0
            while len(sampled) < num_samples and attempts < num_samples * 3:
                new_iid = random.randint(lo + 1, hi - 1)
                if new_iid not in sampled:
                    sampled.add(new_iid)
                    addr = p64 + f"{new_iid:016x}"
                    if addr not in existing_set:
                        results.append((addr, 0, 'interpolate'))
                attempts += 1

        # Forward extrapolation
        for s in range(1, max_steps + 1):
            base_iid = max_iid + step * s
            for off in [0, -1, 1, -2, 2, -3, 3, -5, 5, -8, 8]:
                new_iid = base_iid + off
                if 0 <= new_iid < 2**64:
                    addr = p64 + f"{new_iid:016x}"
                    if addr not in existing_set:
                        results.append((addr, s, 'forward'))

        # Backward extrapolation
        for s in range(1, max_steps + 1):
            base_iid = min_iid - step * s
            for off in [0, -1, 1, -2, 2, -3, 3, -5, 5, -8, 8]:
                new_iid = base_iid + off
                if 0 <= new_iid < 2**64:
                    addr = p64 + f"{new_iid:016x}"
                    if addr not in existing_set:
                        results.append((addr, s, 'backward'))

    return results


# ============================================================
# Module 7: Hybrid Generator (CEX + Exploration) - V27 Fixed
# ============================================================

class HybridGenerator:
    

    def __init__(self, config: Config):
        self.config = config
        self.prefix_data: Dict[str, Dict] = {}
        self.prefixes: List[str] = []
        self.all_active_set: Set[str] = set()
        self.seed_as_set: Set[int] = set()

        # BGP database (optional)
        self.bgp_db: Optional[BGPPrefixDatabase] = None
        if config.explore_ratio > 0:
            self._load_bgp_data()

    def _load_bgp_data(self):
        """Load BGP data"""
        if not self.config.bgp_file:
            print("    Warning: No BGP data file provided, exploration layer will be disabled")
            return

        self.bgp_db = BGPPrefixDatabase()
        if not self.bgp_db.load_from_file(self.config.bgp_file):
            self.bgp_db = None
            print("    Warning: BGP data load failed, exploration layer will be disabled")
        else:
            print(f"    BGP data loaded, exploration layer enabled")

    def fit(self, seeds: List[str]):
        """Train CEX"""
        print(f"  Training hybrid generator: {len(seeds):,} seeds...")

        prefix_iids = defaultdict(list)
        self.all_active_set = set()
        self.seed_as_set = set()

        for s in seeds:
            cont = s if len(s) == 32 and ':' not in s else ipv6_to_cont(s)
            p64 = cont[:16]
            iid = int(cont[16:], 16)
            prefix_iids[p64].append(iid)

            addr_ipv6 = cont_to_ipv6(cont)
            self.all_active_set.add(addr_ipv6)

            # Query AS (if exploration is enabled)
            if self.bgp_db:
                as_num = self.bgp_db.get_as_for_address(addr_ipv6)
                if as_num is not None:
                    self.seed_as_set.add(as_num)

        for p64, iids in prefix_iids.items():
            iids = sorted(set(iids))
            if len(iids) >= 2:
                diffs = [iids[i+1] - iids[i] for i in range(len(iids)-1)]
                median_step = sorted(diffs)[len(diffs)//2]
            else:
                median_step = 1000

            self.prefix_data[p64] = {
                'iids': iids,
                'step': max(1, median_step),
                'min': iids[0],
                'max': iids[-1]
            }

        self.prefixes = list(self.prefix_data.keys())

        # Build standard IPv6 prefix list (for exploration layer filtering)
        self.prefixes_ipv6 = []
        for p64_hex in self.prefixes:
            try:
                full_hex = p64_hex + "0" * 16
                ipv6_full = cont_to_ipv6(full_hex)
                addr_obj = ipaddress.IPv6Address(ipv6_full)
                network = ipaddress.IPv6Network(f"{addr_obj}/64", strict=False)
                self.prefixes_ipv6.append(str(network.network_address))
            except:
                continue

        total_iids = sum(len(d['iids']) for d in self.prefix_data.values())
        print(f"    Prefixes: {len(self.prefixes):,}, IIDs: {total_iids:,}")

    def generate(self, n: int, existing: Set[str]) -> List[str]:
        """
        Generate hybrid candidates.

        Bug1 Fix: pass n_cex directly to _generate_cex_with_priority
        instead of n_cex * 2. The internal method no longer halves with //2.
        """
        print(f"  Hybrid generation (workers={self.config.num_workers})...")

        # Calculate quotas
        if self.config.explore_ratio <= 0 or self.bgp_db is None:
            n_cex = n
            n_explore = 0
            print(f"    Mode: pure CEX ({n_cex:,})")
        else:
            n_explore = int(n * self.config.explore_ratio)
            n_cex = n - n_explore
            print(f"    Quota: CEX={n_cex:,} ({(1-self.config.explore_ratio)*100:.0f}%), "
                  f"explore={n_explore:,} ({self.config.explore_ratio*100:.0f}%)")

        cex_candidates = self._generate_cex_with_priority(n_cex, existing)

        # Generate exploration candidates
        explore_candidates = []
        if n_explore > 0 and self.bgp_db is not None:
            explore_candidates = self._generate_exploration(n_explore, existing)

        # Merge, deduplicate, and truncate to n
        final_candidates = []
        seen = set()

        for addr in cex_candidates + explore_candidates:
            if addr not in seen and addr not in existing:
                seen.add(addr)
                final_candidates.append(addr)
                if len(final_candidates) >= n:
                    break

        print(f"    Generation complete: {len(final_candidates):,} "
              f"(CEX:{min(len(cex_candidates), n_cex):,}, "
              f"explore:{len(explore_candidates):,})")

        return final_candidates[:n]

    

    # Fraction of the final CEX candidates that come from extrapolation
    _EXTRAP_RATIO: float = 0.05

    def _generate_cex_with_priority(self, n: int, existing: Set[str]) -> List[str]:
        """
        Generate CEX candidates with proportional quota allocation.

        Allocation policy:
          - Ideal ratio: interp=(1-_EXTRAP_RATIO), extrap=_EXTRAP_RATIO
          - If either pool has insufficient candidates, re-allocate proportionally
            based on actual availability to maximize total output without wasting
            candidates from either side.
        """
        # 1. Parallel generation of raw CEX candidates
        prefix_items = list(self.prefix_data.items())
        num_chunks = self.config.num_workers * 2
        chunk_size = max(1, len(prefix_items) // num_chunks)

        chunks = []
        for i in range(0, len(prefix_items), chunk_size):
            chunk = prefix_items[i:i + chunk_size]
            chunks.append((chunk, existing, self.config.max_steps))

        all_results: List[Tuple[str, int, str]] = []

        if self.config.num_workers > 1 and len(chunks) > 1:
            with Pool(self.config.num_workers) as pool:
                for chunk_results in pool.imap_unordered(_process_prefix_chunk, chunks):
                    all_results.extend(chunk_results)
        else:
            for chunk in chunks:
                all_results.extend(_process_prefix_chunk(chunk))

        # 2. Global deduplication
        seen: Set[str] = set()
        unique: List[Tuple[str, int, str]] = []
        for item in all_results:
            addr = item[0]
            if addr not in seen and addr not in existing:
                seen.add(addr)
                unique.append(item)

        # 3. Split into interpolation pool and extrapolation pool
        interp_pool: List[Tuple[str, int, str]] = [
            x for x in unique if x[2] == 'interpolate'
        ]
        extrap_pool: List[Tuple[str, int, str]] = [
            x for x in unique if x[2] in ('forward', 'backward')
        ]

        random.shuffle(interp_pool)
        extrap_pool.sort(key=lambda x: (x[1], random.random()))

        # 4. Proportional allocation based on actual pool sizes to maximize total output
        avail_interp = len(interp_pool)
        avail_extrap = len(extrap_pool)
        avail_total  = avail_interp + avail_extrap

        if avail_total == 0:
            return []

        actual_total  = min(n, avail_total)
        actual_interp = min(avail_interp, round(actual_total * avail_interp / avail_total))
        actual_extrap = actual_total - actual_interp

        selected_interp = interp_pool[:actual_interp]
        selected_extrap = extrap_pool[:actual_extrap]

        # 5. Interleave
        # Interleave interval is computed dynamically from the actual extrap ratio
        actual_extrap_ratio = actual_extrap / actual_total if actual_total > 0 else self._EXTRAP_RATIO
        interleave_every = max(1, int(round(1.0 / actual_extrap_ratio))) if actual_extrap_ratio > 0 else actual_total

        final_items: List[Tuple[str, int, str]] = []
        ei = 0
        ii = 0
        slot = 0
        while ii < len(selected_interp) or ei < len(selected_extrap):
            if ei < len(selected_extrap) and slot % interleave_every == (interleave_every - 1):
                final_items.append(selected_extrap[ei])
                ei += 1
            elif ii < len(selected_interp):
                final_items.append(selected_interp[ii])
                ii += 1
            elif ei < len(selected_extrap):
                # interp exhausted: append all remaining extrap directly
                final_items.extend(selected_extrap[ei:])
                break
            slot += 1

        # 6. Detailed statistics
        cnt_interp   = sum(1 for x in final_items if x[2] == 'interpolate')
        cnt_forward  = sum(1 for x in final_items if x[2] == 'forward')
        cnt_backward = sum(1 for x in final_items if x[2] == 'backward')
        cnt_extrap   = cnt_forward + cnt_backward
        total        = len(final_items)
        extrap_pct   = cnt_extrap / total * 100 if total > 0 else 0.0
        ideal_interp_pct = (1.0 - self._EXTRAP_RATIO) * 100
        ideal_extrap_pct = self._EXTRAP_RATIO * 100

        print(
            f"    CEX generated: raw={len(unique):,} -> interleaved={total:,} "
            f"(target={n:,}, interleave=1/{interleave_every}) "
            f"| interp={cnt_interp:,} ({100.0 - extrap_pct:.1f}% / ideal {ideal_interp_pct:.0f}%) "
            f"| extrap={cnt_extrap:,} ({extrap_pct:.1f}% / ideal {ideal_extrap_pct:.0f}%)"
            f" [fwd={cnt_forward:,} bwd={cnt_backward:,}]"
        )

        # 7. Return addresses only
        return [item[0] for item in final_items]

    def _generate_exploration(self, n: int, existing: Set[str]) -> List[str]:
        """
        Exploration layer generation (BGP-guided).

        Bug4 Fix: use a string address set (seen_addrs) instead of a hash set
        for deduplication.
        """
        if self.bgp_db is None or len(self.bgp_db.prefixes) == 0:
            return []

        generated = []

        # Extract interface ID pool (up to 10000)
        interface_ids = []
        for addr in list(self.all_active_set)[:5000]:
            try:
                addr_int = int(ipaddress.IPv6Address(addr))
                iid = addr_int & ((1 << 64) - 1)
                interface_ids.append(iid)
            except:
                continue

        if not interface_ids:
            return []

        if HAS_NUMPY:
            interface_id_pool = np.array(interface_ids, dtype=np.uint64)
        else:
            interface_id_pool = interface_ids

        # Quota allocation: 2a(50%), 2b(40%), 2c(10%)
        quota_2a = int(n * 0.50)
        quota_2b = int(n * 0.40)
        quota_2c = int(n * 0.10)

        # Bug4 Fix: use string set instead of hash set for deduplication
        seen_addrs: Set[str] = set(existing)

        # Exploration 2a: BGP adjacent AS exploration
        adjacent_as = set()
        for as_num in list(self.seed_as_set)[:200]:
            neighbors = self.bgp_db.get_as_neighbors(as_num)
            adjacent_as.update(neighbors[:30])
            if len(adjacent_as) >= 1000:
                break

        new_prefixes_64 = []
        for as_num in list(adjacent_as)[:200]:
            as_prefixes = self.bgp_db.get_as_prefixes(as_num)
            for bgp_prefix in as_prefixes[:30]:
                if bgp_prefix in self.prefixes_ipv6 if hasattr(self, 'prefixes_ipv6') else False:
                    continue
                try:
                    if '/' in bgp_prefix:
                        network = ipaddress.IPv6Network(bgp_prefix, strict=False)
                        prefix_64 = int(network.network_address) >> 64
                    else:
                        prefix_64 = int(ipaddress.IPv6Address(bgp_prefix)) >> 64
                    new_prefixes_64.append(prefix_64)
                except:
                    continue

                if len(new_prefixes_64) >= 5000:
                    break
            if len(new_prefixes_64) >= 1000:
                break

        generated_2a = []
        if len(new_prefixes_64) > 0:
            new_prefixes_64 = new_prefixes_64[:300]
            quota_per_prefix = max(100, quota_2a // len(new_prefixes_64))

            for prefix_64 in new_prefixes_64:
                if len(generated_2a) >= quota_2a:
                    break

                num_to_gen = min(quota_per_prefix * 6, 10000)
                if HAS_NUMPY:
                    random_indices = np.random.randint(0, len(interface_id_pool), size=num_to_gen)
                    selected_interfaces = interface_id_pool[random_indices]
                else:
                    selected_interfaces = [random.choice(interface_id_pool) for _ in range(num_to_gen)]

                for interface_id in selected_interfaces:
                    if len(generated_2a) >= quota_2a:
                        break
                    try:
                        addr_int = (prefix_64 << 64) | int(interface_id)
                        normalized = ipaddress.IPv6Address(addr_int).exploded
                        if normalized not in seen_addrs:
                            seen_addrs.add(normalized)
                            generated_2a.append(ipv6_to_cont(normalized))
                    except:
                        continue

        generated.extend(generated_2a)

        # Exploration 2b: /48-level exploration
        generated_2b = []

        all_bgp_list = list(self.bgp_db.prefixes.keys())
        if len(all_bgp_list) > 1000:
            all_bgp_list = random.sample(all_bgp_list, 1000)

        new_48_to_64_list = []
        temp_48_dict = defaultdict(list)

        for bgp_prefix in all_bgp_list:
            if bgp_prefix in self.prefixes_ipv6 if hasattr(self, 'prefixes_ipv6') else False:
                continue
            try:
                if '/' in bgp_prefix:
                    network = ipaddress.IPv6Network(bgp_prefix, strict=False)
                    addr_int = int(network.network_address) >> 64
                else:
                    addr_int = int(ipaddress.IPv6Address(bgp_prefix)) >> 64

                prefix_48 = addr_int >> 16
                temp_48_dict[prefix_48].append(addr_int)
            except:
                continue

        for prefix_48, prefix_64_list in sorted(temp_48_dict.items(),
                                                 key=lambda x: len(x[1]),
                                                 reverse=True)[:300]:
            new_48_to_64_list.append((prefix_48, prefix_64_list[:100]))

        if len(new_48_to_64_list) > 0:
            quota_per_48 = max(100, quota_2b // len(new_48_to_64_list))

            for prefix_48, prefix_64_list in new_48_to_64_list:
                if len(generated_2b) >= quota_2b:
                    break

                quota_per_64 = max(50, quota_per_48 // len(prefix_64_list))

                for prefix_64 in prefix_64_list:
                    if len(generated_2b) >= quota_2b:
                        break

                    num_to_gen = min(quota_per_64 * 5, 5000)
                    if HAS_NUMPY:
                        random_indices = np.random.randint(0, len(interface_id_pool), size=num_to_gen)
                        selected_interfaces = interface_id_pool[random_indices]
                    else:
                        selected_interfaces = [random.choice(interface_id_pool) for _ in range(num_to_gen)]

                    for interface_id in selected_interfaces:
                        if len(generated_2b) >= quota_2b:
                            break
                        try:
                            addr_int = (prefix_64 << 64) | int(interface_id)
                            normalized = ipaddress.IPv6Address(addr_int).exploded
                            if normalized not in seen_addrs:
                                seen_addrs.add(normalized)
                                generated_2b.append(ipv6_to_cont(normalized))
                        except:
                            continue

        generated.extend(generated_2b)

        # Exploration 2c: /64 neighbor exploration
        generated_2c = []

        known_prefix64_ints = set()
        for p64_hex in self.prefixes:
            try:
                prefix_int = int(p64_hex, 16)
                known_prefix64_ints.add(prefix_int)
            except:
                continue

        neighbor_prefixes = set()
        for prefix_int in list(known_prefix64_ints)[:1000]:
            for offset in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32,
                           64, -64, 128, -128, 256, -256, 512, -512]:
                neighbor = prefix_int + offset
                if neighbor >= 0 and neighbor < (1 << 64):
                    neighbor_prefixes.add(neighbor)

            if len(neighbor_prefixes) >= 5000:
                break

        new_neighbor_prefixes = neighbor_prefixes - known_prefix64_ints

        if len(new_neighbor_prefixes) > 0:
            new_neighbor_list = list(new_neighbor_prefixes)[:1000]
            quota_per_prefix = max(50, quota_2c // len(new_neighbor_list))

            for prefix_64_int in new_neighbor_list:
                if len(generated_2c) >= quota_2c:
                    break

                num_to_gen = min(quota_per_prefix * 4, 3000)
                if HAS_NUMPY:
                    random_indices = np.random.randint(0, len(interface_id_pool), size=num_to_gen)
                    selected_interfaces = interface_id_pool[random_indices]
                else:
                    selected_interfaces = [random.choice(interface_id_pool) for _ in range(num_to_gen)]

                for interface_id in selected_interfaces:
                    if len(generated_2c) >= quota_2c:
                        break
                    try:
                        addr_int = (prefix_64_int << 64) | int(interface_id)
                        normalized = ipaddress.IPv6Address(addr_int).exploded
                        if normalized not in seen_addrs:
                            seen_addrs.add(normalized)
                            generated_2c.append(ipv6_to_cont(normalized))
                    except:
                        continue

        generated.extend(generated_2c)

        return generated


# ============================================================
# Module 8: Pattern Matcher
# ============================================================

class PatternMatcher:
    """Pattern matcher (high-performance version)"""

    def __init__(self, config: Config):
        self.config = config
        self.prefix_data: Dict[str, Dict] = {}
        self._prefix_iids_arrays: Dict[str, List[int]] = {}
        self._prefix_steps: Dict[str, int] = {}

    def fit(self, seeds: List[str]):
        """Learn patterns from seeds"""
        prefix_iids = defaultdict(list)

        for s in seeds:
            cont = s if len(s) == 32 and ':' not in s else ipv6_to_cont(s)
            p64 = cont[:16]
            iid = int(cont[16:], 16)
            prefix_iids[p64].append(iid)

        self.prefix_data.clear()
        self._prefix_iids_arrays.clear()
        self._prefix_steps.clear()

        for p64, iids in prefix_iids.items():
            iids = sorted(set(iids))
            if len(iids) >= 2:
                diffs = [iids[i+1] - iids[i] for i in range(len(iids)-1)]
                median_step = sorted(diffs)[len(diffs)//2]
            else:
                median_step = 1000

            step = max(1, median_step)

            self.prefix_data[p64] = {
                'iids': iids,
                'step': step
            }
            self._prefix_iids_arrays[p64] = iids
            self._prefix_steps[p64] = step

    def filter_matching(self, addrs: List[str]) -> Tuple[List[str], List[str]]:
        
        if not addrs:
            return [], []

        n_workers = self.config.num_workers
        tolerance = self.config.tolerance

        # Opt3: threshold lowered from 10000 to 1000
        if len(addrs) < 1000 or n_workers <= 1:
            return self._filter_single_thread(addrs)

        # Opt4: n_workers*16 chunks for better serialization/scheduling balance;
        # max(50000,...) floor avoids excessive scheduling overhead from too many chunks
        num_chunks = n_workers * 16
        chunk_size = max(50000, len(addrs) // num_chunks)
        chunks = []
        for i in range(0, len(addrs), chunk_size):
            chunks.append(addrs[i:i + chunk_size])

        args_list = [
            (chunk, self._prefix_iids_arrays, self._prefix_steps, tolerance)
            for chunk in chunks
        ]

        all_matched = []
        all_unmatched = []

        with Pool(n_workers) as pool:
            for matched, unmatched in pool.imap_unordered(_filter_chunk_fast, args_list):
                all_matched.extend(matched)
                all_unmatched.extend(unmatched)

        return all_matched, all_unmatched

    def _filter_single_thread(self, addrs: List[str]) -> Tuple[List[str], List[str]]:
        """Single-threaded processing (small datasets)"""
        matched = []
        unmatched = []
        tolerance = self.config.tolerance

        for addr in addrs:
            if len(addr) == 32 and ':' not in addr:
                cont = addr
            else:
                try:
                    cont = ipaddress.IPv6Address(addr).exploded.replace(":", "")
                except:
                    unmatched.append(addr)
                    continue

            p64 = cont[:16]

            if p64 not in self.prefix_data:
                unmatched.append(addr)
                continue

            data = self.prefix_data[p64]
            iids = data['iids']
            step = data['step']

            if step == 0:
                matched.append(addr)
                continue

            iid = int(cont[16:], 16)
            pos = bisect.bisect_left(iids, iid)

            min_dist = float('inf')
            if pos < len(iids):
                min_dist = abs(iids[pos] - iid)
            if pos > 0:
                d = abs(iids[pos - 1] - iid)
                if d < min_dist:
                    min_dist = d

            if min_dist == 0:
                matched.append(addr)
            elif min_dist < step * 0.5:
                unmatched.append(addr)
            else:
                ratio = min_dist / step
                nearest_mult = round(ratio)
                if nearest_mult == 0:
                    unmatched.append(addr)
                else:
                    expected_dist = nearest_mult * step
                    deviation = abs(min_dist - expected_dist) / expected_dist
                    if deviation <= tolerance:
                        matched.append(addr)
                    else:
                        unmatched.append(addr)

        return matched, unmatched

    def add_addresses(self, addrs: List[str]):
        
        # 1. Group new IIDs by prefix (only for known prefixes)
        prefix_new_iids: Dict[str, List[int]] = defaultdict(list)

        for addr in addrs:
            cont = addr if len(addr) == 32 and ':' not in addr else ipv6_to_cont(addr)
            p64 = cont[:16]
            if p64 in self.prefix_data:
                iid = int(cont[16:], 16)
                prefix_new_iids[p64].append(iid)

        if not prefix_new_iids:
            return

        # 2. Bulk merge per prefix (avoids O(n^2) shifting from repeated bisect.insort)
        for p64, new_iids in prefix_new_iids.items():
            old_iids = self._prefix_iids_arrays[p64]  # already sorted
            old_set = set(old_iids)

            # Keep only genuinely new IIDs
            truly_new = sorted({iid for iid in new_iids if iid not in old_set})

            if not truly_new:
                continue

            # Linear merge of two sorted lists, O(|old| + |new|)
            merged = []
            i, j = 0, 0
            while i < len(old_iids) and j < len(truly_new):
                if old_iids[i] <= truly_new[j]:
                    merged.append(old_iids[i])
                    i += 1
                else:
                    merged.append(truly_new[j])
                    j += 1
            if i < len(old_iids):
                merged.extend(old_iids[i:])
            if j < len(truly_new):
                merged.extend(truly_new[j:])

            # Update internal state
            self._prefix_iids_arrays[p64] = merged
            self.prefix_data[p64]['iids'] = merged

            # Recompute step as median of differences
            if len(merged) >= 2:
                diffs = [merged[k+1] - merged[k] for k in range(len(merged)-1)]
                new_step = max(1, sorted(diffs)[len(diffs)//2])
                self.prefix_data[p64]['step'] = new_step
                self._prefix_steps[p64] = new_step


def _filter_chunk_fast(args: Tuple) -> Tuple[List[str], List[str]]:
    
    addrs, prefix_iids_arrays, prefix_steps, tolerance = args

    matched = []
    unmatched = []

    for addr in addrs:
        # Opt6: already 32-char continuous hex, skip parsing entirely
        if len(addr) == 32 and ':' not in addr:
            cont = addr
        else:
            addr_s = addr.strip()
            # Opt1: fast path for standard exploded format (e.g. 2001:0db8:...)
            # Fixed length 39 with exactly 7 colons; strip colons directly
            if len(addr_s) == 39 and addr_s.count(':') == 7:
                cont = addr_s.replace(':', '')
            else:
                # Bug3 fallback: compressed formats (::1, fe80::, 2001:db8::1, etc.)
                try:
                    cont = format(int(ipaddress.IPv6Address(addr_s)), '032x')
                except:
                    unmatched.append(addr)
                    continue

        p64 = cont[:16]

        if p64 not in prefix_iids_arrays:
            unmatched.append(addr)
            continue

        iids = prefix_iids_arrays[p64]
        step = prefix_steps[p64]

        if step == 0:
            matched.append(addr)
            continue

        iid = int(cont[16:], 16)
        pos = bisect.bisect_left(iids, iid)

        min_dist = float('inf')
        if pos < len(iids):
            min_dist = abs(iids[pos] - iid)
        if pos > 0:
            d = abs(iids[pos - 1] - iid)
            if d < min_dist:
                min_dist = d

        if min_dist == 0:
            matched.append(addr)
        elif min_dist < step * 0.5:
            unmatched.append(addr)
        else:
            ratio = min_dist / step
            nearest_mult = round(ratio)
            if nearest_mult == 0:
                unmatched.append(addr)
            else:
                expected_dist = nearest_mult * step
                deviation = abs(min_dist - expected_dist) / expected_dist
                if deviation <= tolerance:
                    matched.append(addr)
                else:
                    unmatched.append(addr)

    return matched, unmatched


# ============================================================
# Module 9: Masscan Scanner
# ============================================================

class MasscanScanner:
    """Masscan scanner"""

    def __init__(self, config: Config):
        self.config = config

    def check_available(self) -> Tuple[bool, str]:
        try:
            result = subprocess.run(['which', 'masscan'], capture_output=True, text=True)
            if result.returncode != 0:
                return False, "masscan not installed"
            return True, "OK"
        except Exception as e:
            return False, str(e)

    def _parse_output(self, output: str) -> Set[str]:
        active = set()
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("open"):
                parts = line.split()
                if len(parts) >= 4 and ":" in parts[3]:
                    active.add(parts[3])
        return active

    def scan_port(self, input_file: str) -> Set[str]:
        cmd = [
            "masscan", "-6",
            "-p", self.config.scan_port,
            "--rate", str(self.config.scan_rate),
            "--source-ip", self.config.source_ip,
            "--interface", self.config.interface,
            "--router-mac-ipv6", self.config.router_mac,
            "--open-only",
            "-oL", "-",
            "-iL", input_file
        ]

        print(f"      Running: masscan -6 -p {self.config.scan_port} --rate {self.config.scan_rate} ...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

            if result.returncode != 0:
                print(f"      Warning: return code {result.returncode}")
                if result.stderr:
                    stderr_lines = result.stderr.strip().split('\n')[:3]
                    for line in stderr_lines:
                        print(f"      {line}")

            return self._parse_output(result.stdout)
        except subprocess.TimeoutExpired:
            print(f"      Error: timeout")
            return set()
        except Exception as e:
            print(f"      Error: {e}")
            return set()

    def scan_ping(self, input_file: str) -> Set[str]:
        cmd = [
            "masscan", "-6",
            "--ping",
            "--rate", str(self.config.ping_rate),
            "--router-mac-ipv6", self.config.router_mac,
            "--interface", self.config.interface,
            "-oL", "-",
            "-iL", input_file
        ]

        print(f"      Running: masscan -6 --ping --rate {self.config.ping_rate} ...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

            if result.returncode != 0:
                print(f"      Warning: return code {result.returncode}")
                if result.stderr:
                    stderr_lines = result.stderr.strip().split('\n')[:3]
                    for line in stderr_lines:
                        print(f"      {line}")

            return self._parse_output(result.stdout)
        except subprocess.TimeoutExpired:
            print(f"      Error: timeout")
            return set()
        except Exception as e:
            print(f"      Error: {e}")
            return set()

    def scan(self, input_file: str) -> Tuple[Set[str], Dict]:
        with open(input_file, 'r') as f:
            num_candidates = sum(1 for line in f if line.strip())

        print(f"  Scanning {num_candidates:,} addresses...")
        start = time.time()

        print(f"    [1/2] Port scan ({self.config.scan_port})...")
        port_hits = self.scan_port(input_file)
        print(f"      Hits: {len(port_hits):,}")

        print(f"    [2/2] ICMP Ping...")
        ping_hits = self.scan_ping(input_file)
        print(f"      Hits: {len(ping_hits):,}")

        all_hits = port_hits | ping_hits
        elapsed = time.time() - start
        hit_rate = len(all_hits) / num_candidates * 100 if num_candidates > 0 else 0

        stats = {
            'candidates': num_candidates,
            'port_hits': len(port_hits),
            'ping_hits': len(ping_hits),
            'total_hits': len(all_hits),
            'hit_rate': hit_rate,
            'elapsed': elapsed
        }

        print(f"  Scan complete: {len(all_hits):,} hits ({hit_rate:.2f}%), {elapsed:.1f}s")

        return all_hits, stats


# ============================================================
# Module 10: Main Controller
# ============================================================

class CEXController:
    """CEX automation controller"""

    def __init__(self, config: Config):
        self.config = config
        self.state = StateManager(config)
        self.alias_filter = AliasFilter(config)
        self.generator = HybridGenerator(config)
        self.matcher = PatternMatcher(config)
        self.scanner = MasscanScanner(config)

    def initialize(self, seeds_file: str) -> Set[str]:
        """Initialize"""
        print("\n[Initialization]")

        addrs = read_addrs(seeds_file)
        if not addrs:
            raise ValueError(f"Cannot read: {seeds_file}")

        seeds_cont = {ipv6_to_cont(a) for a in addrs}
        print(f"  Raw seeds: {len(seeds_cont):,}")

        self.alias_filter.load(verbose=True)

        if self.alias_filter.enabled:
            seeds_ipv6 = [cont_to_ipv6(s) for s in seeds_cont]
            filtered, stats = self.alias_filter.filter_addresses(seeds_ipv6, verbose=True)
            seeds_cont = {ipv6_to_cont(a) for a in filtered}

        self.state.save_seeds(seeds_cont)
        self.state.save_all_active(seeds_cont)

        print(f"  Valid seeds: {len(seeds_cont):,}")

        return seeds_cont

    def run_round(self, round_num: int, num_candidates: int, do_scan: bool = True) -> Dict:
        """Run a single round (with detailed timing)"""
        print(f"\n{'='*70}")
        print(f"Round {round_num} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Memory: {resource_stats.get_current_memory_mb():.1f} MB")
        print(f"{'='*70}")

        round_start = time.time()

        stats = {
            'round': round_num,
            'candidates': 0,
            'candidates_after_alias': 0,
            'hits': 0,
            'hits_after_alias': 0,
            'hit_rate': 0,
            'matched': 0,
            'seeds_before': 0,
            'seeds_after': 0,
            'time_load': 0,
            'time_train': 0,
            'time_generate': 0,
            'time_alias_filter': 0,
            'time_scan': 0,
            'time_pattern_match': 0,
            'time_save': 0,
            'time_total': 0
        }

        # 1. Load state
        print("\n[1/6] Loading state")
        t1 = time.time()
        seeds = self.state.load_seeds()
        history = self.state.load_history()
        all_active = self.state.load_all_active()
        stats['time_load'] = time.time() - t1

        stats['seeds_before'] = len(seeds)
        print(f"  Seeds: {len(seeds):,}, history: {len(history):,} ({stats['time_load']:.2f}s)")

        # 2. Train
        print("\n[2/6] Training model")
        t2 = time.time()
        self.generator.fit(list(seeds))
        self.matcher.fit(list(seeds))
        stats['time_train'] = time.time() - t2
        print(f"  Training time: {stats['time_train']:.2f}s")

        # 3. Generate candidates
        print("\n[3/6] Generating candidates")
        t3 = time.time()
        exclude = seeds | history
        candidates = self.generator.generate(num_candidates, exclude)

        final_cands = candidates
        output_addrs = set(candidates)

        stats['candidates'] = len(final_cands)
        stats['time_generate'] = time.time() - t3
        print(f"  Generation time: {stats['time_generate']:.2f}s")

        resource_stats.update_memory()

        if not final_cands:
            print("  No new candidates")
            stats['time_total'] = time.time() - round_start
            return stats

        # 4. Alias filter on candidates
        print("\n[4/6] Alias filtering candidates")
        t4 = time.time()
        cand_file = os.path.join(self.config.output_dir, f"candidates_r{round_num}.txt")

        if self.alias_filter.enabled:
            temp_file = cand_file + ".tmp"
            with open(temp_file, 'w') as f:
                for addr in final_cands:
                    f.write(cont_to_ipv6(addr) + '\n')

            alias_stats = self.alias_filter.filter_file(temp_file, cand_file, verbose=True)
            os.remove(temp_file)

            stats['candidates_after_alias'] = alias_stats.remaining

            output_addrs = set()
            with open(cand_file, 'r') as f:
                for line in f:
                    addr = line.strip()
                    if addr:
                        output_addrs.add(ipv6_to_cont(addr))
        else:
            with open(cand_file, 'w') as f:
                for addr in final_cands:
                    f.write(cont_to_ipv6(addr) + '\n')
            stats['candidates_after_alias'] = len(final_cands)

        stats['time_alias_filter'] = time.time() - t4
        print(f"  Saved: {cand_file} ({stats['candidates_after_alias']:,})")
        print(f"  Alias filter time: {stats['time_alias_filter']:.2f}s")

        # 5. Scan
        if do_scan:
            print("\n[5/6] Scanning")
            t5 = time.time()
            hits, scan_stats = self.scanner.scan(cand_file)

            stats['hits'] = len(hits)
            stats['hit_rate'] = scan_stats['hit_rate']

            if self.alias_filter.enabled and hits:
                print("  Alias filtering hits...")
                filtered_hits, alias_stats = self.alias_filter.filter_addresses(hits, verbose=True)
                hits = set(filtered_hits)
                stats['hits_after_alias'] = len(hits)
            else:
                stats['hits_after_alias'] = len(hits)

            stats['time_scan'] = time.time() - t5
            print(f"  Scan time: {stats['time_scan']:.2f}s")

            hits_file = os.path.join(self.config.output_dir, f"hits_r{round_num}.txt")
            with open(hits_file, 'w') as f:
                for ip in sorted(hits):
                    f.write(ip + '\n')

            hits_cont = {ipv6_to_cont(ip) for ip in hits}

            # 6. Incremental training
            print("\n[6/6] Incremental training")
            t6 = time.time()
            new_hits = list(hits_cont - seeds)

            if new_hits:
                print(f"  New hits: {len(new_hits):,}")
                matched, unmatched = self.matcher.filter_matching(new_hits)

                stats['matched'] = len(matched)
                print(f"    Pattern matched: {len(matched):,}")
                print(f"    Unmatched: {len(unmatched):,}")

                if matched:
                    for addr in matched:
                        seeds.add(addr)
                    self.matcher.add_addresses(matched)

            stats['time_pattern_match'] = time.time() - t6
            print(f"  Pattern match time: {stats['time_pattern_match']:.2f}s")

            all_active.update(hits_cont)
        else:
            print("\n[5/6] Scan skipped")
            print("\n[6/6] Incremental training skipped")

        # Save state
        t7 = time.time()
        self.state.save_seeds(seeds)
        self.state.append_history(output_addrs)
        self.state.save_all_active(all_active)
        stats['time_save'] = time.time() - t7

        stats['seeds_after'] = len(seeds)
        stats['time_total'] = time.time() - round_start

        resource_stats.update_memory()

        # Summary
        print(f"\n[Round {round_num} complete]")
        print(f"  Candidates: {stats['candidates']:,} -> {stats['candidates_after_alias']:,} (after alias filter)")
        if do_scan:
            print(f"  Hits: {stats['hits']:,} -> {stats['hits_after_alias']:,} ({stats['hit_rate']:.2f}%)")
            print(f"  Pattern matched: {stats['matched']:,}")
        print(f"  Seeds: {stats['seeds_before']:,} -> {stats['seeds_after']:,}")
        print(f"  Round time: {stats['time_total']:.1f}s")
        print(f"    Generate: {stats['time_generate']:.1f}s | Scan: {stats['time_scan']:.1f}s | "
              f"Match: {stats['time_pattern_match']:.1f}s | Save: {stats['time_save']:.1f}s")
        print(f"  Current memory: {resource_stats.get_current_memory_mb():.1f} MB")

        resource_stats.round_stats.append({
            'round': round_num,
            'gen_time': stats['time_generate'],
            'scan_time': stats['time_scan'],
            'train_time': stats['time_train'] + stats['time_pattern_match'],
            'total_time': stats['time_total']
        })

        return stats

    def run(self, seeds_file: str, num_rounds: int, num_candidates: int,
            do_scan: bool = True, auto_reset: bool = True):
        """Run multiple rounds"""
        total_start_time = time.time()

        print("=" * 70)
        print("CEX Auto  - Extrapolation Quota (5% extrap / 95% interp)")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Seeds: {seeds_file}")
        print(f"Rounds: {num_rounds}, per round: {num_candidates:,}")
        print(f"Workers: {self.config.num_workers}")
        print(f"CEX quota: interp={100*(1-self.generator._EXTRAP_RATIO):.0f}%  "
              f"extrap={100*self.generator._EXTRAP_RATIO:.0f}%")

        if self.config.explore_ratio > 0:
            print(f"Mode: CEX + exploration hybrid")
            print(f"  CEX ratio: {(1-self.config.explore_ratio)*100:.0f}%")
            print(f"  Exploration ratio: {self.config.explore_ratio*100:.0f}%")
            print(f"  BGP data: {'loaded' if self.generator.bgp_db else 'not loaded (exploration disabled)'}")
        else:
            print(f"Mode: pure CEX")

        print(f"Alias filter: {'enabled' if self.config.enable_alias_filter else 'disabled'}")

        if do_scan:
            ok, msg = self.scanner.check_available()
            if not ok:
                print(f"\nError: {msg}")
                sys.exit(1)

        if auto_reset and not self.state.is_first_run():
            print("\n[!] Existing state detected, overwriting...")
            if os.path.exists(self.config.state_dir):
                shutil.rmtree(self.config.state_dir)
            self.config.setup_dirs()

        self.initialize(seeds_file)

        all_stats = {
            'rounds': [],
            'total_scanned': 0,
            'total_hits': 0,
            'start_time': datetime.now().isoformat(),
            'timing': {
                'total': 0,
                'generation': 0,
                'alias_filter': 0,
                'scanning': 0,
                'pattern_match': 0
            }
        }

        try:
            for r in range(1, num_rounds + 1):
                round_stats = self.run_round(r, num_candidates, do_scan)

                all_stats['rounds'].append(round_stats)
                all_stats['total_scanned'] = sum(rs['candidates_after_alias'] for rs in all_stats['rounds'])
                all_stats['total_hits'] = sum(rs['hits_after_alias'] for rs in all_stats['rounds'])

                all_stats['timing']['generation'] += round_stats.get('time_generate', 0)
                all_stats['timing']['alias_filter'] += round_stats.get('time_alias_filter', 0)
                all_stats['timing']['scanning'] += round_stats.get('time_scan', 0)
                all_stats['timing']['pattern_match'] += round_stats.get('time_pattern_match', 0)

                self.state.save_stats(all_stats)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        total_elapsed = time.time() - total_start_time
        all_stats['timing']['total'] = total_elapsed
        all_stats['end_time'] = datetime.now().isoformat()
        self.state.save_stats(all_stats)

        self._print_report(all_stats)

    def _print_report(self, stats: Dict):
        """Print final report"""
        print("\n" + "=" * 70)
        print("Final Report")
        print("=" * 70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not stats['rounds']:
            return

        print(f"\n[Scan Statistics]")
        print(f"Rounds completed: {len(stats['rounds'])}")
        print(f"Total scanned: {stats['total_scanned']:,}")
        print(f"Total hits: {stats['total_hits']:,}")

        avg_rate = stats['total_hits'] / stats['total_scanned'] * 100 if stats['total_scanned'] > 0 else 0
        print(f"Average hit rate: {avg_rate:.2f}%")

        print(f"\n[Per-Round Details]")
        print(f"{'Round':<6} {'Candidates':<12} {'Hits':<12} {'Hit Rate':<10} {'New Seeds':<10} {'Time':<10}")
        print("-" * 70)

        total_time = 0
        for rs in stats['rounds']:
            round_time = rs.get('time_total', 0)
            total_time += round_time
            print(f"{rs['round']:<6} {rs['candidates_after_alias']:<12,} {rs['hits_after_alias']:<12,} "
                  f"{rs['hit_rate']:<10.2f}% {rs['matched']:<10,} {round_time:<10.1f}s")

        if len(stats['rounds']) >= 2:
            rates = [rs['hit_rate'] for rs in stats['rounds']]
            trend = "up" if rates[-1] > rates[0] else ("down" if rates[-1] < rates[0] else "stable")
            print(f"\nHit rate trend: {rates[0]:.2f}% -> {rates[-1]:.2f}% ({trend})")

        print(f"\n[Timing Statistics]")

        total_gen = sum(rs.get('time_generate', 0) for rs in stats['rounds'])
        total_scan = sum(rs.get('time_scan', 0) for rs in stats['rounds'])
        total_alias = sum(rs.get('time_alias_filter', 0) for rs in stats['rounds'])
        total_train = sum(rs.get('time_train', 0) + rs.get('time_pattern_match', 0) for rs in stats['rounds'])

        print(f"Total runtime: {resource_stats.format_duration(total_time)}")
        if total_time > 0:
            print(f"  - Generation: {resource_stats.format_duration(total_gen)} ({total_gen/total_time*100:.1f}%)")
            print(f"  - Scanning: {resource_stats.format_duration(total_scan)} ({total_scan/total_time*100:.1f}%)")
            print(f"  - Alias filter: {resource_stats.format_duration(total_alias)} ({total_alias/total_time*100:.1f}%)")
            print(f"  - Training/matching: {resource_stats.format_duration(total_train)} ({total_train/total_time*100:.1f}%)")

        print(f"\n[Resource Statistics]")
        print(f"Peak memory: {resource_stats.peak_memory_mb:.1f} MB")
        print(f"Current memory: {resource_stats.get_current_memory_mb():.1f} MB")
        print(f"CPU cores: {cpu_count()}")
        print(f"Workers used: {self.config.num_workers}")

        if total_time > 0 and stats['total_scanned'] > 0:
            print(f"\n[Performance Metrics]")
            if total_gen > 0:
                print(f"Average generation speed: {stats['total_scanned']/total_gen:,.0f} addr/s")
            if total_scan > 0:
                print(f"Average scan speed: {stats['total_scanned']/total_scan:,.0f} addr/s")
            print(f"Average time per round: {total_time/len(stats['rounds']):.1f}s")

        print(f"\n[Output Files]")
        print(f"State directory: {self.config.state_dir}")
        print(f"Output directory: {self.config.output_dir}")
        seeds = self.state.load_seeds()
        all_active = self.state.load_all_active()
        print(f"Final seed count: {len(seeds):,}")
        print(f"Cumulative active count: {len(all_active):,}")

        print("\n" + "=" * 70)


# ============================================================
# Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CEX Auto  - Extrapolation Quota (5% extrap / 95% interp)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--seeds", required=True, help="Initial seed file")
    parser.add_argument("--rounds", type=int, default=10, help="Number of iterations (default 10)")
    parser.add_argument("--num", type=int, default=1000000, help="Candidates per round (default 1000000)")

    parser.add_argument("--explore-ratio", type=float, default=0.0,
                        help="Exploration ratio (0-1, default 0 means pure CEX mode)")
    parser.add_argument("--bgp-file", help="BGP data file (JSON format, required for exploration layer)")

    parser.add_argument("--alias-file", default="./aliased-prefixes.txt", help="Alias prefix file")
    parser.add_argument("--no-alias-filter", action="store_true", help="Disable alias filter")

    parser.add_argument("--max-steps", type=int, default=30, help="Maximum extrapolation steps (default 30)")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Pattern matching tolerance (default 0.5)")

    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (default CPU-1)")
    parser.add_argument("--state-dir", default="./cex_state", help="State directory")
    parser.add_argument("--output-dir", default="./cex_output", help="Output directory")
    parser.add_argument("--reset", action="store_true", help="Reset state and start from scratch")

    parser.add_argument("--interface", default="", help="Network interface for masscan")
    parser.add_argument("--source-ip", default="", help="Source IPv6 address for masscan")
    parser.add_argument("--router-mac", default="", help="Router MAC address for masscan")

    args = parser.parse_args()

    if args.explore_ratio < 0 or args.explore_ratio > 1:
        print("Error: --explore-ratio must be between 0 and 1")
        sys.exit(1)

    if args.explore_ratio > 0 and not args.bgp_file:
        print("Error: --bgp-file is required when exploration layer is enabled")
        sys.exit(1)

    config = Config()
    config.state_dir = args.state_dir
    config.output_dir = args.output_dir
    config.num_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    config.max_steps = args.max_steps
    config.tolerance = args.tolerance

    config.alias_file = args.alias_file
    config.enable_alias_filter = not args.no_alias_filter

    config.explore_ratio = args.explore_ratio
    config.bgp_file = args.bgp_file

    config.interface = args.interface
    config.source_ip = args.source_ip
    config.router_mac = args.router_mac

    config.setup_dirs()

    if args.reset:
        if os.path.exists(config.state_dir):
            shutil.rmtree(config.state_dir)
        config.setup_dirs()

    controller = CEXController(config)
    controller.run(
        seeds_file=args.seeds,
        num_rounds=args.rounds,
        num_candidates=args.num,
        do_scan=not args.no_scan
    )


if __name__ == "__main__":
    main()
