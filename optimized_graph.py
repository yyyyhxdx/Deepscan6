"""
Optimized Graph Density Cut Detector

Optimizations:
1.  KDTree-accelerated nearest neighbor search (ball_tree)
2.  Sparse directed k-NN graph — symmetrization removed (BFS needs only reachability)
3.  Parallel multi-scale processing via pure functions (no shared-state race)
4.  Vectorized feature extraction: prefix+segment features fully numpy-broadcast
5.  Module-level precomputed mask/shift constants (computed once at import)
6.  collections.deque for O(1) BFS popleft
7.  Direct CSR indptr/indices access in BFS (no per-node sparse-slice creation)
8.  Adaptive parallel address parsing (thread pool for large inputs only)
9.  Per-stage timing breakdown in verbose output
10. _process_scale uses n_jobs=1 internally to avoid CPU over-subscription
11. EnhancedGraphDensityCutFast reuses max-k distances — no extra kNN pass
12. StandardScaler normalization before kNN to balance feature dimensions
13. Median-based density threshold for robustness against extreme sparse points
"""

import numpy as np
from collections import deque
from typing import List, Tuple
import ipaddress
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import time
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Module-level constants (computed once)
# ---------------------------------------------------------------------------

_PREFIX_LENS   = [16, 32, 48, 64, 80, 96]
_PREFIX_MASKS  = np.array(
    [(2**128 - 1) ^ (2**(128 - pl) - 1) for pl in _PREFIX_LENS], dtype=object)
_PREFIX_SHIFTS = np.array([128 - pl for pl in _PREFIX_LENS], dtype=np.int64)
_PREFIX_MAX    = np.array([2**pl - 1  for pl in _PREFIX_LENS], dtype=np.float64)
_SEG_SHIFTS    = np.array([112 - j * 16 for j in range(8)], dtype=np.int64)
_N_FEATURES    = len(_PREFIX_LENS) + len(_SEG_SHIFTS)   # 14

_PARALLEL_THRESHOLD = 2000


def _parse_one_address_graph(ipv6_str: str) -> np.ndarray:
    """14-dim feature vector for one IPv6 address. Pure function, thread-safe."""
    row = np.zeros(_N_FEATURES, dtype=np.float32)
    try:
        a = int(ipaddress.IPv6Address(ipv6_str))
        prefix_ints = np.array([(a & int(m)) >> int(s)
                                for m, s in zip(_PREFIX_MASKS, _PREFIX_SHIFTS)],
                               dtype=np.float64)
        row[:6] = (prefix_ints / _PREFIX_MAX).astype(np.float32)
        seg_ints = (a >> _SEG_SHIFTS.astype(np.int64)) & 0xFFFF
        row[6:] = (seg_ints / 65535.0).astype(np.float32)
    except Exception:
        pass
    return row


def _process_scale(X: np.ndarray, k: int, density_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute outlier mask AND distances for one k-scale.
    Uses median-based threshold: robust against extreme sparse points inflating
    the mean and causing mass false-positive outlier marking.
    n_jobs=1 prevents CPU over-subscription when called from an outer Parallel.
    """
    k_actual = min(k + 1, X.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k_actual, algorithm='ball_tree',
                             metric='euclidean', n_jobs=1)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    densities  = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)
    threshold  = np.median(densities) * density_threshold
    return densities < threshold, distances


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class OptimizedGraphDensityCut:
    """Performance-optimized graph-based density cut detector."""

    def __init__(self, k_neighbors: int = 10, density_threshold: float = 0.3,
                 min_cluster_size: int = 5, contamination: float = 0.1,
                 n_jobs: int = -1, use_cache: bool = True):
        self.k_neighbors       = k_neighbors
        self.density_threshold = density_threshold
        self.min_cluster_size  = min_cluster_size
        self.contamination     = contamination
        self.n_jobs            = n_jobs
        self.use_cache         = use_cache
        self.node_densities    = None
        self.density_scores    = None
        self.timings: dict     = {}

    def ipv6_to_features_vectorized(self, ipv6_list: List[str]) -> np.ndarray:
        """Adaptive parallel feature extraction."""
        t0 = time.perf_counter()
        n  = len(ipv6_list)
        if n < _PARALLEL_THRESHOLD:
            features = np.zeros((n, _N_FEATURES), dtype=np.float32)
            for i, s in enumerate(ipv6_list):
                features[i] = _parse_one_address_graph(s)
        else:
            rows = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                delayed(_parse_one_address_graph)(s) for s in ipv6_list
            )
            features = np.array(rows, dtype=np.float32)
        self.timings['feature_extraction'] = time.perf_counter() - t0
        return features

    def build_knn_graph_fast(self, X: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """
        Build directed k-NN graph (ball_tree).
        Symmetrization omitted: BFS only needs reachability, halving construction cost.
        """
        t0 = time.perf_counter()
        n  = X.shape[0]
        k  = min(self.k_neighbors + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',
                                 metric='euclidean', n_jobs=self.n_jobs)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        row  = np.repeat(np.arange(n), k - 1)
        col  = indices[:, 1:].ravel()
        data = 1.0 / (distances[:, 1:].ravel() + 1e-10)
        adj  = csr_matrix((data, (row, col)), shape=(n, n))
        self.timings['graph_building'] = time.perf_counter() - t0
        return adj, distances

    def calculate_density_vectorized(self, distances: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)
        self.timings['density_computation'] = time.perf_counter() - t0
        return densities

    def fast_clustering(self, adjacency: csr_matrix,
                        densities: np.ndarray) -> np.ndarray:
        """BFS density-cut clustering using raw CSR arrays for O(1) neighbour lookup."""
        t0 = time.perf_counter()
        n  = adjacency.shape[0]
        labels    = np.full(n, -1, dtype=np.int32)
        threshold = np.median(densities) * self.density_threshold
        high      = np.where(densities >= threshold)[0]
        if len(high) == 0:
            self.timings['clustering'] = time.perf_counter() - t0
            return labels

        indptr  = adjacency.indptr
        col_idx = adjacency.indices
        seeds      = high[np.argsort(densities[high])[::-1]]
        visited    = np.zeros(n, dtype=bool)
        cluster_id = 0
        for seed in seeds:
            if visited[seed]:
                continue
            cluster = self._bfs_expand(indptr, col_idx, seed,
                                        densities, threshold, visited)
            if len(cluster) >= self.min_cluster_size:
                labels[cluster] = cluster_id
                cluster_id += 1
        self.timings['clustering'] = time.perf_counter() - t0
        return labels

    def _bfs_expand(self, indptr, col_idx, seed: int,
                    densities: np.ndarray, threshold: float,
                    visited: np.ndarray) -> List[int]:
        """BFS with deque (O(1) popleft) and direct CSR slice access."""
        cluster = []
        queue   = deque([seed])
        while queue:
            node = queue.popleft()
            if visited[node]:
                continue
            visited[node] = True
            cluster.append(node)
            for nb in col_idx[indptr[node]:indptr[node + 1]]:
                if not visited[nb] and densities[nb] >= threshold:
                    queue.append(int(nb))
        return cluster

    def fit_predict(self, ipv6_addresses: List[str],
                    verbose: bool = True) -> np.ndarray:
        wall_start = time.perf_counter()
        self.timings = {}
        n = len(ipv6_addresses)

        if verbose:
            print(f"  [1/4] Extracting features ({n} addresses)...")
        X = self.ipv6_to_features_vectorized(ipv6_addresses)
        if verbose:
            print(f"        {self.timings['feature_extraction']:.3f}s")

        t0 = time.perf_counter()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.timings['feature_scaling'] = time.perf_counter() - t0
        if verbose:
            print(f"  [1.5/4] Feature scaling: {self.timings['feature_scaling']:.3f}s")

        if verbose:
            print(f"  [2/4] Building k-NN graph (k={self.k_neighbors})...")
        adjacency, distances = self.build_knn_graph_fast(X)
        if verbose:
            print(f"        {self.timings['graph_building']:.3f}s")

        if verbose:
            print("  [3/4] Computing densities...")
        self.node_densities = self.calculate_density_vectorized(distances)
        if verbose:
            print(f"        {self.timings['density_computation']:.3f}s")

        if verbose:
            print("  [4/4] Clustering...")
        labels = self.fast_clustering(adjacency, self.node_densities)
        if verbose:
            print(f"        {self.timings['clustering']:.3f}s")

        predictions = np.where(labels == -1, -1, 1)

        # score_graph = normalize(density[v] < threshold): points below the threshold
        # receive scores in (0, 1] proportional to their anomaly degree; normal points
        # score exactly 0. Implements Algorithm 1 Line 2 normalize() semantics.
        density_threshold_val = np.median(self.node_densities) * self.density_threshold
        below = np.maximum(0.0, density_threshold_val - self.node_densities)
        self.density_scores = below / np.max(below) if np.max(below) > 0 else np.zeros_like(below)

        self.timings['total'] = time.perf_counter() - wall_start
        if verbose:
            n_out = int(np.sum(predictions == -1))
            print(f"\n  Results: {n_out}/{n} outliers ({n_out/n:.2%})")
            self._print_timing_table()
        return predictions

    def get_anomaly_scores(self) -> np.ndarray:
        if self.density_scores is None:
            raise ValueError("Call fit_predict() first")
        return self.density_scores

    def _print_timing_table(self):
        rows = [('Feature extraction', 'feature_extraction'),
                ('Feature scaling',    'feature_scaling'),
                ('k-NN graph build',   'graph_building'),
                ('Density computation','density_computation'),
                ('Clustering',         'clustering'),
                ('Total',              'total')]
        print("  ┌─────────────────────────┬──────────┐")
        print("  │ Stage                   │  Time(s) │")
        print("  ├─────────────────────────┼──────────┤")
        for label, key in rows:
            print(f"  │ {label:<23s} │ {self.timings.get(key,0):>8.3f} │")
        print("  └─────────────────────────┴──────────┘")

    def get_performance_stats(self) -> dict:
        return {f'time_{k}': v for k, v in self.timings.items()}


# ---------------------------------------------------------------------------
# Enhanced multi-scale variant
# ---------------------------------------------------------------------------

class EnhancedGraphDensityCutFast(OptimizedGraphDensityCut):
    """
    Multi-scale graph density cut with parallel scale processing.
    Reuses max-k distances from the parallel voting step — no extra kNN pass.
    """

    def __init__(self, multi_scale_k: List[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.multi_scale_k = multi_scale_k or [5, 10, 20]

    def fit_predict(self, ipv6_addresses: List[str],
                    verbose: bool = True) -> np.ndarray:
        wall_start = time.perf_counter()
        self.timings = {}
        n = len(ipv6_addresses)

        if verbose:
            print(f"  [1/3] Extracting features ({n} addresses)...")
        X = self.ipv6_to_features_vectorized(ipv6_addresses)
        if verbose:
            print(f"        {self.timings['feature_extraction']:.3f}s")

        t0 = time.perf_counter()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.timings['feature_scaling'] = time.perf_counter() - t0
        if verbose:
            print(f"  [1.5/3] Feature scaling: {self.timings['feature_scaling']:.3f}s")

        if verbose:
            print(f"  [2/3] Parallel multi-scale kNN (k={self.multi_scale_k})...")
        t0 = time.perf_counter()

        n_par = (len(self.multi_scale_k) if self.n_jobs < 0
                 else min(len(self.multi_scale_k), self.n_jobs))
        results = Parallel(n_jobs=n_par)(
            delayed(_process_scale)(X, k, self.density_threshold)
            for k in self.multi_scale_k
        )
        self.timings['multi_scale'] = time.perf_counter() - t0

        masks_list     = [r[0] for r in results]
        distances_list = [r[1] for r in results]

        # Unanimous vote: all scales must agree to label a point as outlier.
        outlier_votes = np.sum(masks_list, axis=0)
        predictions   = np.where(outlier_votes == len(self.multi_scale_k), -1, 1)

        if verbose:
            print(f"        {self.timings['multi_scale']:.3f}s")

        # Reuse distances from the max-k worker — no extra kNN pass.
        max_k_idx      = self.multi_scale_k.index(max(self.multi_scale_k))
        best_distances = distances_list[max_k_idx]
        self.node_densities = self.calculate_density_vectorized(best_distances)

        # score_graph = normalize(density[v] < threshold): implements Algorithm 1
        # Line 2 normalize() semantics. Normal points score exactly 0.
        density_threshold_val = np.median(self.node_densities) * self.density_threshold
        below = np.maximum(0.0, density_threshold_val - self.node_densities)
        self.density_scores = below / np.max(below) if np.max(below) > 0 else np.zeros_like(below)

        if verbose:
            print(f"  [3/3] Density scores from cached max-k distances  "
                  f"{self.timings['density_computation']:.3f}s")

        self.timings['total'] = time.perf_counter() - wall_start
        if verbose:
            n_out = int(np.sum(predictions == -1))
            print(f"\n  Results: {n_out}/{n} outliers ({n_out/n:.2%})")
            self._print_timing_table_enhanced()
        return predictions

    def _print_timing_table_enhanced(self):
        rows = [('Feature extraction',      'feature_extraction'),
                ('Feature scaling',         'feature_scaling'),
                ('Multi-scale kNN (par.)',   'multi_scale'),
                ('Density scores (cached)',  'density_computation'),
                ('Total',                    'total')]
        print("  ┌───────────────────────────────┬──────────┐")
        print("  │ Stage                         │  Time(s) │")
        print("  ├───────────────────────────────┼──────────┤")
        for label, key in rows:
            print(f"  │ {label:<29s} │ {self.timings.get(key,0):>8.3f} │")
        print("  └───────────────────────────────┴──────────┘")


if __name__ == "__main__":
    print("=== Optimized Graph Detector Test ===\n")
    test_addresses = [
        "2001:db8::1", "2001:db8::2", "2001:db8::3",
        "2001:db8::10", "2001:db8::11", "2001:db8::12",
        "2001:db8:1::1", "2001:db8:1::2",
        "fe80::1", "2001:0:0:0:0:0:0:1",
    ] * 20
    print(f"Test data: {len(test_addresses)} addresses\n")
    print("--- OptimizedGraphDensityCut ---")
    OptimizedGraphDensityCut(k_neighbors=5, density_threshold=0.5,
                              min_cluster_size=3, n_jobs=-1
                              ).fit_predict(test_addresses, verbose=True)
    print("\n--- EnhancedGraphDensityCutFast ---")
    EnhancedGraphDensityCutFast(multi_scale_k=[5, 10, 15], n_jobs=-1
                                 ).fit_predict(test_addresses, verbose=True)
