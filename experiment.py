"""
Experiment Runner
"""

import numpy as np
import time
import json
import os
from typing import List, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from config import ExperimentConfig, get_config, print_config_summary


class Experiment:
    """Experiment manager."""

    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_single_experiment(self, config: ExperimentConfig,
                              ipv6_addresses: List[str],
                              verbose: bool = True) -> Dict:
        """Run a single preprocessing experiment and return a result dict."""
        if verbose:
            print_config_summary(config)
            print("Starting experiment...")

        wall_start = time.perf_counter()
        result = {
            'config_name': config.name,
            'config':      config.to_dict(),
            'n_samples':   len(ipv6_addresses),
            'timestamp':   datetime.now().isoformat(),
        }

        try:
            if config.autoencoder.enabled and config.graph.enabled:
                predictions, scores = self._run_ensemble(config, ipv6_addresses, verbose)
            elif config.autoencoder.enabled:
                predictions, scores = self._run_autoencoder_only(config, ipv6_addresses, verbose)
            elif config.graph.enabled:
                predictions, scores = self._run_graph_only(config, ipv6_addresses, verbose)
            else:
                raise ValueError("At least one algorithm must be enabled")

            n_out   = int(np.sum(predictions == -1))
            n_norm  = int(np.sum(predictions ==  1))
            elapsed = time.perf_counter() - wall_start

            result.update({
                'success':        True,
                'n_outliers':     n_out,
                'n_normal':       n_norm,
                'outlier_ratio':  float(n_out / len(ipv6_addresses)),
                'predictions':    predictions.tolist(),
                'scores':         scores.tolist() if scores is not None else None,
                'execution_time': elapsed,
            })

            if verbose:
                print(f"\n{'─'*50}")
                print(f"  Experiment summary")
                print(f"{'─'*50}")
                print(f"  Total     : {len(ipv6_addresses)}")
                print(f"  Normal    : {n_norm}")
                print(f"  Outliers  : {n_out} ({n_out/len(ipv6_addresses):.2%})")
                print(f"  Wall time : {elapsed:.3f}s")
                print(f"{'─'*50}")

        except Exception as e:
            elapsed = time.perf_counter() - wall_start
            result.update({'success': False, 'error': str(e), 'execution_time': elapsed})
            if verbose:
                print(f"\nExperiment failed after {elapsed:.3f}s: {e}")

        self._save_result(result, ipv6_addresses)
        return result

    # ------------------------------------------------------------------
    # Detector runners
    # ------------------------------------------------------------------

    def _run_ensemble(self, config: ExperimentConfig,
                      ipv6_addresses: List[str], verbose: bool) -> tuple:
        """
        Run AE + Graph concurrently using ThreadPoolExecutor.
        AE training (GPU/CPU) and Graph kNN (CPU) use different resource pools
        and can genuinely overlap.
        """
        from optimized_autoencoder import OptimizedAutoencoderDetector
        from optimized_graph import OptimizedGraphDensityCut, EnhancedGraphDensityCutFast

        ae = OptimizedAutoencoderDetector(
            latent_dim=config.autoencoder.latent_dim,
            hidden_dims=config.autoencoder.hidden_dims,
            learning_rate=config.autoencoder.learning_rate,
            batch_size=config.autoencoder.batch_size,
            epochs=config.autoencoder.epochs,
            contamination=config.autoencoder.contamination,
            device=config.autoencoder.device,
            use_amp=config.autoencoder.use_mixed_precision,
        )

        graph = (
            EnhancedGraphDensityCutFast(
                multi_scale_k=config.graph.multi_scale_k,
                density_threshold=config.graph.density_threshold,
                min_cluster_size=config.graph.min_cluster_size,
                contamination=config.graph.contamination,
                n_jobs=config.graph.n_jobs,
            ) if config.graph.use_enhanced else
            OptimizedGraphDensityCut(
                k_neighbors=config.graph.k_neighbors,
                density_threshold=config.graph.density_threshold,
                min_cluster_size=config.graph.min_cluster_size,
                contamination=config.graph.contamination,
                n_jobs=config.graph.n_jobs,
            )
        )

        weights = [config.ensemble.ae_weight, config.ensemble.graph_weight]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        def run_ae():
            t0 = time.perf_counter()
            ae.fit(ipv6_addresses, verbose=verbose)
            pred   = ae.predict(ipv6_addresses, verbose=verbose)
            scores = ae.get_anomaly_scores()
            return pred, scores, time.perf_counter() - t0

        def run_graph():
            t0 = time.perf_counter()
            pred   = graph.fit_predict(ipv6_addresses, verbose=verbose)
            scores = graph.get_anomaly_scores()
            return pred, scores, time.perf_counter() - t0

        if verbose:
            print(f"\n{'='*50}")
            print("  Running AE + Graph concurrently...")
            print(f"{'='*50}")

        t_ensemble = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_ae    = pool.submit(run_ae)
            future_graph = pool.submit(run_graph)
            ae_pred,    ae_scores,    ae_time = future_ae.result()
            graph_pred, graph_scores, gr_time = future_graph.result()
        elapsed_ensemble = time.perf_counter() - t_ensemble

        if verbose:
            print(f"\n  AE    finished in {ae_time:.3f}s  "
                  f"({int(np.sum(ae_pred==-1))} outliers)")
            print(f"  Graph finished in {gr_time:.3f}s  "
                  f"({int(np.sum(graph_pred==-1))} outliers)")
            print(f"  Concurrent wall time: {elapsed_ensemble:.3f}s  "
                  f"(vs serial ~{ae_time+gr_time:.3f}s)")
            print(f"\n  Fusing ({config.ensemble.voting_strategy} voting, "
                  f"weights={[f'{w:.0%}' for w in weights]})...")

        all_predictions = np.array([ae_pred, graph_pred])
        all_scores      = np.array([ae_scores, graph_scores])

        if config.ensemble.voting_strategy == 'hard':
            votes         = np.sum(all_predictions == -1, axis=0)
            ensemble_pred = np.where(votes > 1, -1, 1)
            ensemble_sc   = np.mean(all_scores, axis=0)

        elif config.ensemble.voting_strategy == 'weighted':
            w_votes       = sum(all_predictions[i] * weights[i] for i in range(2))
            ensemble_pred = np.where(w_votes < 0, -1, 1)
            ensemble_sc   = np.average(all_scores, axis=0, weights=weights)

        else:  # soft (default)
            # Both scores implement Algorithm 1's normalize() semantics:
            #   score_ae    = normalize(reconstruction_error > P95)
            #   score_graph = normalize(density < 0.3 x median_density)
            # score_graph is sparse: normal points score exactly 0, anomalous
            # points score in (0, 1] proportional to their anomaly degree.
            norm = []
            for s in all_scores:
                lo, hi = np.min(s), np.max(s)
                norm.append((s - lo) / (hi - lo) if hi > lo else np.zeros_like(s))
            ensemble_sc   = np.average(np.array(norm), axis=0, weights=weights)
            ensemble_pred = np.where(ensemble_sc > config.ensemble.threshold, -1, 1)

        return ensemble_pred, ensemble_sc

    def _run_autoencoder_only(self, config: ExperimentConfig,
                               ipv6_addresses: List[str], verbose: bool) -> tuple:
        from optimized_autoencoder import OptimizedAutoencoderDetector
        det = OptimizedAutoencoderDetector(
            latent_dim=config.autoencoder.latent_dim,
            hidden_dims=config.autoencoder.hidden_dims,
            learning_rate=config.autoencoder.learning_rate,
            batch_size=config.autoencoder.batch_size,
            epochs=config.autoencoder.epochs,
            contamination=config.autoencoder.contamination,
            device=config.autoencoder.device,
            use_amp=config.autoencoder.use_mixed_precision,
        )
        det.fit(ipv6_addresses, verbose=verbose)
        pred   = det.predict(ipv6_addresses, verbose=verbose)
        scores = det.get_anomaly_scores()
        return pred, scores

    def _run_graph_only(self, config: ExperimentConfig,
                        ipv6_addresses: List[str], verbose: bool) -> tuple:
        from optimized_graph import OptimizedGraphDensityCut, EnhancedGraphDensityCutFast
        det = (
            EnhancedGraphDensityCutFast(
                multi_scale_k=config.graph.multi_scale_k,
                density_threshold=config.graph.density_threshold,
                min_cluster_size=config.graph.min_cluster_size,
                contamination=config.graph.contamination,
                n_jobs=config.graph.n_jobs,
            ) if config.graph.use_enhanced else
            OptimizedGraphDensityCut(
                k_neighbors=config.graph.k_neighbors,
                density_threshold=config.graph.density_threshold,
                min_cluster_size=config.graph.min_cluster_size,
                contamination=config.graph.contamination,
                n_jobs=config.graph.n_jobs,
            )
        )
        pred   = det.fit_predict(ipv6_addresses, verbose=verbose)
        scores = det.get_anomaly_scores()
        return pred, scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_result(self, result: Dict, ipv6_addresses: List[str] = None):
        """Write per-config outputs to output_dir/<config_name>/."""
        config_name = result['config_name']
        config_dir  = os.path.join(self.output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        save_result = result.copy()
        if 'predictions' in save_result:
            save_result['predictions'] = f"<{len(save_result['predictions'])} values>"
        if save_result.get('scores'):
            save_result['scores'] = f"<{len(save_result['scores'])} values>"
        with open(os.path.join(config_dir, f"{config_name}_stats.json"), 'w') as f:
            json.dump(save_result, f, indent=2)

        if not (ipv6_addresses and result.get('success') and 'predictions' in result):
            return

        predictions   = np.array(result['predictions'])
        normal_addrs  = [a for a, p in zip(ipv6_addresses, predictions) if p ==  1]
        outlier_addrs = [a for a, p in zip(ipv6_addresses, predictions) if p == -1]

        with open(os.path.join(config_dir, 'normal_seeds.txt'),  'w') as f:
            f.writelines(a + '\n' for a in normal_addrs)
        with open(os.path.join(config_dir, 'outlier_seeds.txt'), 'w') as f:
            f.writelines(a + '\n' for a in outlier_addrs)

        if result.get('scores') is not None:
            with open(os.path.join(config_dir, 'anomaly_scores.txt'), 'w') as f:
                f.write("# IPv6_Address\tAnomaly_Score\tPrediction\n")
                f.writelines(
                    f"{addr}\t{score:.6f}\t{'outlier' if pred == -1 else 'normal'}\n"
                    for addr, score, pred in zip(
                        ipv6_addresses, result['scores'], predictions)
                )

        print(f"  Saved -> {config_dir}/")
        print(f"    normal_seeds.txt  ({len(normal_addrs)} addresses)")
        print(f"    outlier_seeds.txt ({len(outlier_addrs)} addresses)")
        print(f"    {config_name}_stats.json")
        if result.get('scores') is not None:
            print(f"    anomaly_scores.txt")
