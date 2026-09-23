#!/usr/bin/env python3
"""Core local-necessity simulation for CGLCP.

The experiment holds the global working-graph false-safe rate at 0.10 and
moves the same number of missing causal edges toward or away from one query.
It compares pooling, graph localization, contamination correction, and a
same-budget local oracle on shared score draws.

``PCS Corrected`` denotes the corrected selective conformal procedure in
*Partial Causal Structure Learning for Valid Selective
Conformal Inference under Interventions*.  It is not presented as an acronym
used by that paper.  In this controlled experiment the selector is held fixed
to the working-graph predicted-safe set so that the comparison isolates global
versus local pooling and the paper's additive contamination correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cglcp_methods import (  # noqa: E402
    cglcp_certified_interval,
    create_verification_certificate,
    global_rate_plugin_interval,
    oracle_certificate_budget_interval,
    ordinary_conformal_rank,
)
from baselines.graph_lcp import graph_lcp_thresholds  # noqa: E402
from simulation.graph_design import (  # noqa: E402
    ModularGraphScenario,
    build_modular_graph_scenario,
)


METHODS = [
    "pooled_cp",
    "global_rate_plugin",
    "graph_lcp",
    "pcs_sc_corrected",
    "localized_pcs_sc_corrected",
    "cglcp",
    "oracle_local_rho",
]

LABELS = {
    "pooled_cp": "Pooled CP",
    "global_rate_plugin": "Global-rate plug-in",
    "graph_lcp": "GraphLCP",
    "pcs_sc_corrected": "PCS Corrected",
    "localized_pcs_sc_corrected": "Localized PCS Corrected",
    "cglcp": "CGLCP",
    "oracle_local_rho": r"Oracle local ($\rho$)",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=0.02)
    parser.add_argument("--audit-size", type=int, default=30)
    parser.add_argument("--lcp-bandwidth", type=float, default=4.0)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Rebuild summaries from an existing replicates.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/local_necessity"),
    )
    return parser.parse_args()


def directed_adjacency(nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]):
    adjacency = {node: set() for node in nodes}
    for source, target in edges:
        adjacency[source].add(target)
    return adjacency


def undirected_adjacency(nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]):
    adjacency = {node: set() for node in nodes}
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def can_reach(adjacency: dict[str, set[str]], source: str, target: str) -> bool:
    if source == target:
        return True
    seen = {source}
    stack = [source]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor == target:
                return True
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return False


def bfs_distances(adjacency: dict[str, set[str]], source: str) -> dict[str, int]:
    result = {source: 0}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in result:
                result[neighbor] = result[current] + 1
                queue.append(neighbor)
    return result


def graph_distance_arrays(
    graph: ModularGraphScenario,
    calibration: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    adjacency = undirected_adjacency(graph.nodes, graph.working_edges)
    n = len(calibration)
    matrix = np.full((n, n), np.inf, dtype=float)
    for row, node in enumerate(calibration):
        distances = bfs_distances(adjacency, node)
        matrix[row] = [distances.get(other, math.inf) for other in calibration]
    query_distance = bfs_distances(adjacency, graph.query)
    query = np.asarray(
        [[query_distance.get(node, math.inf) for node in calibration]],
        dtype=float,
    )
    return matrix, query


def additive_corrected_threshold(
    scores: np.ndarray,
    unsafe_count: int,
    alpha: float,
) -> tuple[float, int, float]:
    """Corollary-1 additive correction in the partial-structure paper."""
    n = len(scores)
    penalty = unsafe_count / (n - unsafe_count + 1)
    effective_alpha = alpha - penalty
    if effective_alpha <= 0:
        return math.inf, n + 1, effective_alpha
    rank = ordinary_conformal_rank(n, effective_alpha)
    if rank > n:
        return math.inf, rank, effective_alpha
    return float(np.partition(scores, rank - 1)[rank - 1]), rank, effective_alpha


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    result: list[dict] = []
    for local_rate in sorted({float(row["local_contamination"]) for row in rows}):
        for method in METHODS:
            group = [
                row for row in rows
                if float(row["local_contamination"]) == local_rate
                and row["method"] == method
            ]
            finite = [row for row in group if int(row["finite"]) == 1]
            def wilson(successes: int, total: int) -> tuple[float, float]:
                if total == 0:
                    return math.nan, math.nan
                z = 1.959963984540054
                p = successes / total
                denominator = 1 + z * z / total
                center = (p + z * z / (2 * total)) / denominator
                radius = z * math.sqrt(
                    p * (1 - p) / total + z * z / (4 * total * total)
                ) / denominator
                return center - radius, center + radius

            coverage_ci = wilson(sum(int(row["covered"]) for row in group), len(group))
            finite_ci = wilson(len(finite), len(group))
            finite_coverage_ci = wilson(
                sum(int(row["covered"]) for row in finite), len(finite)
            )
            result.append(
                {
                    "local_contamination": local_rate,
                    "method": method,
                    "method_label": LABELS[method],
                    "repetitions": len(group),
                    "coverage": float(np.mean([int(row["covered"]) for row in group])),
                    "coverage_ci95_lower": coverage_ci[0],
                    "coverage_ci95_upper": coverage_ci[1],
                    "finite_conditional_coverage": (
                        float(np.mean([int(row["covered"]) for row in finite]))
                        if finite else math.nan
                    ),
                    "finite_conditional_coverage_ci95_lower": finite_coverage_ci[0],
                    "finite_conditional_coverage_ci95_upper": finite_coverage_ci[1],
                    "finite_rate": len(finite) / len(group),
                    "finite_rate_ci95_lower": finite_ci[0],
                    "finite_rate_ci95_upper": finite_ci[1],
                    "median_finite_width": (
                        float(np.median([float(row["width"]) for row in finite]))
                        if finite else math.nan
                    ),
                    "reference_bound_success": (
                        float(np.mean([
                            float(row["reference_bound_success"])
                            for row in group
                            if row["reference_bound_success"] != ""
                        ]))
                        if any(row["reference_bound_success"] != "" for row in group)
                        else math.nan
                    ),
                }
            )
    return result






def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_replicates = args.output_dir / "replicates.csv"
    if args.reuse_existing:
        if not existing_replicates.exists():
            raise FileNotFoundError(existing_replicates)
        with existing_replicates.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        summary = summarize(rows)
        write_csv(args.output_dir / "summary.csv", summary)
        return
    rng = np.random.default_rng(args.seed)
    scenario_counts = [1, 3, 6, 12, 15, 24]
    rows: list[dict] = []

    for local_b in scenario_counts:
        graph = build_modular_graph_scenario(
            name=f"aligned_local_b{local_b}",
            pool_size=60,
            max_local_unsafe=24,
            local_unsafe=local_b,
            total_deletions=24,
        )
        reference = directed_adjacency(graph.nodes, graph.reference_edges)
        all_calibration = graph.calibration_interventions
        all_labels = np.asarray(
            [int(can_reach(reference, node, graph.target)) for node in all_calibration],
            dtype=int,
        )
        all_index = {node: index for index, node in enumerate(all_calibration)}
        graph_calibration = graph.predicted_safe_interventions
        predicted_indices = np.asarray([all_index[node] for node in graph_calibration], dtype=int)
        graph_calibration_index = {
            node: index for index, node in enumerate(graph_calibration)
        }
        graph_labels = np.asarray(
            [int(can_reach(reference, node, graph.target)) for node in graph_calibration],
            dtype=int,
        )
        local_indices = np.asarray(
            [graph_calibration_index[node] for node in graph.local_pool], dtype=int
        )
        cal_distance, query_distance = graph_distance_arrays(graph, all_calibration)
        global_b = int(graph_labels.sum())

        for replicate in range(args.repetitions):
            all_scores = np.empty(len(all_calibration), dtype=float)
            all_scores[all_labels == 0] = np.abs(
                rng.normal(0.0, 1.0, size=int((all_labels == 0).sum()))
            )
            all_scores[all_labels == 1] = np.abs(
                rng.normal(0.0, 0.1, size=int((all_labels == 1).sum()))
            )
            scores = all_scores[predicted_indices]
            local_scores = scores[local_indices]
            local_labels = graph_labels[local_indices]
            test_score = float(abs(rng.normal()))
            certificate = create_verification_certificate(
                local_labels,
                args.audit_size,
                args.rho,
                random_state=rng,
            )
            local_b_true = int(local_labels.sum())

            global_interval = global_rate_plugin_interval(
                local_scores, 0.0, graph.global_contamination_rate, alpha=args.alpha
            )
            cglcp_interval = cglcp_certified_interval(
                local_scores,
                0.0,
                certificate.unsafe_upper_count,
                alpha=args.alpha,
                rho=args.rho,
            )
            oracle_interval = oracle_certificate_budget_interval(
                local_scores,
                0.0,
                local_b_true,
                alpha=args.alpha,
                rho=args.rho,
            )
            pcs_global_q, pcs_global_rank, _ = additive_corrected_threshold(
                scores, global_b, args.alpha
            )
            pcs_local_q, pcs_local_rank, _ = additive_corrected_threshold(
                local_scores, local_b_true, args.alpha
            )
            pooled_rank = ordinary_conformal_rank(len(all_scores), args.alpha)
            pooled_q = (
                float(np.partition(all_scores, pooled_rank - 1)[pooled_rank - 1])
                if pooled_rank <= len(all_scores) else math.inf
            )
            graph_lcp_q = float(
                graph_lcp_thresholds(
                    all_scores,
                    cal_distance,
                    query_distance,
                    args.lcp_bandwidth,
                    miscoverage=args.alpha,
                )[0][0]
            )

            method_values = {
                "pooled_cp": (pooled_q, pooled_rank, ""),
                "global_rate_plugin": (
                    global_interval.quantile,
                    global_interval.rank,
                    int(local_b_true <= math.ceil(len(local_scores) * 0.10)),
                ),
                "graph_lcp": (graph_lcp_q, math.nan, ""),
                "pcs_sc_corrected": (pcs_global_q, pcs_global_rank, 1),
                "localized_pcs_sc_corrected": (pcs_local_q, pcs_local_rank, 1),
                "cglcp": (
                    cglcp_interval.quantile,
                    cglcp_interval.rank,
                    int(local_b_true <= certificate.unsafe_upper_count),
                ),
                "oracle_local_rho": (oracle_interval.quantile, oracle_interval.rank, 1),
            }
            for method, (quantile, rank, bound_success) in method_values.items():
                finite = math.isfinite(float(quantile))
                rows.append(
                    {
                        "local_contamination": local_b_true / len(local_scores),
                        "global_contamination": graph.global_contamination_rate,
                        "scenario_local_b": local_b,
                        "replicate": replicate,
                        "method": method,
                        "method_label": LABELS[method],
                        "covered": int(test_score <= float(quantile)),
                        "finite": int(finite),
                        "width": 2.0 * float(quantile) if finite else "inf",
                        "rank": rank,
                        "true_local_b": local_b_true,
                        "audit_size": args.audit_size,
                        "audited_non_safe": certificate.verified_unsafe_count,
                        "b_bar": certificate.unsafe_upper_count,
                        "reference_bound_success": bound_success,
                    }
                )

    summary = summarize(rows)
    write_csv(args.output_dir / "replicates.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    config = {
        **vars(args),
        "output_dir": str(args.output_dir),
        "scenario_local_b": scenario_counts,
        "global_contamination": 0.10,
        "pool_size": 60,
        "unsafe_score_scale": 0.10,
        "pcs_sc_name_note": (
            "PCS-SC is a manuscript shorthand introduced by us; the source paper "
            "labels its procedures Estimated and Corrected."
        ),
        "pcs_sc_simulation_selector": (
            "working-graph predicted-safe set; held fixed to isolate correction and localization"
        ),
        "pooled_and_graph_lcp_calibration_set": "all calibration interventions",
        "pcs_calibration_set": "working-graph predicted-safe interventions",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
