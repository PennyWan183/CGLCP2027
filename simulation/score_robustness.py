"""Unsafe-score robustness simulation and historical RNG warm-up.

The numerical routines are retained from the original simulation suite.
run_s1 is used only to restore the shared random stream before run_s3.
"""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any,Iterable
import numpy as np
from cglcp_methods import (
 cglcp_certified_interval,create_verification_certificate,global_rate_plugin_interval,
 hypergeometric_unsafe_upper_bound,local_uncorrected_interval,ordinary_conformal_rank,
 oracle_certificate_budget_interval,oracle_local_interval,oracle_safe_interval,sharp_contamination_rank,
)
from simulation.graph_design import ModularGraphScenario,build_modular_graph_scenario
METHOD_ORDER=['local_uncorrected','global_rate_plugin','cglcp_certified','oracle_local_alpha','oracle_certificate_budget','oracle_safe']
ALL_METHOD_ORDER=list(METHOD_ORDER)

def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _unsafe_scores(
    rng: np.random.Generator,
    count: int,
    mechanism: str,
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=float)
    scales = {
        "benign": 1.0,
        "moderate": 0.5,
        "severe": 0.1,
    }
    if mechanism == "adversarial":
        return np.zeros(count, dtype=float)
    if mechanism not in scales:
        raise ValueError(f"Unknown unsafe-score mechanism: {mechanism}")
    return np.abs(rng.normal(0.0, scales[mechanism], size=count))

def generate_graph_derived_pool(
    rng: np.random.Generator,
    *,
    graph: ModularGraphScenario,
    mechanism: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    labels = np.asarray(graph.local_unsafe_labels, dtype=np.int8)
    pool_size = labels.size
    unsafe_count = int(labels.sum())
    safe_count = pool_size - unsafe_count
    safe_scores = np.abs(rng.normal(0.0, 1.0, size=safe_count))
    unsafe_scores = _unsafe_scores(rng, unsafe_count, mechanism)
    scores = np.empty(pool_size, dtype=float)
    scores[labels == 0] = safe_scores
    scores[labels == 1] = unsafe_scores
    test_score = float(abs(rng.normal(0.0, 1.0)))
    return scores, labels, test_score

def evaluate_one_pool(
    scores: np.ndarray,
    unsafe_labels: np.ndarray,
    test_score: float,
    *,
    verification_size: int,
    global_rate: float,
    alpha: float,
    rho: float,
    rng: np.random.Generator,
    sampled_indices: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    true_b = int(unsafe_labels.sum())
    if sampled_indices is None:
        certificate = create_verification_certificate(
            unsafe_labels,
            verification_size,
            rho,
            random_state=rng,
        )
        verified_unsafe_count = certificate.verified_unsafe_count
        unsafe_upper_count = certificate.unsafe_upper_count
    else:
        sampled = np.asarray(sampled_indices, dtype=int)
        if sampled.ndim != 1 or sampled.size != verification_size:
            raise ValueError("sampled_indices must match verification_size")
        if np.unique(sampled).size != sampled.size:
            raise ValueError("sampled_indices must be sampled without replacement")
        if sampled.min() < 0 or sampled.max() >= unsafe_labels.size:
            raise ValueError("sampled_indices lie outside the local pool")
        verified_unsafe_count = int(unsafe_labels[sampled].sum())
        unsafe_upper_count = hypergeometric_unsafe_upper_bound(
            int(unsafe_labels.size),
            verification_size,
            verified_unsafe_count,
            rho,
        )
    intervals = [
        local_uncorrected_interval(scores, 0.0, alpha=alpha),
        global_rate_plugin_interval(scores, 0.0, global_rate, alpha=alpha),
        cglcp_certified_interval(
            scores,
            0.0,
            unsafe_upper_count,
            alpha=alpha,
            rho=rho,
        ),
        oracle_local_interval(scores, 0.0, true_b, alpha=alpha),
        oracle_certificate_budget_interval(
            scores,
            0.0,
            true_b,
            alpha=alpha,
            rho=rho,
        ),
        oracle_safe_interval(
            scores,
            unsafe_labels,
            0.0,
            alpha=alpha,
        ),
    ]
    method_names = METHOD_ORDER
    method_rows: list[dict[str, Any]] = []
    for method, interval in zip(method_names, intervals):
        finite = interval.finite
        method_rows.append(
            {
                "method": method,
                "covered": int(test_score <= interval.quantile),
                "finite": int(finite),
                "width": float(interval.width) if finite else "inf",
                "rank": interval.rank,
                "quantile": interval.quantile if finite else "inf",
                "unsafe_count_used": interval.unsafe_count_used,
                "conditional_alpha": interval.conditional_alpha,
            }
        )
    certificate_row = {
        "pool_size": int(scores.size),
        "true_unsafe_count": true_b,
        "verification_size": verification_size,
        "verified_unsafe_count": verified_unsafe_count,
        "unsafe_upper_count": unsafe_upper_count,
        "certificate_valid": int(true_b <= unsafe_upper_count),
        "certificate_excess": unsafe_upper_count - true_b,
        "test_score": test_score,
    }
    return method_rows, certificate_row

def summarize_replicates(
    rows: list[dict[str, Any]],
    group_keys: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in grouped.items():
        finite_widths = [
            float(row["width"])
            for row in group
            if int(row["finite"]) == 1
        ]
        finite_ranks = [
            int(row["rank"])
            for row in group
            if int(row["finite"]) == 1
        ]
        summary = dict(zip(group_keys, key))
        summary.update(
            {
                "repetitions": len(group),
                "coverage": float(np.mean([int(r["covered"]) for r in group])),
                "finite_rate": float(np.mean([int(r["finite"]) for r in group])),
                "median_finite_width": (
                    float(np.median(finite_widths)) if finite_widths else "inf"
                ),
                "median_finite_rank": (
                    float(np.median(finite_ranks)) if finite_ranks else "inf"
                ),
                "certificate_validity": float(
                    np.mean([int(r["certificate_valid"]) for r in group])
                ),
                "mean_certificate_excess": float(
                    np.mean([float(r["certificate_excess"]) for r in group])
                ),
            }
        )
        if all("rank_inflation" in row for row in group):
            summary["median_rank_inflation"] = float(
                np.median([float(row["rank_inflation"]) for row in group])
            )
        summaries.append(summary)
    order_index = {name: index for index, name in enumerate(ALL_METHOD_ORDER)}
    summaries.sort(
        key=lambda row: tuple(
            order_index.get(row[key], row[key]) if key == "method" else row[key]
            for key in group_keys
        )
    )
    return summaries

def run_s1(
    rng: np.random.Generator,
    repetitions: int,
    *,
    alpha: float,
    rho: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verification_size = 30
    scenarios = {
        "distant": 0,
        "near_clean": 1,
        "spread": 3,
        "matched_global": 6,
        "moderate_local": 12,
        "high_local": 15,
        "concentrated": 24,
    }
    rows: list[dict[str, Any]] = []
    for scenario, true_b in scenarios.items():
        graph = build_modular_graph_scenario(
            name=f"s1_{scenario}",
            pool_size=60,
            max_local_unsafe=24,
            local_unsafe=true_b,
            total_deletions=24,
        )
        for replicate in range(repetitions):
            scores, labels, test_score = generate_graph_derived_pool(
                rng,
                graph=graph,
                mechanism="severe",
            )
            method_rows, certificate = evaluate_one_pool(
                scores,
                labels,
                test_score,
                verification_size=verification_size,
                global_rate=graph.global_contamination_rate,
                alpha=alpha,
                rho=rho,
                rng=rng,
            )
            shared = {
                "scenario": scenario,
                "contamination_stratum": contamination_stratum(
                    graph.true_local_unsafe_count / graph.pool_size
                ),
                "replicate": replicate,
                **graph.metadata(),
                **certificate,
            }
            for method_row in method_rows:
                rows.append({**shared, **method_row})
    return rows, summarize_replicates(
        rows,
        ["scenario", "local_contamination", "method"],
    )

def contamination_stratum(delta: float) -> str:
    """Return the prespecified local-contamination bin used in reporting."""

    if delta < 0.05:
        return "0-5%"
    if delta < 0.15:
        return "5-15%"
    if delta <= 0.25:
        return "15-25%"
    return ">25%"

def run_s3(
    rng: np.random.Generator,
    repetitions: int,
    *,
    alpha: float,
    rho: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verification_size = 40
    graph = build_modular_graph_scenario(
        name="s3_b20",
        pool_size=100,
        max_local_unsafe=20,
        local_unsafe=20,
        total_deletions=20,
    )
    mechanisms = ["benign", "moderate", "severe", "adversarial"]
    rows: list[dict[str, Any]] = []
    for mechanism in mechanisms:
        for replicate in range(repetitions):
            scores, labels, test_score = generate_graph_derived_pool(
                rng,
                graph=graph,
                mechanism=mechanism,
            )
            method_rows, certificate = evaluate_one_pool(
                scores,
                labels,
                test_score,
                verification_size=verification_size,
                global_rate=graph.global_contamination_rate,
                alpha=alpha,
                rho=rho,
                rng=rng,
            )
            shared = {
                "mechanism": mechanism,
                "replicate": replicate,
                **graph.metadata(),
                **certificate,
            }
            for method_row in method_rows:
                rows.append({**shared, **method_row})
    return rows, summarize_replicates(rows, ["mechanism", "method"])
