"""Faithful Python port of Guan's localized conformal calibration step.

The implementation follows ``LCP_construction_path_distance`` and
``LCP_alpha`` in the authors' public R/Rcpp implementation.  In particular,
it calibrates the local quantile level jointly over the augmented sample; it
is not the generally invalid shortcut that simply takes a weighted empirical
quantile at level ``1 - alpha``.

Reference implementation:
    https://github.com/LeyingGuan/LCP
"""

from __future__ import annotations

import math

import numpy as np


def _kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    distances = np.asarray(distances, dtype=float)
    weights = np.exp(-distances / bandwidth)
    weights[~np.isfinite(distances)] = 0.0
    return weights


def _id_low_search(values: np.ndarray) -> np.ndarray:
    """Port of the official Rcpp ``id_low_search`` helper."""
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), -1, dtype=int)
    pointer = -1
    for idx, value in enumerate(values):
        while pointer + 1 < len(values) and values[pointer + 1] < value:
            pointer += 1
        result[idx] = pointer
    return result


def _construction_path_distance(
    values: np.ndarray,
    id_low: np.ndarray,
    q_low: np.ndarray,
    q_total: np.ndarray,
    test_to_cal_weight: np.ndarray,
    cal_to_test_weight: np.ndarray,
) -> np.ndarray:
    """Return the official LCP acceptance-level matrix.

    ``values`` is the sorted calibration-score vector extended by infinity.
    Rows correspond to test points and columns to candidate score thresholds.
    """
    n_augmented = len(values)
    n_calibration = n_augmented - 1
    n_test = test_to_cal_weight.shape[0]

    theta = np.zeros((n_test, n_augmented), dtype=float)
    for column in range(1, n_augmented):
        theta[:, column] = theta[:, column - 1]
        start = int(id_low[column - 1] + 1)
        stop = int(id_low[column] + 1)
        if stop > start:
            theta[:, column] += test_to_cal_weight[:, start:stop].sum(axis=1)

    test_total = theta[:, n_calibration]
    alphas = np.zeros_like(theta)
    for test_idx in range(n_test):
        normalizer = q_total + cal_to_test_weight[:, test_idx]
        p1 = np.divide(
            q_low + cal_to_test_weight[:, test_idx],
            normalizer,
            out=np.zeros_like(q_low),
            where=normalizer > 0,
        )
        p2 = np.divide(
            q_low,
            normalizer,
            out=np.zeros_like(q_low),
            where=normalizer > 0,
        )
        theta_normalized = theta[test_idx] / (test_total[test_idx] + 1.0)
        theta_cal = theta_normalized[:n_calibration]

        a1 = np.sort(p1[p1 < theta_cal])
        a2 = np.sort(p2[p2 >= theta_cal])
        a3 = np.sort(id_low[:n_calibration][(p2 < theta_cal) & (p1 >= theta_cal)])

        # The three official while loops count strict inequalities.  Using
        # searchsorted(side="left") gives the same count, including ties.
        alphas[test_idx] = (
            np.searchsorted(a1, theta_normalized, side="left")
            + np.searchsorted(a2, theta_normalized, side="left")
            + np.searchsorted(a3, id_low, side="left")
        ) / (n_calibration + 1.0)
    return alphas


def graph_lcp_thresholds(
    calibration_scores: np.ndarray,
    calibration_distances: np.ndarray,
    test_distances: np.ndarray,
    bandwidth: float,
    miscoverage: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact LCP score thresholds for one or more graph queries.

    Parameters
    ----------
    calibration_scores:
        One nonconformity score per calibration intervention.
    calibration_distances:
        Symmetric calibration-by-calibration graph distance matrix.
    test_distances:
        Test-by-calibration graph distance matrix.  The working graph is
        undirected for localization, so its transpose supplies reverse
        calibration-to-test distances.
    bandwidth:
        Exponential localizer scale in ``exp(-distance / bandwidth)``.
    miscoverage:
        Desired error probability (``alpha`` in conformal notation).

    Returns
    -------
    thresholds, acceptance_levels:
        LCP thresholds and the full acceptance-level matrix used to obtain
        them.  The latter is retained for numerical auditing.
    """
    scores = np.asarray(calibration_scores, dtype=float)
    cal_distance = np.asarray(calibration_distances, dtype=float)
    test_distance = np.atleast_2d(np.asarray(test_distances, dtype=float))
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("calibration_scores must be a finite one-dimensional array")
    n = len(scores)
    if cal_distance.shape != (n, n):
        raise ValueError("calibration_distances has incompatible shape")
    if test_distance.shape[1] != n:
        raise ValueError("test_distances has incompatible shape")
    if not 0 < miscoverage < 1:
        raise ValueError("miscoverage must lie in (0, 1)")
    if n == 0:
        return np.full(test_distance.shape[0], math.inf), np.empty((test_distance.shape[0], 1))

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    values = np.concatenate([sorted_scores, [math.inf]])
    sorted_cal_distance = cal_distance[np.ix_(order, order)]
    sorted_test_distance = test_distance[:, order]

    cal_weight = _kernel(sorted_cal_distance, bandwidth)
    test_to_cal_weight = _kernel(sorted_test_distance, bandwidth)
    cal_to_test_weight = test_to_cal_weight.T
    cumulative_weight = np.cumsum(cal_weight, axis=1)
    id_low = _id_low_search(values)
    q_low = np.zeros(n, dtype=float)
    for row, lower_idx in enumerate(id_low[:n]):
        if lower_idx >= 0:
            q_low[row] = cumulative_weight[row, lower_idx]
    q_total = cumulative_weight[:, -1]

    alphas = _construction_path_distance(
        values,
        id_low,
        q_low,
        q_total,
        test_to_cal_weight,
        cal_to_test_weight,
    )
    target_coverage = 1.0 - miscoverage
    thresholds = np.full(test_distance.shape[0], math.inf, dtype=float)
    for row in range(test_distance.shape[0]):
        accepted = np.flatnonzero(alphas[row] < target_coverage)
        if accepted.size:
            thresholds[row] = values[accepted[-1]]
    return thresholds, alphas


def effective_sample_size(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Kish effective size of the calibration weights for diagnostics."""
    weights = _kernel(np.atleast_2d(np.asarray(distances, dtype=float)), bandwidth)
    numerator = np.square(weights.sum(axis=1))
    denominator = np.square(weights).sum(axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(weights.shape[0], dtype=float),
        where=denominator > 0,
    )
