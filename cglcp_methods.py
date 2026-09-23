"""Reusable conformal interval methods for the CGLCP experiments.

The functions in this module operate on a *realized local calibration pool*.
Pool construction, graph fitting, and prediction-model fitting must be done
without inspecting the final calibration or test scores.

The five public interval methods are:

``local_uncorrected_interval``
    Ordinary split conformal on the local predicted-safe pool.
``global_rate_plugin_interval``
    Sharp-rank plug-in using a global contamination rate as if it were local.
``additive_certified_interval``
    The earlier additive contamination correction using a certified count.
``cglcp_certified_interval``
    The V5 sharp certified-rank method.
``oracle_local_interval``
    The infeasible benchmark that knows the true local unsafe count.
``oracle_safe_interval``
    The stronger infeasible benchmark that knows every unsafe member and
    removes those members before ordinary split conformal calibration.

Use ``create_verification_certificate`` to audit a simple-random subset of a
pool in simulations or benchmarks. In deployment, construct the same
``VerificationCertificate`` from independently obtained audit labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, inf, isfinite
from typing import Dict, Optional, Sequence, Union

import numpy as np
from scipy.stats import hypergeom


@dataclass(frozen=True)
class VerificationCertificate:
    """One-sided finite-population certificate for a realized local pool."""

    pool_size: int
    verification_size: int
    verified_unsafe_count: int
    unsafe_upper_count: int
    rho: float
    sampled_indices: tuple[int, ...] = ()

    @property
    def observed_unsafe_rate(self) -> float:
        return self.verified_unsafe_count / self.verification_size

    @property
    def certified_unsafe_rate(self) -> float:
        return self.unsafe_upper_count / self.pool_size


@dataclass(frozen=True)
class ConformalIntervalResult:
    """Prediction interval plus the rank information used to construct it."""

    method: str
    lower: float
    upper: float
    center: float
    quantile: float
    rank: int
    calibration_size: int
    target_alpha: float
    unsafe_count_used: int = 0
    conditional_alpha: Optional[float] = None

    @property
    def finite(self) -> bool:
        return isfinite(self.quantile)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Return whether ``value`` is covered by the closed interval."""

        return self.lower <= value <= self.upper


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must satisfy 0 < alpha < 1; got {alpha}")


def _validate_scores(calibration_scores: Sequence[float]) -> np.ndarray:
    scores = np.asarray(calibration_scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("calibration_scores must be one-dimensional")
    if np.any(~np.isfinite(scores)):
        raise ValueError("calibration_scores must contain only finite values")
    if np.any(scores < 0.0):
        raise ValueError("calibration_scores must be nonnegative")
    return scores


def _validate_unsafe_count(unsafe_count: int, pool_size: int, name: str) -> int:
    value = int(unsafe_count)
    if value != unsafe_count or not 0 <= value <= pool_size:
        raise ValueError(f"{name} must be an integer in [0, {pool_size}]")
    return value


def _interval_from_rank(
    calibration_scores: Sequence[float],
    center: float,
    rank: int,
    *,
    method: str,
    target_alpha: float,
    unsafe_count_used: int = 0,
    conditional_alpha: Optional[float] = None,
) -> ConformalIntervalResult:
    scores = _validate_scores(calibration_scores)
    n = scores.size
    if not isfinite(center):
        raise ValueError("center must be finite")
    if not 1 <= rank <= n + 1:
        raise ValueError(f"rank must lie in [1, {n + 1}]; got {rank}")

    if rank == n + 1:
        quantile = inf
        lower, upper = -inf, inf
    else:
        quantile = float(np.partition(scores, rank - 1)[rank - 1])
        lower, upper = center - quantile, center + quantile

    return ConformalIntervalResult(
        method=method,
        lower=lower,
        upper=upper,
        center=float(center),
        quantile=quantile,
        rank=rank,
        calibration_size=n,
        target_alpha=target_alpha,
        unsafe_count_used=unsafe_count_used,
        conditional_alpha=conditional_alpha,
    )


def ordinary_conformal_rank(pool_size: int, alpha: float) -> int:
    """Return ``ceil((n + 1) * (1 - alpha))`` in ``[1, n + 1]``."""

    _validate_alpha(alpha)
    if pool_size < 0:
        raise ValueError("pool_size must be nonnegative")
    return ceil((pool_size + 1) * (1.0 - alpha))


def sharp_contamination_rank(
    pool_size: int,
    unsafe_count: int,
    conditional_alpha: float,
) -> int:
    """Smallest sharp rank valid for a known/upper-bounded unsafe count.

    It is the smallest integer ``k`` satisfying

    ``(k - B) / (n - B + 1) >= 1 - conditional_alpha``.
    """

    _validate_alpha(conditional_alpha)
    unsafe_count = _validate_unsafe_count(
        unsafe_count, pool_size, "unsafe_count"
    )
    safe_count = pool_size - unsafe_count
    return unsafe_count + ceil((1.0 - conditional_alpha) * (safe_count + 1))


@lru_cache(maxsize=None)
def hypergeometric_unsafe_upper_bound(
    pool_size: int,
    verification_size: int,
    verified_unsafe_count: int,
    rho: float,
) -> int:
    """Invert the exact hypergeometric CDF to upper-bound unsafe pool members.

    Returns the largest ``b`` such that

    ``P_{X ~ Hypergeom(pool_size, b, verification_size)}(X <= x) >= rho``.

    Consequently, for every fixed realized pool,
    ``P(B <= returned_bound) >= 1 - rho`` under simple-random verification
    without replacement.
    """

    n = int(pool_size)
    s = int(verification_size)
    x = int(verified_unsafe_count)
    if n != pool_size or n < 1:
        raise ValueError("pool_size must be a positive integer")
    if s != verification_size or not 1 <= s <= n:
        raise ValueError(f"verification_size must be an integer in [1, {n}]")
    if x != verified_unsafe_count or not 0 <= x <= s:
        raise ValueError(f"verified_unsafe_count must be an integer in [0, {s}]")
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must satisfy 0 < rho < 1; got {rho}")

    # The CDF is nonincreasing in the population unsafe count b.
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cdf = float(hypergeom.cdf(x, n, mid, s))
        if cdf >= rho:
            lo = mid
        else:
            hi = mid - 1
    return lo


def create_verification_certificate(
    unsafe_labels: Sequence[Union[bool, int]],
    verification_size: int,
    rho: float,
    *,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> VerificationCertificate:
    """Audit a simple-random subset and return its exact count certificate.

    ``unsafe_labels[j]`` is used only when index ``j`` is sampled. Supplying the
    full vector is convenient for simulations and reference-graph benchmarks;
    a deployment implementation may instead obtain those sampled labels from
    an independent assay.
    """

    labels = np.asarray(unsafe_labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("unsafe_labels must be a nonempty one-dimensional array")
    if not np.all(np.isin(labels, [0, 1, False, True])):
        raise ValueError("unsafe_labels must contain only boolean/0/1 values")
    n = int(labels.size)
    if not 1 <= verification_size <= n:
        raise ValueError(f"verification_size must lie in [1, {n}]")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    sampled = np.sort(
        rng.choice(n, size=verification_size, replace=False)
    )
    x = int(labels[sampled].astype(bool).sum())
    upper = hypergeometric_unsafe_upper_bound(
        n, verification_size, x, rho
    )
    return VerificationCertificate(
        pool_size=n,
        verification_size=verification_size,
        verified_unsafe_count=x,
        unsafe_upper_count=upper,
        rho=rho,
        sampled_indices=tuple(int(j) for j in sampled),
    )


def local_uncorrected_interval(
    calibration_scores: Sequence[float],
    center: float,
    *,
    alpha: float = 0.10,
) -> ConformalIntervalResult:
    """Ordinary split conformal interval on the realized local pool."""

    scores = _validate_scores(calibration_scores)
    rank = ordinary_conformal_rank(scores.size, alpha)
    return _interval_from_rank(
        scores,
        center,
        rank,
        method="local_uncorrected",
        target_alpha=alpha,
        conditional_alpha=alpha,
    )


def oracle_local_interval(
    calibration_scores: Sequence[float],
    center: float,
    true_local_unsafe_count: int,
    *,
    alpha: float = 0.10,
) -> ConformalIntervalResult:
    """Sharp oracle benchmark using the true realized local unsafe count."""

    scores = _validate_scores(calibration_scores)
    rank = sharp_contamination_rank(
        scores.size, true_local_unsafe_count, alpha
    )
    return _interval_from_rank(
        scores,
        center,
        rank,
        method="oracle_local",
        target_alpha=alpha,
        unsafe_count_used=int(true_local_unsafe_count),
        conditional_alpha=alpha,
    )


def oracle_safe_interval(
    calibration_scores: Sequence[float],
    unsafe_labels: Sequence[Union[bool, int]],
    center: float,
    *,
    alpha: float = 0.10,
) -> ConformalIntervalResult:
    """Oracle CP after deleting every labeled-unsafe calibration member.

    Unlike :func:`oracle_local_interval`, this benchmark knows the identity of
    each unsafe member rather than only the realized unsafe count ``B``.  It is
    therefore an information upper bound, not a deployable competitor.
    """

    scores = _validate_scores(calibration_scores)
    labels = np.asarray(unsafe_labels)
    if labels.ndim != 1 or labels.size != scores.size:
        raise ValueError("unsafe_labels must match calibration_scores")
    if not np.all(np.isin(labels, [0, 1, False, True])):
        raise ValueError("unsafe_labels must contain only boolean/0/1 values")
    safe_scores = scores[~labels.astype(bool)]
    rank = ordinary_conformal_rank(safe_scores.size, alpha)
    return _interval_from_rank(
        safe_scores,
        center,
        rank,
        method="oracle_safe",
        target_alpha=alpha,
        unsafe_count_used=int(labels.astype(bool).sum()),
        conditional_alpha=alpha,
    )


def global_rate_plugin_interval(
    calibration_scores: Sequence[float],
    center: float,
    global_contamination_rate: float,
    *,
    alpha: float = 0.10,
) -> ConformalIntervalResult:
    """Sharp-rank heuristic that plugs the global rate into a local pool.

    The implied local unsafe count is conservatively rounded up. This baseline
    has no local validity guarantee when graph errors are spatially clustered.
    """

    scores = _validate_scores(calibration_scores)
    if not 0.0 <= global_contamination_rate <= 1.0:
        raise ValueError("global_contamination_rate must lie in [0, 1]")
    plugged_count = ceil(scores.size * global_contamination_rate)
    rank = sharp_contamination_rank(scores.size, plugged_count, alpha)
    return _interval_from_rank(
        scores,
        center,
        rank,
        method="global_rate_plugin",
        target_alpha=alpha,
        unsafe_count_used=plugged_count,
        conditional_alpha=alpha,
    )


def additive_certified_interval(
    calibration_scores: Sequence[float],
    center: float,
    unsafe_upper_count: int,
    *,
    alpha: float = 0.10,
    rho: float = 0.0,
) -> ConformalIntervalResult:
    """Earlier additive contamination correction using a count certificate.

    On a valid certificate, it sets

    ``alpha' = beta - Bbar / (n - Bbar + 1)``,

    where ``beta = (alpha - rho) / (1 - rho)`` allocates the end-to-end
    failure budget in the same way as CGLCP.  Passing ``rho=0`` recovers the
    known-count/additive formula.  If ``alpha' <= 0``, the method abstains with
    an infinite interval.
    """

    scores = _validate_scores(calibration_scores)
    _validate_alpha(alpha)
    if not 0.0 <= rho < alpha:
        raise ValueError(
            f"rho must satisfy 0 <= rho < alpha; got rho={rho}"
        )
    upper = _validate_unsafe_count(
        unsafe_upper_count, scores.size, "unsafe_upper_count"
    )
    safe_lower = scores.size - upper
    additive_penalty = upper / (safe_lower + 1)
    beta = (alpha - rho) / (1.0 - rho)
    adjusted_alpha = beta - additive_penalty

    if adjusted_alpha <= 0.0:
        rank = scores.size + 1
    else:
        rank = ordinary_conformal_rank(scores.size, adjusted_alpha)

    return _interval_from_rank(
        scores,
        center,
        rank,
        method="additive_certified",
        target_alpha=alpha,
        unsafe_count_used=upper,
        conditional_alpha=max(adjusted_alpha, 0.0),
    )


def oracle_certificate_budget_interval(
    calibration_scores: Sequence[float],
    center: float,
    true_local_unsafe_count: int,
    *,
    alpha: float = 0.10,
    rho: float = 0.02,
) -> ConformalIntervalResult:
    """Oracle efficiency benchmark that knows ``B`` but retains ``rho``.

    This is ``k_rho(B)`` from the V5 efficiency decomposition: it isolates
    rank inflation caused by estimating ``B`` from the unavoidable cost of
    reserving a certificate-failure budget.  It differs from
    :func:`oracle_local_interval`, which knows ``B`` and spends no ``rho``.
    """

    scores = _validate_scores(calibration_scores)
    _validate_alpha(alpha)
    if not 0.0 < rho < alpha:
        raise ValueError(f"rho must satisfy 0 < rho < alpha; got rho={rho}")
    unsafe_count = _validate_unsafe_count(
        true_local_unsafe_count, scores.size, "true_local_unsafe_count"
    )
    beta = (alpha - rho) / (1.0 - rho)
    rank = sharp_contamination_rank(scores.size, unsafe_count, beta)
    return _interval_from_rank(
        scores,
        center,
        rank,
        method="oracle_certificate_budget",
        target_alpha=alpha,
        unsafe_count_used=unsafe_count,
        conditional_alpha=beta,
    )


def cglcp_certified_interval(
    calibration_scores: Sequence[float],
    center: float,
    unsafe_upper_count: int,
    *,
    alpha: float = 0.10,
    rho: float = 0.02,
) -> ConformalIntervalResult:
    """CGLCP sharp interval from a one-sided unsafe-count certificate."""

    scores = _validate_scores(calibration_scores)
    _validate_alpha(alpha)
    if not 0.0 < rho < alpha:
        raise ValueError(f"rho must satisfy 0 < rho < alpha; got rho={rho}")
    upper = _validate_unsafe_count(
        unsafe_upper_count, scores.size, "unsafe_upper_count"
    )
    beta = (alpha - rho) / (1.0 - rho)
    rank = sharp_contamination_rank(scores.size, upper, beta)
    return _interval_from_rank(
        scores,
        center,
        rank,
        method="cglcp_certified",
        target_alpha=alpha,
        unsafe_count_used=upper,
        conditional_alpha=beta,
    )


def run_all_interval_methods(
    calibration_scores: Sequence[float],
    center: float,
    certificate: VerificationCertificate,
    *,
    global_contamination_rate: float,
    true_local_unsafe_count: int,
    alpha: float = 0.10,
) -> Dict[str, ConformalIntervalResult]:
    """Run all five paper methods on the same realized scores/certificate.

    This convenience function is intended for simulation and reference-graph
    benchmarks, where the true local unsafe count is available to the oracle
    baseline. The returned dictionary keys equal each result's ``method``.
    """

    scores = _validate_scores(calibration_scores)
    if certificate.pool_size != scores.size:
        raise ValueError(
            "certificate.pool_size must match len(calibration_scores)"
        )

    results = [
        local_uncorrected_interval(scores, center, alpha=alpha),
        global_rate_plugin_interval(
            scores,
            center,
            global_contamination_rate,
            alpha=alpha,
        ),
        additive_certified_interval(
            scores,
            center,
            certificate.unsafe_upper_count,
            alpha=alpha,
            rho=certificate.rho,
        ),
        cglcp_certified_interval(
            scores,
            center,
            certificate.unsafe_upper_count,
            alpha=alpha,
            rho=certificate.rho,
        ),
        oracle_local_interval(
            scores,
            center,
            true_local_unsafe_count,
            alpha=alpha,
        ),
    ]
    return {result.method: result for result in results}


__all__ = [
    "ConformalIntervalResult",
    "VerificationCertificate",
    "additive_certified_interval",
    "cglcp_certified_interval",
    "create_verification_certificate",
    "global_rate_plugin_interval",
    "hypergeometric_unsafe_upper_bound",
    "local_uncorrected_interval",
    "oracle_certificate_budget_interval",
    "oracle_local_interval",
    "oracle_safe_interval",
    "ordinary_conformal_rank",
    "run_all_interval_methods",
    "sharp_contamination_rank",
]
