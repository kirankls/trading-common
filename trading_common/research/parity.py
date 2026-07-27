"""Paired-comparison statistics for validating one result set against
another — backtest vs live, shadow vs production, or any two paired
outcome samples (docs/DAYTRADER_BACKTEST_INTEGRATION_PROMPT.md Phase 2).

Ported from Chanakya's ``options_advisor.learning.promotion`` — the
canonical source of this exact pattern in either project.  DayTrader's own
``research/parity_gate.py`` asks the same fundamental question ("does
dataset A differ meaningfully from dataset B, paired sample by paired
sample, and is A ever materially worse in some bucket?"), but its own
implementation is a full live-bot-vs-harness replay deeply coupled to
DayTrader's trading engine (``engine.runner.BotRunner``,
``brokers.paper.PaperBroker``, ``engine.regime.RegimeClassifier``, …) — not
a reusable statistical primitive. What's shared here is the STATISTICS,
not the replay machinery that produces the two datasets being compared.

Both functions below are pure — no I/O, no knowledge of what the paired
samples actually represent (R-multiples, dollar P&L, or anything else two
comparable datasets might track).
"""
from __future__ import annotations

import random
from typing import Any, Sequence

__all__ = ["paired_bootstrap_p_value", "bucketed_materially_worse"]


def paired_bootstrap_p_value(
    diffs: Sequence[float],
    *,
    iters: int = 2000,
    seed: int = 12345,
) -> float:
    """One-sided bootstrap p-value for H0: ``mean(diffs) <= 0`` against H1:
    ``mean(diffs) > 0`` — i.e. dataset A outperforms dataset B on paired
    samples, where ``diffs[i] = a[i] - b[i]``.

    Resamples ``diffs`` with replacement ``iters`` times; the p-value is
    the fraction of bootstrap means ``<= 0`` (how often the observed
    outperformance could vanish under resampling). Deterministic given
    ``seed``.

    Returns 1.0 (cannot reject H0) for an empty or non-positive-mean
    sample — no evidence of outperformance is the safe default, not an
    error.
    """
    diffs = [float(d) for d in diffs]
    n = len(diffs)
    if n == 0:
        return 1.0
    observed_mean = sum(diffs) / n
    if observed_mean <= 0:
        return 1.0
    rng = random.Random(seed)
    le_zero = 0
    for _ in range(iters):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += diffs[rng.randrange(n)]
        if sample_sum / n <= 0:
            le_zero += 1
    return le_zero / iters


def bucketed_materially_worse(
    paired: Sequence[dict[str, Any]],
    *,
    bucket_key: str,
    a_key: str,
    b_key: str,
    margin: float,
    min_samples: int,
) -> list[str]:
    """Return the buckets (with ``>= min_samples`` paired observations)
    where dataset A's mean is materially worse than dataset B's — i.e.
    ``mean(a) - mean(b) < -margin``. An empty list means no bucket where A
    is materially worse than B.

    Each dict in *paired* must carry ``a_key``, ``b_key`` (both numeric —
    entries missing either are skipped) and ``bucket_key`` (the grouping
    value, e.g. a VIX-regime label; defaults to the literal string
    ``"unknown"`` when a dict lacks it rather than being dropped).
    """
    groups: dict[str, list[tuple[float, float]]] = {}
    for d in paired:
        a_val = d.get(a_key)
        b_val = d.get(b_key)
        if a_val is None or b_val is None:
            continue
        bucket = d.get(bucket_key, "unknown")
        groups.setdefault(bucket, []).append((float(a_val), float(b_val)))

    worse: list[str] = []
    for bucket, pairs in groups.items():
        if len(pairs) < min_samples:
            continue
        mean_a = sum(a for a, _ in pairs) / len(pairs)
        mean_b = sum(b for _, b in pairs) / len(pairs)
        if (mean_a - mean_b) < -margin:
            worse.append(bucket)
    return worse
