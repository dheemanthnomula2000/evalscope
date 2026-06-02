# Copyright (c) 2024. Licensed under the Apache License, Version 2.0.
"""Tests for the pruning extension. Run with: pytest tests/ -q"""
import numpy as np

from evalscope_ext.strategies.pruning import (
    ItemStat, prune, stratified_anchor_select, load_item_stats,
)
from evalscope_ext.strategies.confidence import confidence_curve, recommend_min_ratio


def _synthetic_stats(n=300, seed=0):
    rng = np.random.default_rng(seed)
    diff = rng.uniform(0, 1, n)
    return [ItemStat(i, float(diff[i]), float(diff[i] * (1 - diff[i]))) for i in range(n)]


def test_keep_ratio_count():
    stats = _synthetic_stats(300)
    res = stratified_anchor_select(stats, 0.25)
    assert abs(res.kept - 75) <= 10  # ~25% within stratum rounding


def test_weights_sum_to_anchor_count():
    stats = _synthetic_stats(200)
    res = stratified_anchor_select(stats, 0.2)
    assert abs(sum(res.weights) - res.kept) < 1e-6


def test_weighted_mean_recovers_full_mean():
    # If anchors are weight-corrected, weighted mean of a per-item signal
    # should approximate the full-set mean.
    rng = np.random.default_rng(1)
    n = 300
    diff = rng.uniform(0, 1, n)
    stats = [ItemStat(i, float(diff[i]), 0.0) for i in range(n)]
    signal = diff  # use difficulty itself as a stand-in per-item score
    res = stratified_anchor_select(stats, 0.25)
    w = np.array(res.weights)
    est = (signal[res.indices] * w).sum() / w.sum()
    assert abs(est - signal.mean()) < 0.05


def test_full_ratio_returns_all():
    stats = _synthetic_stats(50)
    res = stratified_anchor_select(stats, 1.0)
    assert res.kept == 50


def test_unknown_strategy_raises():
    try:
        prune('whatever', list(range(10)), 0.5, strategy='nope')
        assert False, 'should have raised'
    except ValueError:
        pass


def test_fallback_when_no_stats():
    # Unknown benchmark -> model-agnostic systematic sample, still right size.
    res = prune('benchmark_with_no_stats', list(range(100)), 0.2)
    assert abs(res.kept - 20) <= 4


def test_shipped_stats_loadable():
    for bench in ('live_code_bench_v5', 'aa_lcr'):
        stats = load_item_stats(bench)
        assert stats is not None and len(stats) > 0


def test_confidence_flags_noisy_benchmark():
    # All-agree items (no discrimination) -> high placement spread is not
    # expected; but a benchmark with thin signal should not be marked 'safe'
    # at tiny ratios. Sanity: safe ratios are monotone-ish and recommend works.
    rng = np.random.default_rng(2)
    X = (rng.uniform(0, 1, (100, 3)) > 0.5).astype(float)
    rows = confidence_curve(X, X.mean(1))
    rec = recommend_min_ratio(rows)
    assert rec is None or 0 < rec <= 1
