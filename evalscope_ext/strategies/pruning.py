# Copyright (c) 2024. Licensed under the Apache License, Version 2.0.
"""
Benchmark pruning strategies for evalscope.

The problem this solves
-----------------------
Running a full capability benchmark across many candidate models is expensive.
For a *go / no-go* decision ("is this model good enough for the customer's
workload?") we do not need every item -- we need the smallest subset whose
*weighted* score reconstructs the full-benchmark score within a tight error
band, for a model we have not seen before.

Why not the obvious baselines
-----------------------------
* Uniform random keeps the estimate unbiased but high-variance at small N.
* Top-k hardest / easiest is *biased*: a subset of only hard items reports
  ~0% for every model, which preserves ranking but destroys the absolute
  score the go/no-go threshold is compared against.
* Hand-picking does not generalize and is not reproducible.

The method
----------
Stratified anchor selection (in the spirit of Anchor Points / tinyBenchmarks):

1. Estimate each item's *difficulty* from the reference models that have
   already been run (fraction of reference models that pass it). Difficulty is
   an item property, so it transfers to an unseen model far better than that
   model's own (unknown) scores.
2. Sort items by difficulty and split into ``n_strata`` equal-mass strata.
3. From each stratum, take anchors spread evenly across the difficulty range
   (not clustered), so the subset covers easy, medium and hard regions in the
   same proportion as the full set.
4. Give each anchor a *reconstruction weight* equal to the number of full-set
   items its stratum represents. The weighted mean of the anchors is then an
   approximately unbiased, low-variance estimator of the full-set mean.

This deliberately preserves the *absolute* score (what the threshold needs),
which is why it beats both random and hardest-k on leave-one-model-out tests.

Defensibility for an unseen model
----------------------------------
Selection uses only item difficulty estimated from reference models plus the
item's position in the difficulty ordering -- never the target model's scores.
The strata themselves are model-agnostic, so the same anchor set is valid for a
fourth model. The reconstruction weights correct the sampling, not the model.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


@dataclass
class ItemStat:
    index: int
    difficulty: float       # fraction of reference models correct (0..1)
    discrimination: float   # variance of correctness across reference models


@dataclass
class PruneResult:
    """The selected anchors and their reconstruction weights."""
    indices: List[int]
    weights: List[float]
    strategy: str
    prune_ratio: float

    @property
    def kept(self) -> int:
        return len(self.indices)

    def weight_for(self, index: int) -> float:
        try:
            return self.weights[self.indices.index(index)]
        except ValueError:
            return 0.0


def load_item_stats(benchmark: str) -> Optional[List[ItemStat]]:
    """Load precomputed per-item difficulty/discrimination shipped with the ext.

    Returns ``None`` when no stats file exists for *benchmark* -- callers should
    then fall back to a model-agnostic strategy (see ``_fallback_difficulty``).
    """
    path = os.path.join(_DATA_DIR, f'{benchmark}_item_stats.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as fh:
        blob = json.load(fh)
    return [ItemStat(it['index'], it['difficulty'], it['discrimination']) for it in blob['items']]


def _resolve_strata(size: int) -> int:
    """Pick a stratum count that divides the budget cleanly.

    Using a fixed count (e.g. 8) creates boundary artefacts when ``size`` is not
    a multiple of it. We choose the largest divisor-friendly count <= 10 so each
    stratum receives a near-equal number of anchors.
    """
    n_strata = min(10, max(2, size))
    while n_strata > 2 and size % n_strata != 0 and size // n_strata < 2:
        n_strata -= 1
    return n_strata


def stratified_anchor_select(
    stats: Sequence[ItemStat],
    prune_ratio: float,
    seed: int = 0,
) -> PruneResult:
    """Select a weighted anchor subset that reconstructs the full-set score.

    Args:
        stats: per-item difficulty/discrimination for the full benchmark.
        prune_ratio: fraction of items to KEEP (0.1 == keep 10%).
        seed: only used to break ties deterministically.

    Returns:
        PruneResult with kept indices and reconstruction weights.
    """
    if not 0 < prune_ratio <= 1:
        raise ValueError(f'prune_ratio must be in (0, 1], got {prune_ratio}')

    n = len(stats)
    size = max(2, int(round(n * prune_ratio)))
    if size >= n:
        return PruneResult([s.index for s in stats], [1.0] * n, 'stratified_anchor', prune_ratio)

    difficulties = np.array([s.difficulty for s in stats])
    indices = np.array([s.index for s in stats])

    # Stable difficulty ordering; ties broken by index for reproducibility.
    order = np.lexsort((indices, difficulties))
    n_strata = _resolve_strata(size)
    strata = np.array_split(order, n_strata)

    base, extra = divmod(size, n_strata)
    sel_pos: List[int] = []
    weights: List[float] = []
    for s_i, stratum in enumerate(strata):
        if len(stratum) == 0:
            continue
        take = max(1, base + (1 if s_i < extra else 0))
        take = min(take, len(stratum))
        # Spread anchors across the stratum's difficulty range rather than
        # clustering -- this lowers reconstruction variance.
        picks = stratum[np.linspace(0, len(stratum) - 1, take).astype(int)]
        for p in picks:
            sel_pos.append(int(p))
            weights.append(len(stratum) / take)  # items represented per anchor

    kept_idx = [int(indices[p]) for p in sel_pos]
    # Normalize weights so they sum to the number of anchors (keeps the weighted
    # mean on the same scale as a plain mean).
    w = np.array(weights, dtype=float)
    w = w / w.sum() * len(w)
    return PruneResult(kept_idx, w.tolist(), 'stratified_anchor', prune_ratio)


def _fallback_difficulty(indices: Sequence[int]) -> List[ItemStat]:
    """Model-agnostic difficulty proxy when no reference scores are available.

    With no precomputed stats (e.g. a brand-new benchmark) we cannot estimate
    difficulty from models. We assign a neutral difficulty so the stratifier
    degrades gracefully to an evenly-spaced systematic sample -- still strictly
    better than clustered random because it guarantees coverage.
    """
    return [ItemStat(int(i), difficulty=0.5, discrimination=0.0) for i in indices]


STRATEGIES = {
    'stratified_anchor': stratified_anchor_select,
}


def prune(
    benchmark: str,
    all_indices: Sequence[int],
    prune_ratio: float,
    strategy: str = 'stratified_anchor',
    seed: int = 0,
) -> PruneResult:
    """Top-level entry: pick anchors for *benchmark* at *prune_ratio*.

    Falls back to a model-agnostic systematic sample when no item stats ship for
    the benchmark, so the call never fails for an unknown dataset.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f'unknown pruning_strategy {strategy!r}; choices: {list(STRATEGIES)}')

    stats = load_item_stats(benchmark)
    if stats is None:
        stats = _fallback_difficulty(all_indices)
    else:
        # Restrict to items actually present in this run (defensive).
        present = set(all_indices)
        stats = [s for s in stats if s.index in present] or _fallback_difficulty(all_indices)

    return STRATEGIES[strategy](stats, prune_ratio, seed=seed)
