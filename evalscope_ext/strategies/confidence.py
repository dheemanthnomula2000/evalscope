# Copyright (c) 2024. Licensed under the Apache License, Version 2.0.
"""
Prune-confidence signal.

Before trusting a pruned run, a deployment engineer needs to know whether the
chosen ``prune_ratio`` is large enough for *this* benchmark. We answer that
without needing the target model: we measure how much the reconstructed
full-set score moves as the anchor placement is jittered across many valid
samples of the same size. If the score barely moves, the subset size is large
enough that the exact anchors do not matter -- the estimate is stable and the
ratio is safe. If it swings, the benchmark carries too little prunable signal
at that size and we say *hold*.

This is deliberately a property of the benchmark + ratio + reference scores
only, so it transfers to an unseen model. It is the mechanism that stops the
pruner from over-pruning a noisy benchmark such as an LLM-judged one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# Calibrated on the shipped reference data via leave-one-model-out:
#   spread <= SAFE_T  corresponds to held-out MAE within ~1-1.5 points
#   spread <= MARGINAL_T corresponds to held-out MAE within ~3 points
SAFE_T = 0.014
MARGINAL_T = 0.025


@dataclass
class ConfidenceRow:
    prune_ratio: float
    n_kept: int
    spread: float
    verdict: str  # 'safe' | 'marginal' | 'hold'


def _placement_spread(score_matrix: np.ndarray, difficulty: np.ndarray,
                      prune_ratio: float, trials: int = 200, seed: int = 0) -> float:
    """Std of the reconstructed full-set accuracy across jittered anchor sets."""
    rng = np.random.default_rng(seed)
    n, k = score_matrix.shape
    size = max(2, int(round(n * prune_ratio)))
    order = np.lexsort((np.arange(n), difficulty))
    per_model = []
    for j in range(k):
        ests = []
        for _ in range(trials):
            offs = rng.uniform(0, 1, size)
            pos = ((np.linspace(0, len(order) - 1e-9, size) + offs).astype(int)) % len(order)
            ests.append(score_matrix[order[pos], j].mean())
        per_model.append(float(np.std(ests)))
    return float(np.mean(per_model))


def confidence_curve(score_matrix: np.ndarray, difficulty: np.ndarray,
                     ratios: Sequence[float] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)) -> List[ConfidenceRow]:
    """Compute the safe-to-prune verdict for a range of keep ratios.

    Args:
        score_matrix: items x reference_models matrix of per-item scores.
        difficulty: per-item difficulty (item mean across reference models).
        ratios: keep-ratios to evaluate.
    """
    n = score_matrix.shape[0]
    rows: List[ConfidenceRow] = []
    for r in ratios:
        spread = _placement_spread(score_matrix, difficulty, r)
        verdict = 'safe' if spread <= SAFE_T else ('marginal' if spread <= MARGINAL_T else 'hold')
        rows.append(ConfidenceRow(r, max(2, int(round(n * r))), round(spread, 4), verdict))
    return rows


def recommend_min_ratio(rows: Sequence[ConfidenceRow]) -> Optional[float]:
    """Smallest keep-ratio that is at least 'marginal'.

    Returns ``None`` when no evaluated ratio is trustworthy -- the caller should
    then warn that the benchmark should not be pruned (run it in full).
    """
    safe = [r.prune_ratio for r in rows if r.verdict == 'safe']
    if safe:
        return min(safe)
    marginal = [r.prune_ratio for r in rows if r.verdict == 'marginal']
    if marginal:
        return min(marginal)
    return None
