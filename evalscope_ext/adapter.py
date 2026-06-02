# Copyright (c) 2024. Licensed under the Apache License, Version 2.0.
"""
evalscope integration for benchmark pruning.

Design choice -- why a mixin on the data adapter
------------------------------------------------
evalscope loads a benchmark through a ``DefaultDataAdapter`` subclass whose
``load()`` returns ``(test_dataset, fewshot_dataset)`` as ``DatasetDict``s.
Pruning is fundamentally a *post-load filter on the test set*, so the cleanest
upstream-friendly extension point is to wrap ``load()``: call the parent, then
filter the returned ``DatasetDict`` down to the selected anchors. This:

* touches no existing benchmark code (no forks of aa_lcr / live_code_bench),
* works for ANY benchmark that ships item-stats, and
* is driven entirely by ``--dataset-args`` (``pruning_strategy``,
  ``prune_ratio``), matching evalscope's existing ``extra_params`` convention.

Reconstruction weights are attached to each surviving sample's metadata so the
scorer/summarizer can compute the weighted mean (an unbiased estimate of the
full-benchmark score) rather than a plain mean over the subset.

Usage (registration)
---------------------
Create a pruned variant of an existing benchmark by mixing this in front of the
benchmark's adapter class and registering under a new name, e.g.
``live_code_bench_pruned``. See ``register.py``.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .strategies.pruning import prune

try:  # evalscope is the runtime host; guard import so the module is testable standalone.
    from evalscope.api.dataset import Dataset, DatasetDict  # type: ignore
    from evalscope.utils.logger import get_logger  # type: ignore
    logger = get_logger()
except Exception:  # pragma: no cover - exercised only outside evalscope
    DatasetDict = object  # type: ignore
    import logging
    logger = logging.getLogger(__name__)


# Default benchmark name -> shipped item-stats key. Lets one pruned adapter
# resolve the right stats file regardless of the registered variant name.
PRUNE_BENCHMARK_KEY = '_prune_stats_key'


class PruningAdapterMixin:
    """Mix in front of a ``DefaultDataAdapter`` subclass to enable pruning.

    Reads ``pruning_strategy`` (default ``stratified_anchor``) and
    ``prune_ratio`` (default ``1.0`` == no pruning) from the benchmark's
    ``extra_params`` and filters the loaded test set to the chosen anchors.
    """

    def load(self) -> Tuple['DatasetDict', Optional['DatasetDict']]:  # type: ignore[override]
        test_dataset, fewshot_dataset = super().load()  # type: ignore[misc]

        params = self.extra_params or {}  # type: ignore[attr-defined]
        ratio = float(params.get('prune_ratio', 1.0))
        strategy = params.get('pruning_strategy', 'stratified_anchor')
        if ratio >= 1.0:
            return test_dataset, fewshot_dataset

        stats_key = getattr(self, PRUNE_BENCHMARK_KEY, None) or getattr(self, 'name', '')

        for subset_name in list(test_dataset.keys()):
            dataset = test_dataset[subset_name]
            kept = self._prune_subset(dataset, stats_key, strategy, ratio)
            test_dataset[subset_name] = kept

        return test_dataset, fewshot_dataset

    # -- helpers ---------------------------------------------------------
    def _sample_index(self, sample, position: int) -> int:
        """Best-effort stable index for a sample.

        Prefer an explicit ``index`` in the sample metadata (matches the shipped
        reviews join key); fall back to enumeration order.
        """
        meta = getattr(sample, 'metadata', None) or {}
        for key in ('index', 'id', 'sample_id'):
            if key in meta:
                try:
                    return int(meta[key])
                except (TypeError, ValueError):
                    pass
        sid = getattr(sample, 'id', None)
        try:
            return int(sid)
        except (TypeError, ValueError):
            return position

    def _prune_subset(self, dataset, stats_key, strategy, ratio):
        positions = {self._sample_index(s, i): i for i, s in enumerate(dataset)}
        all_indices = list(positions.keys())

        result = prune(stats_key, all_indices, ratio, strategy=strategy)
        keep = set(result.indices)

        # Attach reconstruction weights to surviving samples' metadata.
        for idx in result.indices:
            pos = positions.get(idx)
            if pos is None:
                continue
            sample = dataset[pos]
            meta = dict(getattr(sample, 'metadata', None) or {})
            meta['prune_weight'] = result.weight_for(idx)
            meta['prune_strategy'] = result.strategy
            try:
                sample.metadata = meta
            except Exception:
                pass

        logger.info(
            f'[prune] {stats_key}: kept {result.kept}/{len(all_indices)} '
            f'({ratio:.0%}) via {result.strategy}'
        )
        return dataset.filter(lambda s: self._sample_index(s, -1) in keep)
