# Copyright (c) 2024. Licensed under the Apache License, Version 2.0.
"""
Register pruned variants of existing evalscope benchmarks.

Importing this module registers ``live_code_bench_pruned`` and
``aa_lcr_pruned`` so they can be selected with::

    evalscope eval --model <m> --datasets live_code_bench_pruned \
        --dataset-args '{"pruning_strategy": "stratified_anchor", "prune_ratio": 0.25}'

Each variant reuses the upstream adapter's data loading and scoring untouched;
only the post-load anchor filter is added by ``PruningAdapterMixin``.
"""
from __future__ import annotations

from .adapter import PRUNE_BENCHMARK_KEY, PruningAdapterMixin


def _make_pruned(base_adapter_cls, variant_name: str, stats_key: str, pretty: str):
    """Build and register a pruned subclass of *base_adapter_cls*."""
    from evalscope.api.benchmark import BenchmarkMeta  # type: ignore
    from evalscope.api.registry import register_benchmark  # type: ignore

    extra = {
        'pruning_strategy': {
            'type': 'str', 'default': 'stratified_anchor',
            'choices': ['stratified_anchor'],
            'description': 'Sample-selection strategy.',
        },
        'prune_ratio': {
            'type': 'float', 'default': 1.0,
            'description': 'Fraction of items to keep (e.g. 0.25 keeps 25%).',
        },
    }

    @register_benchmark(
        BenchmarkMeta(
            name=variant_name,
            pretty_name=pretty,
            tags=getattr(base_adapter_cls, '_tags', []),
            description=f'Pruned variant of {stats_key} using stratified anchor selection.',
            extra_params=extra,
        )
    )
    class _Pruned(PruningAdapterMixin, base_adapter_cls):  # type: ignore[misc, valid-type]
        pass

    setattr(_Pruned, PRUNE_BENCHMARK_KEY, stats_key)
    _Pruned.__name__ = f'{base_adapter_cls.__name__}Pruned'
    return _Pruned


def register_all():
    """Register every pruned variant. Safe to call once at import time."""
    from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter  # type: ignore

    _make_pruned(AALCRAdapter, 'aa_lcr_pruned', 'aa_lcr', 'AA-LCR (pruned)')

    # live_code_bench's adapter class name may vary across evalscope versions;
    # resolve it dynamically to stay robust to the pinned SHA.
    import importlib
    lcb_mod = importlib.import_module('evalscope.benchmarks.live_code_bench.live_code_bench_adapter')
    lcb_cls = next(
        obj for name, obj in vars(lcb_mod).items()
        if name.endswith('Adapter') and isinstance(obj, type)
    )
    _make_pruned(lcb_cls, 'live_code_bench_pruned', 'live_code_bench_v5', 'LiveCodeBench v5 (pruned)')


# Auto-register on import for convenience.
try:
    register_all()
except Exception:  # pragma: no cover - host without these benchmarks
    pass
