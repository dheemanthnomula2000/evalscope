# Copyright (c) 2024. Licensed under the Apache License, Version 2.0.
"""
compare_runs -- quantify how faithfully a pruned run reproduces the full run.

Run contract (from the task):

    evalscope eval --model <m> --datasets live_code_bench --output ./results_full/
    evalscope eval --model <m> --datasets live_code_bench_pruned \
        --dataset-args '{"pruning_strategy":"stratified_anchor","prune_ratio":0.25}' \
        --output ./results_pruned/
    python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/

It reads the per-sample scores from both runs, recomputes the pruned estimate
as a *weight-corrected* mean (using the ``prune_weight`` stored on each kept
sample), and reports:

* full score, pruned (weighted) estimate, and absolute error,
* the cost saving (items run),
* a go/no-go agreement check against a customer threshold.

The tool is defensive about evalscope's on-disk layout, which varies by
version: it searches for per-sample J/JSONL review files and falls back to
reading any ``*.jsonl`` under the output dir.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple


def _find_sample_files(root: str) -> List[str]:
    patterns = [
        os.path.join(root, '**', 'reviews', '*.jsonl'),
        os.path.join(root, '**', '*review*.jsonl'),
        os.path.join(root, '**', '*.jsonl'),
    ]
    seen: List[str] = []
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            if f not in seen:
                seen.append(f)
        if seen:
            break
    return seen


def _extract_score(obj: dict) -> Optional[float]:
    """Pull a scalar score from a review row across known schemas."""
    ss = obj.get('sample_score') or obj.get('score') or {}
    val = ss.get('score', ss).get('value', ss) if isinstance(ss, dict) else ss
    if isinstance(val, dict):
        for k in ('pass', 'acc', 'accuracy', 'correct', 'score'):
            if k in val:
                try:
                    return float(val[k])
                except (TypeError, ValueError):
                    return None
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _load_run(root: str) -> Dict[int, Tuple[float, float]]:
    """Return {index: (score, weight)}. Weight defaults to 1.0 when absent."""
    out: Dict[int, Tuple[float, float]] = {}
    for path in _find_sample_files(root):
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'index' not in row:
                    continue
                score = _extract_score(row)
                if score is None:
                    continue
                meta = row.get('metadata') or {}
                weight = float(meta.get('prune_weight', 1.0))
                out[int(row['index'])] = (score, weight)
    return out


def weighted_mean(rows: Dict[int, Tuple[float, float]]) -> float:
    if not rows:
        return float('nan')
    num = sum(s * w for s, w in rows.values())
    den = sum(w for _, w in rows.values())
    return num / den if den else float('nan')


def compare(full_dir: str, pruned_dir: str, threshold: Optional[float] = None) -> dict:
    full = _load_run(full_dir)
    pruned = _load_run(pruned_dir)

    full_score = weighted_mean(full)
    pruned_score = weighted_mean(pruned)
    abs_err = abs(full_score - pruned_score) if full and pruned else float('nan')

    report = {
        'full_items': len(full),
        'pruned_items': len(pruned),
        'cost_saving': round(1 - len(pruned) / len(full), 3) if full else None,
        'full_score': round(full_score, 4),
        'pruned_estimate': round(pruned_score, 4),
        'absolute_error': round(abs_err, 4),
    }
    if threshold is not None:
        report['threshold'] = threshold
        report['full_verdict'] = 'go' if full_score >= threshold else 'no-go'
        report['pruned_verdict'] = 'go' if pruned_score >= threshold else 'no-go'
        report['verdicts_agree'] = report['full_verdict'] == report['pruned_verdict']
    return report


def _print(report: dict) -> None:
    print('\nPruned-vs-full comparison')
    print('-' * 34)
    print(f"  full benchmark items : {report['full_items']}")
    print(f"  pruned items         : {report['pruned_items']}")
    if report.get('cost_saving') is not None:
        print(f"  cost saving          : {report['cost_saving']:.0%}")
    print(f"  full score           : {report['full_score']}")
    print(f"  pruned estimate      : {report['pruned_estimate']}")
    print(f"  absolute error       : {report['absolute_error']}")
    if 'threshold' in report:
        agree = 'AGREE' if report['verdicts_agree'] else 'DISAGREE'
        print(f"  threshold {report['threshold']}: full={report['full_verdict']} "
              f"pruned={report['pruned_verdict']} -> {agree}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description='Compare a pruned evalscope run against the full run.')
    ap.add_argument('--full', required=True, help='Output dir of the full run.')
    ap.add_argument('--pruned', required=True, help='Output dir of the pruned run.')
    ap.add_argument('--threshold', type=float, default=None,
                    help='Optional go/no-go quality bar (e.g. 0.6) to check verdict agreement.')
    ap.add_argument('--json', action='store_true', help='Emit JSON only.')
    args = ap.parse_args()

    report = compare(args.full, args.pruned, args.threshold)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print(report)


if __name__ == '__main__':
    main()
