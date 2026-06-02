"""Reproduce the leave-one-model-out validation numbers in HANDOUT_A / README.

Usage:
    python scripts_validate.py --reviews "/path/to/Evals/Part 1/reviews"

Expects the shipped review files named:
    {benchmark}__{model}.jsonl   e.g. live_code_bench_v5__gpt-oss-120b.jsonl
"""
import argparse, json, glob, os, numpy as np
from evalscope_ext.strategies.pruning import stratified_anchor_select, ItemStat

MODELS = ['gpt-oss-120b', 'kimi-k2.5', 'minimax-m2.5']
BENCHES = [('live_code_bench_v5', 'pass'), ('aa_lcr', 'acc')]

def load(reviews_dir, bench, metric):
    M = {}
    for m in MODELS:
        d = {}
        path = os.path.join(reviews_dir, f"{bench}__{m}.jsonl")
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                d[r['index']] = float(r['sample_score']['score']['value'][metric])
        M[m] = d
    idx = sorted(set.intersection(*[set(d) for d in M.values()]))
    return np.array(idx), np.array([[M[m][i] for m in MODELS] for i in idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reviews', required=True, help='Dir with the shipped *.jsonl review files')
    args = ap.parse_args()
    for bench, metric in BENCHES:
        idx, X = load(args.reviews, bench, metric); n, k = X.shape; full = X.mean(0)
        print(f"\n{bench}: {n} items | full acc {dict(zip(MODELS,[round(f,3) for f in full]))}")
        print(f"{'keep':>6}{'n':>5}{'anchorMAE':>11}{'randMAE':>9}{'improve':>9}{'maxErr':>8}")
        for frac in [0.10, 0.15, 0.20, 0.25, 0.30]:
            ae, re_, mx = [], [], 0
            for held in range(k):
                tr = [j for j in range(k) if j != held]
                diff = X[:, tr].mean(1); disc = X[:, tr].var(1)
                stats = [ItemStat(int(idx[i]), float(diff[i]), float(disc[i])) for i in range(n)]
                res = stratified_anchor_select(stats, frac)
                pos = {int(idx[i]): i for i in range(n)}
                sel = [pos[i] for i in res.indices]; w = np.array(res.weights)
                est = (X[sel, held] * w).sum() / w.sum()
                e = abs(est - full[held]); ae.append(e); mx = max(mx, e)
                rng = np.random.default_rng(100 + held)
                re_.append(np.mean([abs(X[rng.choice(n, len(sel), replace=False), held].mean() - full[held]) for _ in range(300)]))
            a, r = np.mean(ae), np.mean(re_)
            print(f"{frac:>6.0%}{len(sel):>5}{a:>11.4f}{r:>9.4f}{100*(1-a/r):>8.0f}%{mx:>8.4f}")

if __name__ == '__main__':
    main()
