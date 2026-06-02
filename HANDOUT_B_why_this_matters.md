# Handout B — Why this matters and how to use it

**Audience:** developers, test engineers, product, and the customer team.

## What this changes for the customer conversation

Today, answering *"is this model good enough for your coding workload?"* means
running the full benchmark on every candidate model — slow and expensive, so it
happens rarely and late. With the pruner, the coding benchmark runs on **a
quarter of the items and still lands within about one point** of the full score.
That turns a multi-hour, batch-scheduled evaluation into something a sales
engineer can run live during an eval cycle and trust for a go/no-go answer.

Concretely: **75% less compute and time for the LiveCodeBench answer, with
negligible change to the decision.**

## How to run it tomorrow

Inside the evalscope fork, three commands:

```bash
# 1. Full run (baseline, optional once you trust the pruned number)
evalscope eval --model <model> --datasets live_code_bench --output ./results_full/

# 2. Pruned run — keep 25%
evalscope eval --model <model> --datasets live_code_bench_pruned \
    --dataset-args '{"pruning_strategy":"stratified_anchor","prune_ratio":0.25}' \
    --output ./results_pruned/

# 3. Compare — reconstructs the full score and checks the go/no-go verdict
python -m evalscope_ext.tools.compare_runs \
    --full ./results_full/ --pruned ./results_pruned/ --threshold 0.6
```

The compare step prints the full score, the pruned estimate, the cost saving, and
whether both land on the **same side of the customer's quality bar**.

## The one honesty rule built into the tool

The pruner **will not let you over-prune a benchmark that cannot support it.** For
the long-context benchmark (AA-LCR), where scores come from an LLM judge and the
signal is thin, the tool reports **"do not prune — run in full."** You get the
aggressive saving where it is safe (coding) and an explicit stop where it is not.
No silent wrong answers.

## What the multimodal probe gives that random sampling cannot

If the customer adds image workloads next quarter, a *random* MMMU sample tells
you the model's average score but not *why* it fails. Our probe deliberately picks
images that break a weak image encoder — dense charts, small text, fine counting —
and contrasts each against a text-only version of the same question. That iso­lates
**"can the model see it"** from **"can the model reason about it,"** so you can tell
a customer specifically whether the *vision* is production-ready, not just quote an
average.

## Why a customer-facing PM should care

It compresses the time-to-answer in an eval from days to minutes for the capability
that can be safely pruned, makes the go/no-go defensible (a measured ±1-point band,
not a vibe), and protects the relationship by never shipping a confident answer the
underlying data cannot support.
