# Handout B — Why this matters and how to use it

**Audience:** developers, test engineers, product, and the customer team.

## What this changes for the customer conversation

Right now, answering *"is this model good enough for your coding workload?"* means
running the full benchmark on every candidate — it is slow and expensive, so it
tends to happen late and rarely. With the pruner, the coding benchmark runs on a
quarter of the items and still lands within about a point of the full score. In
practice that turns a multi-hour, batch-scheduled evaluation into something a
sales engineer can run during a live eval and trust for a go/no-go answer.

The short version: about 75% less compute and time for the coding answer, with
effectively no change to the decision.

## How someone runs it tomorrow

Inside the evalscope fork, three commands:

```bash
# Full run — the baseline you can stop needing once you trust the pruned number
evalscope eval --model <model> --datasets live_code_bench --output ./results_full/

# Pruned run — keep 25%
evalscope eval --model <model> --datasets live_code_bench_pruned \
    --dataset-args '{"pruning_strategy":"stratified_anchor","prune_ratio":0.25}' \
    --output ./results_pruned/

# Compare — reconstructs the full score and checks the go/no-go verdict
python -m evalscope_ext.tools.compare_runs \
    --full ./results_full/ --pruned ./results_pruned/ --threshold 0.6
```

The compare step prints the full score, the pruned estimate, the items saved, and
whether both land on the **same side of the customer's quality bar**.

## The honesty rule built into the tool

The pruner will not let you over-compress a benchmark that cannot support it. For
the long-context benchmark (AA-LCR), where the scores come from an LLM judge and
the usable signal is thin, the tool reports **do not prune — run it in full**. So
you get the aggressive saving where it is safe (coding) and an explicit stop where
it is not. No quiet wrong answers, which is the failure mode that would actually
hurt a customer conversation.

## What the multimodal probe gives you that random sampling cannot

If the customer moves into image workloads next quarter, a random MMMU sample
tells you the average score but not *why* a model fails. The probe deliberately
selects images that break a weak image encoder — dense charts, small text, fine
counting — and pairs each with a text-only version of the same question. That
separates *can the model see it* from *can the model reason about it*, so you can
tell a customer specifically whether the **vision** is production-ready rather than
quoting one blended number.

## Why a customer-facing PM should care

It cuts the time-to-answer on the prunable capability from days to minutes, makes
the go/no-go defensible — a measured error band, not a gut call — and protects the
relationship by never shipping a confident answer the underlying data cannot
support.
