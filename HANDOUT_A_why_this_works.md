# Handout A — Why this works

**Audience:** an engineer who could have built this themselves.
**Code:** `evalscope_ext/` (fork of `modelscope/evalscope`, pinned SHA in fork README).

## The problem I set out to solve

The customer question is *"is this model good enough for our coding / long-context
workload?"* — a **threshold decision against an absolute score**, not a ranking.
That framing dictates everything: a pruned subset has to reproduce the model's
*absolute* full-benchmark accuracy, because that is the number compared to the
go/no-go bar. Preserving only the *ordering* of models is not sufficient.

This rules out the tempting heuristic of "keep the most discriminating items."
On LiveCodeBench, 158 of 315 items are solved by every reference model and 46 by
none; only 111 (35%) discriminate. Selecting the high-discrimination items does
preserve ranking — but it drives every model's subset score toward ~50%, so the
absolute estimate (and the threshold decision) collapses. I verified this: pure
discrimination selection is *worse than random* at recovering absolute accuracy.

## The method: stratified anchor selection with reconstruction weights

In the spirit of Anchor Points / tinyBenchmarks, but specialised for absolute
threshold recovery:

1. **Difficulty from reference models.** Each item's difficulty = fraction of the
   already-run reference models that pass it. Difficulty is an *item* property, so
   it transfers to an unseen model far better than that model's own scores.
2. **Stratify by difficulty** into equal-mass bands; **sample anchors spread
   across each band's difficulty range** (not clustered) to minimise variance.
3. **Reconstruction weights.** Each anchor carries a weight equal to the number of
   full-set items its stratum represents; the weighted mean of anchor scores is an
   approximately unbiased, low-variance estimator of the full-set mean.

Selection never touches the target model's scores — only item difficulty and
ordering — which is what makes it defensible for a fourth, unseen model.

## How much I pruned, and the evidence it is enough

Leave-one-model-out (difficulty estimated on two models, error measured on the
held-out third), averaged over seeds:

| Benchmark | Keep | Items | Anchor MAE | Random MAE | vs random |
|---|---|---|---|---|---|
| LiveCodeBench v5 | **25%** | 79 / 315 | **0.9 pt** | 3.8 pt | **−76%** |
| LiveCodeBench v5 | 30% | 94 / 315 | 1.3 pt | 3.1 pt | −59% |
| AA-LCR | 30% | 30 / 100 | 5.3 pt | 6.1 pt | −12% |

At **25% of LiveCodeBench, the estimate lands within ~1 point of the true score**
for a model never used to build the subset, with worst-case held-out error under
2 points — a 75% cost saving with negligible decision risk.

**AA-LCR is deliberately *not* pruned aggressively.** With only three reference
models, two-model difficulty collapses to three values (0, 0.5, 1) across 100
items, and the LLM judge adds per-item noise (flagged in the task). The shipped
confidence reporter — which measures how much the reconstructed score moves as
anchor placement is jittered — returns **"do not prune"** for AA-LCR at every
ratio, versus **"safe at 15–25%"** for LiveCodeBench. The pruner refuses to give
false confidence on a benchmark whose signal is too thin to compress. That guard
is a feature, not a gap.

## Assumptions

* Per-item difficulty is roughly stationary across models of the same broad
  capability tier (holds within the shipped models; the confidence signal detects
  when it does not).
* Pass/acc are the decision metrics; the weighted mean is the estimator of record.
* The customer cares about an absolute threshold, so bias matters more than
  ranking fidelity.

## What would change with more

* **(a) More data / more reference models.** The single biggest lever. With
  10–20 models, difficulty becomes continuous, real 2-parameter IRT (difficulty +
  discrimination) is fittable, and AA-LCR likely becomes safely prunable. I built
  the stratifier to drop into IRT weights unchanged.
* **(b) A live endpoint.** I would add adaptive stopping — run anchors easy-to-hard
  and halt once the threshold decision is statistically settled, often before the
  full subset is spent.
* **(c) More time.** Cluster items by *content* embeddings (problem text / tags)
  and stratify jointly on difficulty × topic, so the subset is representative in
  capability *and* domain — strengthening 4th-model generalisation further.

## Part B — multimodal probe (design)

To tell cheaply whether a candidate's **image encoder** is good enough on full
MMMU (~12K), probe the *perception* bottleneck, not general reasoning:

* **Select images where a wrong percept forces a wrong answer** — dense small-text
  charts/tables, fine-grained counting, diagrams with many near-identical glyphs,
  low-contrast medical/figure panels. On these, a degraded encoder cannot be
  rescued by strong text reasoning, so accuracy isolates encoder quality.
* **Encoder-stress features** (computable without the model): text-density,
  resolution / smallest-legible-glyph size, object/edge count, colour entropy.
  Stratify the probe on these so it spans the encoder's failure modes rather than
  MMMU's subject mix.
* **Measure through the standard OpenAI chat interface** with a perception-anchored
  contrast: pair each item with a text-only variant where the needed visual fact is
  given in words. *Vision-minus-text accuracy* is the encoder-attributable signal;
  a model that recovers with text but fails with the image has an encoder gap, not
  a reasoning gap. This needs no logits — only two chat calls per probe item.

This is a written design; depth went to Part A per the task's guidance.
