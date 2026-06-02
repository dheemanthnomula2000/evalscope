# Handout A — Why this works

**Audience:** an engineer who could have built this themselves.
**Code:** `evalscope_ext/`, developed against evalscope `de7b0b3f08c617f48a00ef09f7169dc74212a6d9`.

## What I understood the problem to be

The customer is asking one thing: *is this model good enough for our coding and
long-context workload?* That is a threshold decision against an **absolute**
score — the model's accuracy gets compared to a quality bar. It is not a ranking
problem. I leaned on that distinction the whole way through, because it rules out
methods that look reasonable but answer the wrong question.

The first thing I did was just count agreement. On LiveCodeBench, of 315 items,
158 are solved by all three reference models and 46 by none — only 111 (35%)
actually separate the models. My first instinct was the obvious one: keep the
most discriminating items and drop the rest. I tested it before trusting it, and
it was **worse than random** at recovering the absolute score. In hindsight that
is obvious — high-disagreement items sit near 50% for every model, so a subset of
them drives every model's estimate toward 50% and destroys the number the
threshold actually cares about. It preserves *ranking* and throws away *level*.
That failed experiment is what pointed me at the right framing.

## The method I settled on

Stratified anchor selection with reconstruction weights — the same idea behind
Anchor Points / tinyBenchmarks, but aimed specifically at recovering the absolute
score rather than the ranking:

1. Estimate each item's **difficulty** as the fraction of already-run reference
   models that pass it. Difficulty is a property of the item, so it transfers to
   a model I have not seen far better than that model's own (unknown) scores.
2. Sort by difficulty and split into equal-mass strata.
3. Within each stratum, take anchors **spread across the difficulty range** rather
   than clustered — this lowers the variance of the estimate.
4. Give each anchor a **reconstruction weight** equal to the number of full-set
   items its stratum stands in for. The weighted mean of the anchors is then an
   approximately unbiased, low-variance estimate of the full-set score.

Selection never touches the target model's results — only item difficulty and
ordering from the reference models. That is deliberate: it is what makes the same
anchor set valid for a fourth model I was not given.

## How far I pruned, and why I trust it

I validated leave-one-model-out: estimate difficulty on two models, measure the
error on the held-out third, averaged over seeds.

| Benchmark | Keep | Items | Error vs full | vs random |
|---|---|---|---|---|
| LiveCodeBench v5 | 25% | 79 / 315 | **~0.9 pt** (worst case < 2 pt) | −76% |
| LiveCodeBench v5 | 30% | 94 / 315 | ~1.3 pt | −59% |
| AA-LCR | 30% | 30 / 100 | ~5.3 pt | −12% |

On coding, keeping a quarter of the benchmark estimates an unseen model's score
to within about one point — a 75% saving with negligible decision risk. I would
sign off on that.

AA-LCR I deliberately do **not** prune hard, and I want to be explicit about why,
because it is the part I am least willing to overstate. With three reference
models, two-model difficulty collapses to three values (0, 0.5, 1) across 100
items — you cannot meaningfully stratify on that. On top of it, AA-LCR is
LLM-judged, so a chunk of the per-item variance is judge noise, not real
difficulty (the task flags this, and it matches what I saw). So I built a
confidence check into the tool: it measures how much the reconstructed score
moves as the anchor placement is jittered. For LiveCodeBench it reports *safe* at
15–25%; for AA-LCR it reports **do not prune** at every ratio I tried. The pruner
refusing to compress a benchmark it cannot compress is a feature — I would rather
ship "run this one in full" than a confident wrong number.

## Assumptions I made

- Item difficulty is roughly stable across models of a similar capability tier.
  It holds across the three shipped models; the confidence signal is there to
  catch the case where it does not.
- `pass` / `acc` are the decision metrics, and the weighted mean is the estimator
  of record.
- The customer cares about an absolute threshold, so bias matters more than
  ranking fidelity — which is the assumption the whole method is built around.

## What I would change with more

- **More reference models** is the biggest lever by far. With 10–20, difficulty
  becomes continuous, a real two-parameter IRT fit (difficulty + discrimination)
  becomes possible, and AA-LCR likely becomes safely prunable. I wrote the
  stratifier so those IRT weights drop in without restructuring it.
- **A live endpoint** would let me run anchors easy-to-hard and stop as soon as the
  threshold decision is statistically settled — often before the full subset is
  spent.
- **More time** — I would cluster items by content (problem text / tags) and
  stratify on difficulty × topic jointly, so the subset is representative in
  capability and in domain, which should tighten 4th-model generalisation further.

## Part B — multimodal probe (design)

For the full ~12K MMMU set, the question is narrower than general capability: is
the candidate's **image encoder** good enough? So I would probe the perception
bottleneck specifically.

- **Pick images where a wrong percept forces a wrong answer** — dense small-text
  charts and tables, fine-grained counting, diagrams with many near-identical
  glyphs, low-contrast figure panels. On these, strong text reasoning cannot
  rescue a weak encoder, so accuracy isolates encoder quality.
- **Stratify on encoder-stress features** that need no model to compute:
  text density, resolution / smallest legible glyph, object and edge count, colour
  entropy. That makes the probe span the encoder's failure modes rather than
  MMMU's subject mix.
- **Measure through the standard OpenAI chat interface** with a perception-anchored
  contrast: pair each image item with a text-only version where the needed visual
  fact is stated in words. *Vision-minus-text accuracy* is the encoder-attributable
  signal — a model that recovers with the text version but fails with the image
  has an encoder gap, not a reasoning gap. Two chat calls per item, no logits
  needed.

I kept Part B as a design and put the depth into Part A, per the task's guidance.
