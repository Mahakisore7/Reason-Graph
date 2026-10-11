# Results So Far

Sections 1–3 use **Groq** (`qwen/qwen3.8-27b`) with typed-out confidence. They are
**provisional** — not the paper's confidence method. See decision D6 in
[03-decisions.md](03-decisions.md).

## 1. Calibration table — how much can we trust the LLM's confidence?

We asked the LLM for its confidence on every fit-half segment, sorted the answers into 5
buckets, and counted how many in each bucket were really true or false.

**Fit half (300 segments: 247 true, 53 false)** — this is the table actually used for testing:

| LLM confidence | % of true segments here | % of false segments here |
|---|---|---|
| 0 – 0.19 (very unsure) | 17% | 40% |
| 0.19 – 0.39 | 4% | 2% |
| 0.39 – 0.69 | 7% | 8% |
| 0.69 – 0.89 | 4% | 11% |
| 0.89 – 1.0 (very sure) | 68% | 40% |

**How to read it:**
- False segments land in "very unsure" more than twice as often as true ones → low confidence
  is a real warning sign.
- But **40% of the false segments got "very sure" scores.** The LLM is often confidently
  wrong — exactly the problem the paper targets.
- The middle buckets are nearly empty: the model mostly answers ~0 or ~1, rarely in between.
  That's a side effect of typed-out confidence (D6).

*Earlier version, fit on all 684 segments (superseded by D18):* very sure = 66% of true vs.
46% of false; very unsure = 21% vs. 36%. Same story.

## 2. Accuracy — does the tree help?

**Measure: AUROC.** Pick one true and one false segment at random. AUROC is how often the
method gives the true one the higher score. 0.5 = coin flip, 1.0 = perfect.

| Run | Test data | Raw LLM confidence | After belief tree |
|---|---|---|---|
| First run *(flawed — see below)* | 60 random segments (7 false) | 0.64 | 0.68 |
| Held-out run | 310-segment test half (49 false) | in progress (243 of 310 done) | — |

**Why the first run doesn't count:** its calibration table was built from data that included
its own test sentences (fixed in D18), and 7 false examples is far too few for a stable
AUROC. It's kept here as history, not as a result.

## 3. What we saw in individual cases (first run)

| Actual | Raw confidence | After tree | Effect |
|---|---|---|---|
| False | 0.90 | 0.47 | Helped — confidently wrong answer pulled down |
| True | 0.20 | 0.40 | Helped — wrongly doubted answer pulled up |
| False | 0.90 | 0.55 | Helped |
| True | 0.80 | 0.25 | **Hurt** — a correct answer pushed the wrong way |
| False | 0.90 | 0.78 | Barely moved — still confidently wrong |

The tree is not a uniform improvement. Some trees contain misleading children, and the
hand-set transition numbers (D12) may be too strong or too weak.

## 3b. Llama-3-8B on Kaggle — the paper's confidence method

Run on Kaggle's free GPUs with **Llama-3-8B-Instruct** (one of the paper's two models), reading
real True/False token probabilities. Calibration fit on the fit half, tested once on the test
half (304 segments, 49 false, after removing list openers — D26). Every variant is scored on the
same sentences and all are reported (D28).

| What's scored | AUROC |
|---|---|
| Llama's raw confidence | **0.670** |
| 5-bucket table only | 0.420 |
| 5-bucket table + tree | 0.499 |
| Smooth curve only (D27) | 0.670 — same as raw; a monotone curve never reorders |
| **Smooth curve + tree** | **0.588** |

**Headline: in our setup, the belief tree lowers accuracy (0.670 → 0.588).** This is not yet a
faithful reproduction; three reasons below, all found by inspecting the data and comparing with
the paper's own published code and trees.

### Why — what the investigation found

**1. The 5-bucket table broke on small data.** With 53 false examples over 5 buckets, it learned
that *very low* confidence means *true* (bucket 0: 18 true vs 3 false). Cause: non-claims
(list openers — now removed) and sentences that need context. The smooth curve fixes this part.

**2. Sentences that need context can't be judged alone.** 84 test sentences start with
"This / They / It / However…" ("They are typically located on the wings…").

| | Sentences | Raw AUROC | Curve + tree AUROC |
|---|---|---|---|
| Needs context | 84 (12 false) | 0.399 — worse than a coin flip | 0.361 |
| Stands alone | 220 (37 false) | **0.745** | 0.667 |

**3. Llama over-decomposes, and decomposition uses the strictest rule.** Llama split **297 of
304** sentences into ~3.5 "sub-facts" each, true and false alike. Decomposition says *the parent is
true only if every sub-fact is true*, so one noisy sub-fact score drags the sentence down. Sub-facts
are often meaningless or invented:

> "The question is, who shaves the barber?" → "There is a question." / "There exists a barber."
>
> "As of now, OpenAI has not released GPT-4 yet." → adds "OpenAI has released GPT-3" — not in the
> original sentence.

Keeping only some children (curve calibration): decomposition children only 0.601; premise
children only 0.641; full tree 0.588. The NLI labels were *not* the problem — they agreed with
Llama's intent (all 17 "contradicting" premises labelled contradiction).

### Comparison with the paper's own code and trees

From the official repo (`UCSB-NLP-Chang/BTProp`):

| | Paper (FELM trees, ChatGPT) | Ours (Llama) |
|---|---|---|
| Decomposition | 390 expansions (26%) | 297 (98%) |
| Premises ("logical relation") | 679 (46%) | ~7 |
| Correction ("perturbation") | 411 (28%) | ~7 |

- **Decomposition prompt:** theirs is long, with worked examples — including one sentence that
  stays a single claim and one with no checkable claim — and demands full names, no pronouns, no
  overlapping claims. Ours was one loose instruction with no examples.
- **Strategy choice:** they ask the LLM whether to use premises, correction, or both. We run both
  (D8) — but decomposition almost always fired first, so this rarely mattered.
- **Data:** they evaluate on a **human-preprocessed** FELM-Science file
  (`felm_science_preprocess_v4.json`, not public) whose statements are already rewritten to stand
  alone (e.g. "If you … swallow a few watermelon seeds …, there is no need to worry." → "Swallowing
  a few watermelon seeds while eating the fruit … does not require any worry."). We use raw FELM
  segments.

**So our trees are not the paper's trees yet.** Next step: adopt their prompts and make sentences
stand alone before building trees.


## 3c. Llama-3-8B, the paper's own method + stand-alone sentences (D29, D30)

Second Kaggle run: the paper's prompts and tree rules, every sentence first rewritten to stand
alone. Same split: calibration fit on 291 sentences, tested once on 304 (49 false).

**Stand-alone rewriting:** 385 of 594 sentences rewritten; 122 proposed rewrites rejected by the
safety check (original kept). Examples:
- kept: "No, WiFi generally has a longer range than Bluetooth." → "WiFi generally has a longer
  range than Bluetooth."
- kept: "The distance between the slits is…" → "The distance between Young's two slits is…"
- rejected, correctly: "Yes, there can be an explosion without fire." → a different sentence
  about causes of explosions.
- rejected, too strictly: "They code for traits…" → "Genes code for traits…". The check is
  conservative: when unsure, it keeps the original.

| What's scored | First run (3b) | This run | 95% range (bootstrap) |
|---|---|---|---|
| Llama's raw confidence | 0.670 | **0.745** | 0.67–0.81 |
| 5-bucket table only | 0.420 | 0.635 | 0.54–0.73 |
| 5-bucket table + tree | 0.499 | 0.678 | 0.58–0.77 |
| Smooth curve only | 0.670 | 0.745 | (same as raw) |
| **Smooth curve + tree** | 0.588 | **0.741** | 0.65–0.82 |

**What this shows:**
- Stand-alone sentences were the biggest single fix: raw AUROC 0.670 → 0.745.
- The tree no longer hurts (was −0.08), but does not clearly help: curve + tree minus raw is
  −0.004, with a bootstrap range of −0.05 to +0.04 (2,000 resamples of the test sentences) — no
  real difference. With the table, the tree adds +0.04 (range −0.04 to +0.13), also not
  significant.
- 50 sentences got no children (nothing checkable to expand); 254 got a tree.
- The paper reports BTProp beating the *best baseline* by about 3–9% on FELM-Science. Likely
  reasons we see no gain: trees of depth 1, not 2 (D16, free-GPU time limit); the paper's
  hand-cleaned dataset (not public); one test set of 49 false sentences, so differences under
  ~0.05 cannot be told apart from chance.

## 3d. Our extension 1 — one graph per answer, loopy BP (D32, D33)

Same 304 test sentences, same trees and calibration as 3c; only the links between sentences of
the same answer are new. Run on the laptop GPU (RTX 4050) from the saved Llama answers.

- 61 answers, **168 links** between sentences: 96 contradiction, 53 entailment, 15 reverse
  entailment, 4 equivalence. 32 answer graphs had real loops.
- **Convergence: 122 of 122 runs converged** (all 32 looped ones too), at most 76 rounds.

| What's scored | Tree (paper) | Graph (ours) |
|---|---|---|
| 5-bucket table | 0.678 | **0.697** |
| Smooth curve | **0.741** | 0.728 |

Only answers that got at least one link (231 sentences, 33 false): table 0.675 → 0.718, curve
0.783 → 0.772.

**What this shows:** no clear gain. Graph minus tree with the curve is −0.014 (bootstrap range
−0.07 to +0.04); with the table it is +0.02. Neither is beyond chance. The likely culprit is the
links themselves: 96 of 168 are "contradiction", far more than two sentences of one ChatGPT
answer really contradict each other. The NLI model tends to call two different facts about one
topic a contradiction (known weakness 4). Loopy BP itself behaved well: it always settled, and
on trees it matches the exact math. **Next step if time allows:** keep a contradiction link only
when NLI is confident, or only entailment links, decided on the fit half — not the test half.

## 3e. Our extension 2 — cheap surrogate confidence scores (D34)

812 LLM-scored nodes (304 sentences + 508 premises/claims). Each node's score predicted by a
surrogate that never saw its answer.

**How close to Llama (mean absolute error):** surrogate 0.247 vs. always guessing the average
0.258. It learned only a little — hashed words carry little of what Llama knows.

| Llama scores replaced | Any node: curve AUROC | Children only: curve AUROC | Calls saved (children only) |
|---|---|---|---|
| 0% | 0.741 | 0.741 | 0 of 812 |
| 25% | 0.727 ± 0.005 | 0.744 ± 0.006 | 127 |
| 50% | 0.700 ± 0.016 | 0.748 ± 0.010 | 254 |
| 75% | 0.677 ± 0.023 | 0.751 ± 0.012 | 381 |
| 100% | 0.627 | **0.757** | 508 (63%) |

(With the 5-bucket table, every replacement lowers AUROC: 0.678 → 0.553 any node, → 0.609
children only.)

**What this shows:**
- Replacing a sentence's *own* score hurts steadily — that score is the main evidence.
- Replacing only the children's scores **saves 63% of the confidence calls with no loss** in
  the curve setting (0.741 → 0.757, within chance). Honest reading: the surrogate's guesses sit
  near the average, so children then carry little evidence — which tells us the children's real
  Llama scores were adding little anyway (matches 3c: tree ≈ raw).
- Only confidence calls are saved; generating premises and corrections still costs LLM calls.

## 4. Known weaknesses of these numbers

1. **Typed-out confidence, not logprobs.** Fixed by the Llama-3-8B run on Kaggle (D22).
2. **Transition numbers are guesses** (D12), not learned.
3. **Shallow trees** — depth 1, one correction sample (D16).
4. **NLI quirk** — unrelated pairs sometimes labelled "contradiction" (D10), which can add
   false evidence against a claim.
5. **One small dataset** — FELM-Science only; the paper also uses WikiBio-GPT3 and FactCheckGPT.

## Comparison with the paper

Same model (Llama-3-8B-Instruct), same confidence method, the paper's prompts. Differences that
remain: tree depth 1 vs. 2, our automatic stand-alone rewriting vs. their hand-cleaned file. Our
result: the tree method matches raw confidence (0.741 vs. 0.745) instead of beating it — see 3c.
