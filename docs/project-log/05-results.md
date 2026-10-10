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


## 4. Known weaknesses of these numbers

1. **Typed-out confidence, not logprobs.** Fixed by the Llama-3-8B run on Kaggle (D22).
2. **Transition numbers are guesses** (D12), not learned.
3. **Shallow trees** — depth 1, one correction sample (D16).
4. **NLI quirk** — unrelated pairs sometimes labelled "contradiction" (D10), which can add
   false evidence against a claim.
5. **One small dataset** — FELM-Science only; the paper also uses WikiBio-GPT3 and FactCheckGPT.

## Comparison with the paper

Not done yet. A fair comparison needs the paper's confidence method (token probabilities —
coming from the Llama-3-8B Kaggle run) and similar tree depth. Until then, compare only
**raw vs. tree within our own runs**.
