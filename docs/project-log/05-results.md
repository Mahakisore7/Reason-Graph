# Results So Far

All numbers below use **Groq** (`qwen/qwen3.8-27b`) with typed-out confidence. They are
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
| **Held-out run** | 310-segment test half (49 false) | *pending* | *pending* |

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

## 4. Known weaknesses of these numbers

1. **Typed-out confidence, not logprobs.** Needs OpenAI credit to fix.
2. **Transition numbers are guesses** (D12), not learned.
3. **Shallow trees** — depth 1, one correction sample (D16).
4. **NLI quirk** — unrelated pairs sometimes labelled "contradiction" (D10), which can add
   false evidence against a claim.
5. **One small dataset** — FELM-Science only; the paper also uses WikiBio-GPT3 and FactCheckGPT.

## Comparison with the paper

Not done yet. A fair comparison needs the same confidence method (OpenAI logprobs) and
similar tree depth. Until then, compare only **raw vs. tree within our own runs**.
