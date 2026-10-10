# Decision Log

Every real choice we made, the alternatives, and the reason. Newest at the bottom of each
section. When you make a new decision, add it here.

## Project setup

**D1. Write our own implementation; don't copy the official code or our teammate's.**
- *Options:* run the official BTProp repo; reuse teammate's `BT-Prop` code; write our own.
- *Chosen:* write our own, phase by phase. Use the official repo only as a reference for facts
  (which NLI model, which bin edges, which datasets).
- *Why:* we need to understand and explain every part. Our novelty (DAG, surrogate scorer)
  also needs code we fully control.

**D2. Keep the package flat as `src/`.**
- *Why:* the repo is already named Reason-Graph; an extra nested package name adds nothing.

**D3. Every LLM sits behind one common interface, with a fake version for tests.**
- *Why:* tree-building logic can be tested for free and deterministically, and switching
  Groq ↔ OpenAI is a one-line change.

**D4. Cache every LLM answer on disk.**
- *Why:* never pay twice for the same question; interrupted runs resume for free.

## Which LLM

**D5. OpenAI (`gpt-3.5-turbo-0125`) for the final numbers.**
- *Why:* it's one of the two models the paper itself used, and it exposes **logprobs** — the
  model's internal probability for the words "True" and "False". That's the paper's
  confidence method.

**D6. Groq for development and provisional results.**
- *Why:* OpenAI account has no credit; Groq is free.
- *Catch:* Groq rejected logprobs on **every model we tried** (tested live, not assumed).
  So on Groq we ask the model to type a number 0–1. This is weaker: models tend to answer
  round numbers like 0.9 or 1.0. **Groq numbers are always labelled provisional.**

**D7. Groq model: `qwen/qwen3.8-27b`.**
- `llama-3.3-70b-versatile` no longer exists on Groq (404).
- `openai/gpt-oss-20b` is a "reasoning" model: it spent its token budget thinking and
  returned an empty answer.
- `qwen/qwen3.8-27b` answers directly.

## Building the tree

**D8. When a claim doesn't decompose, run Premises *and* Correction (not one or the other).**
- *Options:* (a) ask the LLM which strategy to use, as the paper does; (b) run both.
- *Chosen:* (b). Simpler, more thorough; costs more LLM calls per node.

**D9. Always label a link as `label(child, parent)` and run NLI in both directions.**
- A one-direction NLI model only says "entails / contradicts / neutral". Running both
  directions tells us *which way* the implication goes:
  - both entail → equivalence
  - child ⇒ parent only → entailment
  - parent ⇒ child only → reverse entailment
  - either contradicts → contradiction
  - neither → neutral (child dropped)

**D10. NLI model: `cross-encoder/nli-deberta-v3-large`.**
- *Why:* same as the official repo. We read the label order from the model itself instead of
  assuming it.
- *Known quirk:* on two completely unrelated sentences it sometimes says "contradiction"
  instead of "neutral".

**D11. Correction children: confidence fixed at 1.0, not NLI-labelled, treated as neutral
evidence about the parent.**
- *Why:* they are rewritten alternatives of the claim, not separate facts, so a
  support/contradict label doesn't fit them. The 1.0 rule comes from the plan.

**D12. Starting transition numbers (how strongly a child's truth follows its parent's).**

| Link | P(child true \| parent true) | P(child true \| parent false) | Reasoning |
|---|---|---|---|
| Equivalence | 0.95 | 0.05 | Same truth value almost always |
| Entailment (child ⇒ parent) | 0.65 | 0.05 | If parent is false, child must be false (contrapositive). Parent true says little |
| Reverse entailment (parent ⇒ child) | 0.95 | 0.35 | If parent is true, child must be true. Parent false says little |
| Contradiction | 0.05 | 0.95 | Opposite truth values almost always |

- **These are reasoned guesses, not learned from data.** Fitting them is an open item.

## Data and evaluation

**D13. Dataset: FELM-Science.** One of the paper's three benchmarks, openly licensed, small
enough to afford. Details in [04-dataset-felm-science.md](04-dataset-felm-science.md).

**D14. Load the raw `all.jsonl` file directly from HuggingFace.**
- *Why:* the dataset's official loader is a script that the current `datasets` library no
  longer runs.

**D15. Calibration buckets: same 5 as the official repo (edges 0.19, 0.39, 0.69, 0.89).**
- *Why:* keeps our table comparable with theirs, even though we fit it ourselves.

**D16. Tree depth 1, one correction sample, during evaluation.**
- *Why:* cost. Depth 2 multiplies LLM calls per sentence many times over on a free tier.
- *Effect:* our trees are smaller than the paper's — expect weaker gains.

**D17. Tree → math conversion lives in `scripts/`, not `src/`.**
- *Why:* it's a quick bridge to get a Phase 2 number. The clean, reusable version is planned for
  Phase 5, once the DAG and surrogate scorer exist.

**D18. Build the calibration table and test on separate halves of the data.** *(fix)*
- *Mistake found:* the first version built the table from all 684 sentences, then tested on 60
  of those same sentences — like seeing the exam answers before the exam.
- *Now:* split whole answers 50/50. One half builds the table, the other is only used for
  testing.
- *Split by answer, not by sentence:* sentences from the same answer share topic and wording,
  so a sentence-level split would still leak.
- *Stratified:* answers with at least one error are split separately from error-free answers.
  A plain shuffle gave 28 false examples to one half and 74 to the other.
- *Seed fixed at 42:* we did not try several seeds and keep the one that looked best.

**D19. Remove non-claims and duplicates, found by reading the data.**
- 64 segments under 4 words, e.g. `"2."`, `"Sure!"`, `"Regenerate response"`, `"Mud."`. All
  are labelled true, so they would make any method look better. Real claims start around 5
  words, so the cutoff is 4.
- 2 answers appear twice, word for word. One copy kept.

**D20. Test on the whole test half (310 sentences), not a 60-sentence sample.**
- *Why:* the 60-sentence sample had only 7 false examples — too few for a stable AUROC.

## Moving to open-source models

**D21. Free, open-source models only. No paid APIs.**
- *Why:* no budget. This replaces D5: OpenAI is no longer the path to the final numbers.
- *Bonus:* running an open model ourselves gives the real True/False token probabilities — the
  paper's method — which Groq never could. So this is both free *and* more faithful.

**D22. Final reproduction model: Llama-3-8B-Instruct, run on Kaggle's free GPUs.**
- *Why this model:* it's one of the two models the paper itself evaluates, and it's open-source.
- *Why Kaggle:* the model is 16 GB; the laptop GPU has 6 GB and would run out of memory. Kaggle
  gives 2 × 16 GB T4 GPUs free (~30 h/week), and runs continue with the laptop closed.
- *Which copy:* `NousResearch/Meta-Llama-3-8B-Instruct` — the same weights as Meta's official
  copy (same 16.1 GB, same Llama 3 licence), but downloadable without waiting for Meta's manual
  approval.
- *How:* `src/llm/hf_client.py` loads the model with `transformers`, in float16 (T4s don't
  support bfloat16), split across both GPUs.

**D23. One Groq account only — no rotating several accounts' keys.**
- *Considered:* using 6 Gmail accounts' keys to get 6× the daily limit.
- *Rejected:* Groq's Services Agreement §6.3(d)(iii) forbids use "in a manner intended to avoid
  incurring Fees", and §7 allows suspension without notice. Risk of losing access mid-project.
- *Instead:* the Groq run continues within one account's limit (1,000 requests/day), resuming
  from cache each day.

**D24. Stricter True/False prompt.**
- *Before:* `True or False? {statement}`. *Now:* "Is the following statement true or false?
  Answer with exactly one word: True or False."
- *Why:* we read the model's *first* word. Chat models otherwise open with "The statement is…",
  hiding the answer. (Groq uses its own prompt, so Groq results are unaffected.)

**D25. Decomposition keeps numbered lines only.**
- *Found by:* testing with a small open model, which wrote an intro sentence before its list
  and a remark after it — both would have become fake claims in the tree.
- *Checked:* none of Groq's 138 decomposition answers had extra lines, so earlier Groq results
  are unaffected.

## After the first Llama-3-8B run (calibration fix)

**D26. Also remove list openers.**
- *Found by:* reading the sentences Llama rated near 0 but labelled true.
- *What:* segments like "Sure, here are some papers on summarization:\n\n1." — an intro plus the
  first list number. 16 in the raw data, all labelled true. A line break before the number is
  required, so "The ratio is 3:1." is kept.

**D27. Smooth calibration curve alongside the 5-bucket table.**
- *Problem:* with 53 false examples, the table flipped direction between buckets (very low
  confidence counted as evidence of *true*), dropping AUROC from 0.66 to 0.42 before any tree.
- *Chosen:* logistic regression of the label on logit(confidence) (Platt scaling), fit on the fit
  half. Two numbers instead of ten, and a higher confidence can never count as weaker evidence.
- *Deviation from the paper:* the paper uses a binned table, fit on more data. We keep the table
  and report both.

**D28. Report every variant; disclose the second look at the test half.**
- *Why:* the Llama results were the first look at the test half; the fixes above were the second.
  Both fixes target problems visible in the **fit half** (the flipped table, the list openers),
  not test-half scores. To avoid quietly picking the best-looking option, every variant (raw,
  table, curve, with and without the tree) is scored on the same sentences and all are reported.
