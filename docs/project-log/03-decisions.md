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

## Adopting the paper's own method (Step B)

**D29. Build trees exactly as the official BTProp code does.**
- *Why:* our trees were 98% decomposition; the paper's FELM trees are 26% (05-results §3b). Our
  prompts were my own wording. A reproduction should use the paper's method, so any difference in
  results is about the method, not my prompt writing.
- *What changed:*
  - **Prompts copied word for word** from the official repo (`src/prompts.py`, Llama versions):
    confidence, decomposition, strategy choice, supportive premises, key-question correction.
    Verified identical by script. The repo has no licence file; the texts are reproduced with
    attribution for research reproduction. (Changes D1: we now use their prompts, still our own
    code.)
  - **Strategy choice by the LLM** (premises / correction / both), as in the paper. Replaces D8.
  - **Premises: supportive only**, 2 samples, at most 3 kept after the NLI filter. The official
    code has a contradicting-premise branch, but a `for/else` there always empties it, so their
    trees have supportive premises only — we follow what the code does.
  - **Corrections are evidence now:** 5 sampled answers, statement revised to match each, every
    revision NLI-labelled against the original; a revision that contradicts the original counts
    against it. Confidence fixed at 1.0 with the paper's own correction emission (0.8 if true vs
    0.2 if false). Replaces D11, which had switched this evidence off.
  - **Link probabilities from the official code** (src/hmm.py): equivalence 1.0/0.0; parent ⇒ child
    1.0/0.5; child ⇒ parent 0.5/0.0; contradiction 0.0/0.5. Replaces my guesses (D12), which had a
    logic error: "parent false" is no evidence about a contradicting child (0.5), not 0.95.
  - **Every child NLI-filtered**, decomposition claims included; only the root may be decomposed.
  - **Sampling done properly:** new `complete_samples(prompt, n)`; repeated `complete()` calls
    would have replayed one cached answer n times.
- *Small deviations, on purpose:*
  - Few-shot examples are sent in one message; the official Llama code splits them into chat turns.
  - Two official parsing slips fixed: `line.strip("Question: ")` strips *characters* (can eat the
    end of a question); fixed-width slicing breaks from "Claim 10:" on.
  - A root with "no check-worthy claims" keeps its own confidence; the official code scores such
    statements 1.0 (definitely true).
  - Tree depth stays 1 (paper: 2) — D16.

**D30. Make each segment stand alone before checking it.**
- *Why:* the paper used a hand-cleaned FELM-Science file (not public) where sentences already stand
  alone, and its confidence prompt says to treat statements needing context as false. On our raw
  segments, "This/They/It…" sentences scored AUROC 0.40 even before any tree.
- *How:* Llama rewrites each segment using its question and full answer as context, told not to
  add or correct anything. Labels never change.
- *Safety check:* a rewrite is kept only if the NLI model says it **entails the original** (still
  says everything the original said). Found necessary in testing: a small model dropped the clause
  "which is hot enough to melt lead" — possibly the false part — and swapped one sentence for
  another; both were rejected and the originals kept.

**D31. Version every cache entry by method.**
- *Found:* cached confidence scores are keyed by statement text, not by prompt, so after a prompt
  change old scores would be silently reused.
- *Now:* every cache tag ends in a method version (`btprop`). Earlier runs are untouched, under
  their old tags.
- *Consequence:* the unfinished Groq held-out run (243 of 310, our old prompts) is superseded and
  won't be continued; the Llama run with the official method replaces it.

## Phase 3 — graph instead of tree (our novelty, extension 1)

**D32. Loopy BP on a factor graph with joint factors, not pairwise edges only.**
- *Why:* the Phase 3 plan allowed pairwise links only, which cannot express the paper's
  decomposition rule ("parent true only if *all* its claims are true" is a rule over the whole
  group at once). With factors that can cover any number of claims, every paper tree carries over
  unchanged, so the graph result is *exactly* the tree result whenever no extra links are added.
- *Checked:* on the Phase 0 tree the graph code gives 0.8209483368719038, matching to 12 decimals
  (independent-children and decomposition trees both); on a looped "diamond" it is within 0.05
  of a brute-force answer from all 16 true/false combinations, and — as expected on a real loop —
  not exact.
- *Choices written down:* synchronous ("flooding") updates; damping 0.5; tolerance 1e-6; at most
  200 rounds. Convergence is reported per run, never assumed (0 rounds ⇒ "not converged").

**D33. Where the graph comes from: one graph per FELM answer.**
- *Idea:* the paper checks every sentence of an answer in isolation. But sentences of one answer
  are about the same thing and can support or contradict each other. We join the trees of all
  sentences in an answer, adding a link wherever NLI says two sentences are related (same four
  relationships and probabilities as tree links). A sentence linked to two others has two
  "parents", so loops appear and loopy BP replaces the exact tree recursion.
- *Softened links:* cross-sentence link probabilities are kept within [0.05, 0.95], so one NLI
  mistake can make two sentences unlikely together but never impossible. The value 0.05 was fixed
  before seeing any test result and is not tuned.
- *Fair comparison:* same test sentences, same trees (same LLM cache), same calibration; only the
  links differ. Both "tree" and "graph" AUROC are reported for both calibrations.
- *Limit:* links are between sentence roots only (not premises of different sentences), to keep
  the number of NLI calls small; a possible follow-up.

## Phase 4 — cheaper confidence scores (our novelty, extension 2)

**D34. How the surrogate is trained and tested.**
- *What it is:* hashed words and word pairs (4096 slots, no vocabulary, nothing downloaded) →
  ridge regression → clipped to [0, 1]. It predicts the **LLM's confidence**, never the human
  truth label, so it can stand in for an LLM call.
- *4096 slots, not the plan's 256:* with word pairs, 256 slots would make many unrelated words
  collide.
- *No leakage:* test answers are split into 5 groups; each group's nodes are predicted by a
  surrogate trained on the other 4 groups plus the fit-half sentences (whose Llama scores already
  exist from fitting the calibration). A node is never predicted by a model that saw its answer.
- *Honest baseline:* its error is reported next to "always predict the average confidence"; it
  has to beat that to count as having learned anything.
- *The curve:* 0, 25, 50, 75, 100% of LLM-scored nodes replaced (random, 5 repeats, mean ± std),
  tree AUROC recomputed each time. Two schemes: any node, or child nodes only (root keeps its
  real score). Corrections are not counted — their confidence is fixed at 1.0 and never costs a
  call.
- *Limit, stated plainly:* only the confidence calls are saved. Generating premises and
  corrections still needs the LLM, so the saving is a share of confidence calls, not of all calls.

## Phase 7 — evaluation

**D35. Which links between sentences to keep — chosen on the fit half.**
- *Problem found (results 3d):* 96 of 168 cross links were "contradiction", mostly two different
  facts about one topic. They pushed true sentences down.
- *Three rules:* `any` (the paper's labeller), `strict` (contradiction only if NLI says so in
  both directions), `no_contradiction` (drop contradiction links, keep entailment-type links).
- *Chosen without the test half:* each rule scored on the fit half (one node per sentence with
  its Llama confidence, linked under the rule; no new LLM calls). Fit-half AUROC: none 0.724,
  any 0.699, strict 0.727, **no_contradiction 0.736** → chosen, then run once on the test half.
- *Cost of the choice:* contradictions between sentences are no longer used at all; a real one
  ("…in Berlin" / "…in Paris") is now missed. A better contradiction detector could bring them back.

**D36. AUC-PR and F1, as the paper reports.**
- Positive class = false statement (a hallucination); hallucination score = 1 − P(true).
- F1 needs a cut-off. It is cross-fitted: answers split into 5 groups, each group classified with
  the cut-off that maximises F1 on the other 4. No sentence helps choose its own cut-off.
- Significance by *cluster* bootstrap: whole answers resampled (sentences of one answer are not
  independent), 2,000 times.

**D37. Study of loopy BP against exact inference as loops increase (mentor feedback).**
- *Why:* the Review 1 plan promised, "per mentor feedback", a comparison of exact inference with
  loopy BP on graphs of increasing loopiness. Until now we had one small looped graph checked
  against brute force, plus convergence on real graphs.
- *Design:* random 10-node binary graphs = a random spanning tree plus 0, 1, 2, 3, 4, 6, 8 or 10
  extra edges (the number of independent cycles). 200 graphs per setting, two coupling regimes
  (mixed link types like our real graphs; all strong equivalence links, the hard case), with and
  without damping. Exact marginals by brute force over all 2^10 assignments.
- *Result:* exact on trees (error 0); mean error of P(true) rises with loops — mixed 0.023 at
  1 cycle, 0.094 at 3, 0.148 at 10; strong 0.096 at 1, 0.159 at 3, 0.193 at 10; worst single
  error up to 0.85. Convergence at least 99.5% undamped. Damping 0.5 did not help (at least 97%,
  about twice the rounds), so it is a safe but unnecessary choice; every real run converged.
- *What it means for our results:* of the 31 test answers that got any cross link, 26 are
  loop-free, where BP is exact; only 5 contain a cycle. Our graph gains therefore come mostly from
  linking sentences, not from the loopy part — loopy BP matters for the minority with cycles, and
  would matter more on denser graphs, where this study shows its error is large.
