# What We Built

Each phase is on its own git branch, reviewed and merged through a pull request.

## Phase 0 — The math engine

**Branch:** `phase-0/hmt-implementation` · **Merged:** PR #1 · **Commit:** `131e80d`

**What it does:** takes a finished belief tree (claims + confidence scores) and computes the
probability that the top claim is true. No AI, no internet — pure probability.

**How it works, simply:** start at the leaves of the tree and move upward. At each claim, ask:
"how well do the scores below me fit if I'm true, vs. if I'm false?" At the top, combine both
answers with Bayes' rule. This is the **upward pass of a Hidden Markov Tree**.

**Two kinds of parent-child link (transition models):**
- *Independent children* — each child's truth depends on the parent, separately
  (used for premises and corrections).
- *Decomposition* — the parent is true **only if every child is true** (used when a claim is
  split into its separate facts).

**File:** `src/inference/hmt.py`

**How we know it's right:** 8 tests in `tests/test_hmt.py`, including three small trees whose
answer was **worked out by hand first**, then compared to the code to many decimal places.

## Phase 1 — Talking to LLMs

**Branch:** `phase-1/llm-client` · **Merged:** PR #3 · **Commits:** `c8dbc99`, `1e1b80b`

| Piece | File | Why it exists |
|---|---|---|
| Common interface | `src/llm/base.py` | Every LLM looks the same to the rest of the code, so we can swap Groq ↔ OpenAI by changing one line |
| Fake LLM | `src/llm/fake.py` | Canned answers for tests — free, instant, same result every time |
| Disk cache | `src/llm/cache.py` | Saves every answer to disk; asking the same thing twice never costs twice. Also means a crashed run resumes where it stopped |
| Prompts | `src/llm/prompts.py` | All question wording in one place |
| OpenAI client | `src/llm/openai_client.py` | The paper's method: reads the model's internal probability for "True" vs "False". **Works, but account has no credit yet** |
| Groq client | `src/llm/groq_client.py` | Free. Groq doesn't expose internal probabilities, so the model types out a number 0–1 instead — a weaker method |
| Open-source model client | `src/llm/hf_client.py` | Runs an open model ourselves (Llama-3-8B on Kaggle) and reads the real True/False probabilities — the paper's method, free. *(added during Phase 2)* |

**How we know it's right:** cache tests (4); both real clients tested live. OpenAI correctly
reported "no credit" after retrying; Groq returned sensible scores.

## Phase 2 — Building trees and testing on real data

**Branch:** `phase-2/tree-construction` · **Commits:** `eff26d0`, `c1843df`, `7e90e33`, `469d872`
· **Not merged yet**

| Piece | File | What it does |
|---|---|---|
| Strategies | `src/tree/strategies.py` | Decomposition, Premises, Correction — each turns one claim into child claims |
| Tree builder | `src/tree/builder.py` | Runs the strategies level by level up to a max depth; records every claim, its confidence, and its link to its parent |
| Relationship types | `src/tree/relationships.py` | The 5 link types (equivalence, entailment, reverse entailment, contradiction, neutral) and starting transition numbers for each |
| NLI labeller | `src/tree/nli_labeler.py` | A dedicated sentence-relationship model (same one the official code uses), run in both directions |
| Calibration fitting | `scripts/fit_emission_model.py` | Learns "how often is a claim with confidence X actually true?" from labelled data |
| Evaluation | `scripts/evaluate_felm.py` | Builds a tree per test sentence, runs the Phase 0 math, measures accuracy (AUROC) |
| Backend choice | `scripts/backends.py` | `--backend groq` or `--backend hf --model ...`; separate cache and result files per model |
| Kaggle notebook | `notebooks/kaggle_felm_llama3.ipynb` | Runs both scripts on Kaggle's free GPUs with Llama-3-8B |

**How we know it's right:**
- 9 tests for strategies + builder, all using the fake LLM: decomposed children get the
  "equivalence" link, unrelated premises are dropped, correction children get confidence fixed
  at 1.0 and are never re-scored, the tree stops at exactly the max depth.
- 6 tests for how the two NLI directions combine into one link type.
- 3 tests for the fit/test data split.
- 3 tests for the True/False probability arithmetic. The open-model client was also run for
  real with a small model (Qwen2.5-0.5B): true statements scored 0.95 / 0.82, false ones
  0.24 / 0.03, and the whole fit → evaluate flow completed.
- NLI model checked live: "in Paris" → "in France" correctly labelled one-way (entailment),
  a paraphrase labelled equivalence, Berlin vs. Paris labelled contradiction.
- The tree → math conversion was checked on a hand-built tree: a strongly supporting,
  confident child pushes an uncertain parent upward, as it should.

**35 tests total, all passing.** Every commit also passes automatic checks: `ruff` (style),
`mypy --strict` (types), `pytest`.

## Not built yet

- Phase 3: DAG + loopy belief propagation (novelty 1)
- Phase 4: surrogate confidence scorer (novelty 2)
- Phase 5+: full pipeline assembly, full evaluation on all three datasets, final report
