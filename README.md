# Reason-Graph

**Catching LLM hallucinations with belief propagation.** A from-scratch reproduction of
BTProp — *“A Probabilistic Framework for LLM Hallucination Detection via Belief Tree
Propagation”* (Hou et al., NAACL 2025) — plus two extensions of our own:

1. **One graph per answer, solved with loopy belief propagation** — sentences of the same answer
   are linked when they support each other, instead of each being checked alone.
2. **A cheap surrogate confidence scorer** — a tiny text model stands in for some of the LLM's
   confidence calls.

Course project for 22AIE301 Probabilistic Reasoning. Open-source models only; no paid APIs.

## Results (FELM-Science, Llama-3-8B-Instruct)

Test half: 304 sentences from 61 ChatGPT answers, 49 of them false. Positive class = false
sentence. Calibration and every choice were made on the separate fit half.

| Method | AUROC | AUC-PR | F1 |
|---|---|---|---|
| Llama's raw True/False confidence | 0.745 | 0.329 | 0.371 |
| BTProp tree (the paper's method, our reproduction) | 0.741 | 0.403 | 0.439 |
| **Answer graph + loopy BP (ours)** | **0.773** | **0.475** | 0.438 |
| Random guessing | 0.500 | 0.161 | — |

- The graph is best on AUROC and AUC-PR (F1 ties the tree). Only "graph vs raw, AUC-PR" clears
  the 95% bar in a cluster bootstrap — and only just; the rest point the same way but are not
  proven with 49 false sentences.
- The surrogate replaces **63% of the LLM confidence calls** (all child nodes) with no loss in
  AUROC (0.741 → 0.757).
- Loopy BP converged on 122 of 122 answer graphs.

Full numbers, confidence intervals and what they do and don't show:
[docs/project-log/05-results.md](docs/project-log/05-results.md).

## How it works

```
sentence ──► LLM builds a belief tree ──► NLI labels each link ──► LLM confidence per node
             (decompose / premises /      (supports, implies,      (P(True) from next-token
              corrections — paper's        contradicts; DeBERTa)     probabilities)
              own prompts)
                         │
                         ▼
   Hidden Markov Tree: hidden truth per node, confidence = noisy evidence (fitted calibration),
   links = logical dependencies ──► exact belief propagation ──► P(sentence is true)

ours: trees of all sentences in an answer + links between related sentences
      ──► factor graph with loops ──► loopy belief propagation ──► P(true) for every sentence
```

Plain-language walkthrough: [docs/project-log/01-big-picture.md](docs/project-log/01-big-picture.md).

## Repository layout

| Path | What's there |
|---|---|
| `src/inference/hmt.py` | Exact belief propagation on a Hidden Markov Tree; emission models (bucket table, logistic curve) |
| `src/inference/loopy.py` | Loopy belief propagation on a factor graph (extension 1) |
| `src/llm/` | LLM clients (Hugging Face models, Groq, OpenAI, a fake for tests), disk cache, prompts |
| `src/tree/` | Belief-tree construction with the paper's strategies; NLI relationship labelling |
| `src/surrogate/` | Hashed-words + ridge-regression confidence scorer (extension 2) |
| `src/eval/metrics.py` | AUROC, AUC-PR, cross-fitted F1 |
| `src/pipeline.py` | `verify_statement` (paper's method) and `verify_answer` (our graph) — the whole method in one call |
| `scripts/` | Experiments and the demo (below) |
| `notebooks/kaggle_felm_llama3.ipynb` | Runs the full experiment on Kaggle's free GPUs |
| `tests/` | 75 tests, including hand-computed and brute-force checks |
| `docs/project-log/` | **Start here to understand the project**: decisions (D1–D36), results, glossary, Kaggle guide |
| `docs/phases/` | The original phase-by-phase plan |

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install dependencies into .venv
uv run pytest                # 75 tests, no GPU or API key needed
```

Optional, for the Groq demo backend: copy `.env.example` to `.env`, fill in `GROQ_API_KEY`, and
load it with `export $(grep '^GROQ_API_KEY=' .env)`. Never commit `.env`.

**GPU on a laptop (optional).** `uv sync` installs a CPU build of PyTorch. To use an NVIDIA GPU
for the NLI model, install the CUDA build into the environment and run scripts with the
environment's Python directly (`uv run` may switch back to the CPU build):

```bash
uv pip install --python .venv --no-deps "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
PYTHONPATH=. .venv/Scripts/python.exe scripts/<script>.py ...     # Windows; .venv/bin/python elsewhere
```

## Reproducing the results

**1. Llama-3-8B runs on Kaggle** (free 2× T4 GPU, about 5–7 hours) — step-by-step in
[docs/project-log/07-running-on-kaggle.md](docs/project-log/07-running-on-kaggle.md). The
notebook runs, in order:

```bash
python scripts/decontextualize_felm.py --backend hf     # make each sentence stand alone
python scripts/fit_emission_model.py   --backend hf --standalone   # calibration, fit half
python scripts/evaluate_felm.py        --backend hf --standalone   # paper's tree, test half
python scripts/evaluate_graph.py       --backend hf --standalone   # extension 1
python scripts/evaluate_surrogate.py   --backend hf --standalone   # extension 2
```

Download `reason_graph_results.zip` from the notebook's Output tab and unzip it into the repo
root (it fills `results/` and `llm_cache/`).

**2. Everything else replays the saved Llama answers** — add `--offline`, no Llama needed (the
NLI model runs; a GPU makes it minutes instead of hours):

```bash
python scripts/choose_link_rule.py --backend hf --standalone --offline     # D35, fit half only
python scripts/evaluate_graph.py   --backend hf --standalone --offline --link-rule no_contradiction
python scripts/report_metrics.py                                         # AUROC / AUC-PR / F1 table
```

## Demo

```bash
python scripts/verify.py --backend groq "The Eiffel Tower, completed in 1889, is located in Berlin."
```

```
[root, -, confidence 0.00] The Eiffel Tower, completed in 1889, is located in Berlin.
    [decomposition, contradiction, confidence 1.00] The Eiffel Tower was completed in 1889.
    [decomposition, reverse_entailment, confidence 0.00] The Eiffel Tower is located in Berlin.
P(true) = 0.224 -> LIKELY HALLUCINATION
```

Add `--answer` and several sentences to score them together with the answer graph. The Groq
backend types out its confidence instead of giving token probabilities — fine for a demo; the
reported numbers come from Llama-3-8B.

From Python:

```python
from src.pipeline import verify_statement
result = verify_statement(llm, labeler, calibration, "Some claim.")
result.posterior_true, result.is_trustworthy, result.raw_tree
```

## Honest limitations

- One dataset (FELM-Science) with 49 false test sentences: differences under ~0.05 AUROC can't be
  told apart from chance.
- Trees are one level deep (the paper's are two), to fit free GPU time — the likely reason the
  reproduced tree only matches raw confidence instead of beating it.
- The paper used a hand-cleaned FELM file that is not public; we approximate it by rewriting
  sentences to stand alone, with an NLI safety check.
- The chosen link rule drops contradictions between sentences, so a real one is not used.
- The surrogate saves only confidence calls; generating premises and corrections still needs the LLM.

## Development

```bash
uv run ruff check . && uv run ruff format --check .   # style
uv run mypy                                            # strict type checks
uv run pytest                                          # tests
```

Pre-commit hooks run the same checks. Branches are `phase-N/<topic>`, merged by pull request.

## Credits

- Base paper and method: Hou et al., NAACL 2025. The LLM prompts in `src/llm/prompts.py` are
  copied from the official BTProp code, with attribution in that file (decision D29).
- Data: FELM (`hkust-nlp/felm` on Hugging Face). NLI model: `cross-encoder/nli-deberta-v3-large`.
  LLM: Llama-3-8B-Instruct.

License: Apache 2.0 (see `LICENSE`).
