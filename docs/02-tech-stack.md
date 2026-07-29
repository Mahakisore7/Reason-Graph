# Tech Stack & System Design Practices

Every choice below has a reason. If a reason stops applying, change the choice — don't keep a
tool out of habit.

## Environment & Dependency Management

| Concern | Choice | Why |
|---|---|---|
| Python version | 3.11+ | Modern typing features (`X \| Y` unions), performance improvements. |
| Package/env manager | **uv** | Single tool for venv + install + lockfile; far faster than pip/poetry; `uv.lock` gives fully reproducible installs for every teammate and CI. |
| Dependency spec | `pyproject.toml` (PEP 621) | Standard, tool-agnostic; not locked into uv-specific config for the dependency list itself. |
| Lockfile | `uv.lock`, committed to git | "Works on my machine" is not an acceptable failure mode for a project with a mentor review. |

**Commands the whole team should use identically:**
```bash
uv sync              # install exact locked deps into .venv
uv run pytest        # run tests inside the managed env
uv add <package>      # add a new dependency (updates pyproject.toml + uv.lock together)
uv run ruff check .   # lint
```

## Code Quality

| Concern | Choice | Why |
|---|---|---|
| Linting + formatting | **ruff** | One fast tool replaces flake8 + isort + black; near-instant on a repo this size. |
| Type checking | **mypy** (or pyright) in CI | Catches an entire class of "wrong shape of data passed between modules" bugs before they become a debugging session. |
| Pre-commit hooks | **pre-commit** running ruff + mypy on every commit | Nobody has to remember to lint; broken code never reaches `main`. |
| Docstrings | Google-style, on every public function | This project's whole point is explainability — code that can't explain itself defeats the purpose. |

## Testing Strategy

- **Unit tests, no network:** `bpverify.inference` (the actual belief-propagation math) is tested
  against small, hand-computed examples with a known correct posterior — see `tests/test_hmt.py`.
  This layer must NEVER require an LLM API key to test, by design (see the module-boundary rule in
  `01-architecture.md`).
- **Integration tests, mocked LLM:** `bpverify.tree` and `bpverify.llm` are tested against a fake
  `LLMClient` that returns canned responses, so CI never spends API budget or depends on external
  uptime.
- **Manual/notebook validation, real LLM:** actual end-to-end runs against a real API happen in
  notebooks or scripts, logged and cached (below), not in the automated test suite.

## LLM Access

**Before picking a provider, read `docs/05-llm-provider-choice.md`** — it covers a concrete,
easy-to-miss technical requirement (this project's confidence-scoring method needs real
`logprobs` support) and what was found checking several providers against it.

| Concern | Choice | Why |
|---|---|---|
| Interface | An `LLMClient` **protocol** (structural typing) in `bpverify/llm/base.py`, with `OpenAIClient` and `LocalHFClient` implementations | Swap backbones (matching the base paper's own GPT-3.5 / Llama-3-8B comparison) without touching any calling code. |
| Local/open backbone | Llama-3-8B-Instruct via Hugging Face `transformers` or vLLM | Free, fast iteration; matches one of the base paper's two tested backbones directly. |
| API backbone | Any GPT-family model via its official SDK | Matches the base paper's other tested backbone; used sparingly for final numbers given cost. |
| Caching | Every call cached to disk keyed by a hash of (prompt, model, params) | Re-running an experiment (or fixing a bug downstream) must never mean re-paying for or re-waiting on identical LLM calls. |
| Retries | Exponential backoff on rate-limit/transient errors (e.g. `tenacity`) | API flakiness should not look like a research finding. |

## Graph, ML & Evaluation Libraries

| Purpose | Library | Notes |
|---|---|---|
| Graph representation & traversal | `networkx` | Both the tree (Phase 1) and DAG (Phase 3) cases; cycle detection built in, useful for validating loopy-BP inputs. |
| NLI (entailment/contradiction) | a pretrained NLI checkpoint via `transformers` | Matches what the base paper itself uses to label parent-child relationships. |
| Surrogate scorer | `sentence-transformers` (embeddings) + `scikit-learn` (classifier head) or a small `torch` MLP | Kept intentionally lightweight — this component's job is to be *cheap*, not state-of-the-art. |
| Metrics | `scikit-learn.metrics` (`roc_auc_score`, `average_precision_score`, `f1_score`) | Identical metrics to the base paper, for directly comparable numbers. |
| Data wrangling | `pandas` | Ablation tables, result logs. |
| Plotting | `matplotlib` | Convergence curves, accuracy-vs-cost curves — matches the base paper's own analysis style. |

## Configuration & Reproducibility

- All experiment parameters (max tree depth, loopy-BP iteration cap, convergence threshold,
  surrogate delegation fraction, model names) live in a single `configs/*.yaml` file per
  experiment, loaded via `pydantic` — never hard-coded magic numbers inside a function.
- Every experiment run is seeded and logs its exact config alongside its results (a plain JSON
  sidecar file next to the results CSV is enough for a project this size — no need for a heavy
  experiment-tracking service).

## CI

A minimal GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push: `uv sync`,
`ruff check`, `mypy`, `pytest`. Nothing merges to `main` with a red check.

## If a Demo UI Is Wanted Later (Stretch, Not Now)

Not needed for the core deliverable, but if a visual demo of a belief tree/graph and its computed
beliefs would help the final presentation: a small **Vite + React** app, managed with **pnpm**, is
the equivalent modern-standard choice on the JS side — mirroring the uv choice on the Python side
(fast, lockfile-based, no legacy baggage). Not scaffolded now; add only if Phases 1–4 finish with
time to spare.
