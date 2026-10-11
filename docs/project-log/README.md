# Project Log

`docs/phases/` is the **plan** — what we intended to build. This folder is the **record** —
what we actually built, why we made each choice, and what came out. Written in plain language,
so it can be read before a mentor review or used to explain the project to someone new.

## Read in this order

| File | What it answers |
|---|---|
| [01-big-picture.md](01-big-picture.md) | What problem are we solving, and how does the method work? (with a worked example) |
| [02-what-we-built.md](02-what-we-built.md) | What exists in the code now, phase by phase, and how each piece was checked |
| [03-decisions.md](03-decisions.md) | Every real choice we made, the alternatives, and why we picked what we picked |
| [04-dataset-felm-science.md](04-dataset-felm-science.md) | Where the data comes from, what's inside it, how we cleaned and split it |
| [05-results.md](05-results.md) | Numbers so far, what they do and don't show, known weaknesses |
| [06-glossary.md](06-glossary.md) | Short definitions of every technical term used here |
| [07-running-on-kaggle.md](07-running-on-kaggle.md) | Step-by-step: running the Llama-3-8B reproduction on Kaggle's free GPUs |

## Status at a glance

| Phase | What | State |
|---|---|---|
| 0 | Belief-propagation math engine (Hidden Markov Tree) | Done, merged (PR #1) |
| 1 | Connection to LLMs (fake, cache, OpenAI, Groq) | Done, merged (PR #3) |
| 2 | Belief-tree construction + first real evaluation on FELM-Science | First Llama run: tree lowered AUROC (0.670 → 0.588) — our trees weren't the paper's. Rebuilt with the paper's own method (D29) + stand-alone sentences (D30); Kaggle run pending |
| 3 | Our novelty 1: one graph per answer + loopy belief propagation | Done. Graph ≈ tree (no clear gain); loopy BP converged on 122/122 runs — results 3d |
| 4 | Our novelty 2: cheap surrogate confidence scorer | Done. Children-only delegation saves 63% of confidence calls with no AUROC loss — results 3e |

## Open items

- **Faithful reproduction run.** Run the updated notebook on Kaggle (paper's method +
  stand-alone sentences) — steps in [07-running-on-kaggle.md](07-running-on-kaggle.md).
- **Groq held-out run** (old prompts) is superseded and won't be continued (D31).
- **Transition parameters are hand-set guesses** (see [03-decisions.md](03-decisions.md)), not
  learned from data.
- **Tree depth is 1** to keep API calls affordable; the paper's trees go deeper.
- **API keys** pasted into a chat session earlier should be rotated (platform.openai.com,
  console.groq.com).

Keep this folder updated as we go: new decision → add to `03-decisions.md`; new number →
add to `05-results.md`.
