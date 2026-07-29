# Project Charter — BP-Verify

**Course:** 22AIE301 Probabilistic Reasoning
**One-line pitch:** Verify multi-step LLM reasoning chains by modelling claim truthfulness as a
hidden Markov tree / factor graph and running belief propagation over it, extending BTProp
(NAACL 2025) from trees to general reasoning graphs.

## Problem Statement

LLMs answer hard questions by reasoning through several steps. A single wrong intermediate step
can silently invalidate an otherwise fluent final answer, and most hallucination-detection methods
score only the final answer, ignoring the logical structure connecting the steps that produced it.

## Goals

- **G1 — Understand deeply.** Reproduce a real, peer-reviewed method (BTProp) closely enough to
  explain every design decision and every equation to a non-trivial audience (course mentor,
  reviewers).
- **G2 — Extend honestly.** Generalise BTProp's tree-structured inference to a general factor
  graph (DAG) using loopy belief propagation, and reduce its per-verification LLM query cost with
  a surrogate confidence model — both directions the base paper's own authors left open.
- **G3 — Ship something evaluable.** Produce a working, tested pipeline with quantitative results
  (AUROC / AUC-PR / F1 against ground truth) comparable to the base paper's own reporting, not just
  a demo.

## Non-Goals (explicitly out of scope for this semester)

- Training or fine-tuning a new LLM. We only *use* existing LLMs (API or open-weight) as a
  component.
- A production deployment / hosted service. This is a research-engineering project, not a
  product — no auth, multi-tenant, or SLA concerns.
- A polished frontend. A CLI and notebooks are enough; a UI is a stretch goal at most, and only if
  Phases 1–4 finish early.

## Success Criteria

| # | Criterion | Definition of "done" |
|---|---|---|
| 1 | Baseline reproduced | Our reimplementation's AUROC on FELM-Science is within a few points of the paper's reported number, on the same backbone. |
| 2 | DAG extension works | Loopy BP runs on real claim graphs; convergence behaviour is measured and reported honestly (including when it *doesn't* converge). |
| 3 | Cost extension works | Surrogate scorer trained and validated; an accuracy-vs-delegation-fraction curve is produced. |
| 4 | Explainable to a mentor | Every module has a doc explaining *why*, not just *what* — this repo's `docs/` folder is the primary artifact for that. |

## Stakeholders

- **Team:** implementation, experiments, write-up.
- **Course mentor / faculty:** reviews progress at Mid-Sem and Final review; needs to be able to
  follow the reasoning, not just see results.
- **(Optional, stretch) External reviewers:** if results are strong enough to shape into a
  workshop submission.

## Key Risks

| Risk | Mitigation |
|---|---|
| Loopy BP doesn't converge reliably on real claim graphs | Treated as a real empirical question, not assumed away; report non-convergence rate honestly (Section on limitations in the final write-up). |
| LLM API costs/rate limits slow down iteration | Cache every LLM call by prompt hash from day one (see `docs/02-tech-stack.md`); use a small open-weight model locally for rapid iteration, save API-model runs for final numbers. |
| Surrogate scorer needs more labelled data than expected | Scope it as a tunable dial (delegate 0–100%), not an all-or-nothing switch — partial success is still a reportable result. |
| Scope creep (Extension 3 — real reasoning-chain dataset) | Explicitly marked a stretch goal in the roadmap; Phases 1–4 alone are a complete, defensible project. |

## Reference Documents

- `docs/01-architecture.md` — system design and data flow
- `docs/02-tech-stack.md` — tooling decisions and why
- `docs/03-roadmap.md` — phase-by-phase plan with definitions of done
- `docs/04-research-notes-btprop.md` — condensed, accurate notes on the base paper
