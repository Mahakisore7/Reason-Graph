# Roadmap — One Phase at a Time

Each phase has its own detailed teaching doc in `docs/phases/` — overview, concepts you need
first, architecture diagrams, the interface contract to implement against (given as code blocks
within each doc — class/function signatures and docstrings, not working code), hand-computed
checkpoints to verify your own work, production practices, git workflow, suggested file structure,
and a Definition of Done. **This is a guide, not a solution.** Work through the phases in order;
don't start a phase until the previous one's DoD is genuinely met, not just "the code runs."

Align the week numbers to your actual Mid-Sem / Final review dates once announced — the ordering
matters more than the exact weeks.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done and verified by you

## [Phase 0 — Environment Setup & Your First Algorithm](phases/phase-0-setup-and-first-algorithm.md)

- [ ] `uv sync` confirmed working after setting up the project scaffold yourself (Part A of the
      phase doc).
- [ ] `src.inference.hmt` implemented: exact upward-downward belief propagation on a tree.
- [ ] Your own `tests/test_hmt.py` passes all three hand-computed checkpoints in the phase doc.

**DoD:** you can explain, from memory, why a tree structure specifically makes this inference
exact rather than approximate.

## [Phase 1 — LLM Access Layer](phases/phase-1-environment-llm-access.md)

- [ ] `FakeLLMClient`, `CachingLLMClient` implemented and tested.
- [ ] Provider chosen deliberately — read `docs/05-llm-provider-choice.md` first; it covers a real
      technical trap (Grok's current models silently ignore `logprobs`) worth knowing before you
      build around the wrong provider.
- [ ] `OpenAIClient` (or your chosen provider's client) implemented and its logprobs behaviour
      personally confirmed with a real key.

**DoD:** not started.

## [Phase 2 — Reproduce the Baseline](phases/phase-2-tree-construction.md)

- [ ] Three child-generation strategies implemented and tested against `FakeLLMClient`.
- [ ] `build_belief_tree` orchestration implemented and tested.
- [ ] Real NLI labeller wired in; emission model fit against real labelled data.
- [ ] Reproduced AUROC on FELM-Science within a few points of the paper's own reported number.

**DoD:** not started. **This is the sanity-check gate for the whole project** — don't trust any
Phase 3/4 "improvement" claim until this passes for real.

## [Phase 3 — Generalise to a DAG + Loopy BP](phases/phase-3-loopy-bp.md)

- [ ] `run_loopy_bp` implemented.
- [ ] Cross-validated exact against Phase 0's tree algorithm on a tree-shaped graph.
- [ ] Checked against a brute-force oracle on a genuinely looped graph, with a documented
      tolerance.
- [ ] Convergence measured on real claim DAGs (once Phase 2 is producing real data to extend).

**DoD:** not started.

## [Phase 4 — Cost Reduction via Surrogate Scorer](phases/phase-4-surrogate-scorer.md)

- [ ] `SurrogateConfidenceModel` + `HashingEmbedder` implemented, proven to learn real signal on a
      synthetic pattern.
- [ ] Trained on real logged data; accuracy-vs-delegation-fraction curve produced.

**DoD:** not started.

## [Phase 5 — Full Assembly](phases/phase-5-full-assembly.md)

- [ ] `DispatchingTransition` + `build_hidden_markov_tree` + `verify_statement` implemented.
- [ ] Both cross-validation tests (decomposition path, premise+correction path) pass.
- [ ] Run against a real LLM + real NLI labeller + real fitted emission model.
- [ ] *(Stretch, Extension 3)* Applied to real multi-step reasoning traces via `FactorGraph`/loopy
      BP.

**DoD:** not started. Read this phase's doc even before you reach it — it describes a specific
design trap (mixing transition rules across one tree) worth knowing about ahead of time.

## [Phase 6 — Mid-Semester Review Checkpoint](phases/phase-6-midsem-review.md)

**DoD:** a short deck/demo a mentor unfamiliar with the daily details can follow end-to-end, with
every presented number reproducible on demand.

## [Phase 7 — Full Evaluation](phases/phase-7-full-evaluation.md)

**DoD:** tree-vs-DAG comparison, full ablations, convergence statistics, and qualitative
failure-case analysis, complete enough to write the final report directly from.

## [Phase 8 — Final Review & Write-Up](phases/phase-8-final-review.md)

**DoD:** final deliverable submitted; `uv sync && uv run pytest` passes on a fresh clone, verified
by someone who didn't write the code; `docs/` and code are in a state a stranger could pick up.

---

**How to use these docs:** each phase doc contains the full interface contract you're building —
class/function signatures and docstrings, given as code blocks within the doc — plus hints for how
to think through the actual logic, which is yours to write. Set up your own repo starting with
Phase 0's Part A, create each file as its phase doc describes, and write your own tests against
every phase's hand-computed checkpoints — that's the single best way to know your implementation
is actually correct rather than just plausible-looking.
