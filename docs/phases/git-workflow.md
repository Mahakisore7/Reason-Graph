# Git Workflow

Cross-cutting practice, referenced by every phase doc rather than repeated in each one.

## Branching Model

Trunk-based, kept deliberately simple for a small team and a fixed-length project — no
`develop` branch, no long-lived release branches.

- `main` — always green (CI passing). Nobody commits to `main` directly.
- `phase-N/<short-topic>` — one branch per unit of work within a phase, e.g.
  `phase-1/llm-client-protocol`, `phase-3/loopy-bp-core`. Branch names carry the phase number so
  `git log --all --oneline` doubles as a rough project timeline.
- Merge to `main` via pull request only, even solo — PRs are where the "explain this to your
  mentor" habit gets built, because you have to write down *why*, not just *what*.

## Commit Messages — Conventional Commits

```
<type>(<scope>): <short summary>

[optional body — the *why*, not a restatement of the diff]
```

| Type | Use for |
|---|---|
| `feat` | New capability (`feat(inference): add loopy BP core algorithm`) |
| `fix` | Bug fix |
| `test` | Adding/fixing tests only |
| `docs` | Documentation only |
| `refactor` | No behaviour change |
| `chore` | Tooling, CI, dependency bumps |

Why bother for a course project: `git log --oneline --grep=feat` becomes an instant "what did we
actually build" list when writing the final report, without re-reading every diff.

## Pull Request Checklist

Before requesting review (even self-review, solo):

- [ ] `uv run pytest` passes locally
- [ ] `uv run ruff check .` and `uv run mypy` are clean
- [ ] New code has tests — for `src.inference`, tests must include at least one hand-computed
      expected value, not just "it runs without crashing" (see `tests/test_hmt.py` for the pattern)
- [ ] The PR description states which phase/task this closes (link to `docs/phases/phase-N-*.md`)

## Tagging

Tag the commit that closes each phase's Definition of Done: `git tag phase-1-complete`, etc. This
gives you `git diff phase-1-complete phase-2-complete` for free when writing up "what changed this
phase" for a review — much less error-prone than reconstructing it from memory.

## What Never Gets Committed

Enforced by `.gitignore` (already configured): `.venv/`, `data/` (raw datasets), `llm_cache/`
(cached API responses — can be large and may contain data you don't have redistribution rights
to), `.env` / any file holding an API key, `results/*.csv` and `*.json` (regenerable — commit the
scripts that produce them, not the numbers themselves, except for a final frozen results file
explicitly named as such at submission time).
