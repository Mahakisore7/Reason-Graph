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

## Status at a glance

| Phase | What | State |
|---|---|---|
| 0 | Belief-propagation math engine (Hidden Markov Tree) | Done, merged (PR #1) |
| 1 | Connection to LLMs (fake, cache, OpenAI, Groq) | Done, merged (PR #3) |
| 2 | Belief-tree construction + first real evaluation on FELM-Science | Code done; faithful reproduction number still pending (needs OpenAI credit) |
| 3+ | Our novelty: DAG + loopy belief propagation, surrogate scorer | Not started |

## Open items

- **Faithful reproduction number.** Current numbers use Groq, which can only give a typed-out
  confidence, not the paper's method. Re-run with OpenAI once billing credit is added.
- **Transition parameters are hand-set guesses** (see [03-decisions.md](03-decisions.md)), not
  learned from data.
- **Tree depth is 1** to keep API calls affordable; the paper's trees go deeper.
- **API keys** pasted into a chat session earlier should be rotated (platform.openai.com,
  console.groq.com).

Keep this folder updated as we go: new decision → add to `03-decisions.md`; new number →
add to `05-results.md`.
