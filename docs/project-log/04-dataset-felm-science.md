# Dataset: FELM-Science

## What it is

**FELM** ("Factuality Evaluation of Large Language Models") is a public benchmark built by the
HKUST NLP group. It tests how well a method can catch factual errors in AI-written answers.

- Questions were collected from places like Quora, Twitter, exam-style subject sets, and
  hand-written prompts.
- **ChatGPT** wrote the answers.
- **Humans** split each answer into segments (usually a sentence) and marked each segment
  true or false. For false ones they also recorded the error type and a reference link.

The BTProp paper uses FELM-Science as one of its three benchmarks. The other two are
WikiBio-GPT3 and FactCheckGPT.

## Where we get it

HuggingFace: `hkust-nlp/felm`. We download the raw file `all.jsonl` directly, because the
dataset's own loader script no longer works with the current `datasets` library.

## What's inside

All of FELM: 847 answers across 5 domains — reasoning (208), math (194), world knowledge (184),
writing/recommendation (136), **science (125)**. We use only science.

One science record, shortened:

| Field | Example |
|---|---|
| `prompt` | "What happens to you if you eat watermelon seeds?" |
| `response` | ChatGPT's full answer |
| `segmented_response` | The answer split into 4 segments |
| `labels` | `[True, False, False, False]` — one per segment |
| `type` | `[None, "knowledge_error", "irrelevant_with_qst", "irrelevant_with_qst"]` |
| `comment` | Annotator's note, e.g. "a watermelon seed would make it all the way through your digestive system…" |
| `ref` | Source link the annotator used |

**Science in numbers (raw):**
- 125 answers → 684 segments
- 582 true, **102 false** (about 15%)
- 39 answers contain at least one false segment
- Why the false segments are false: 92 knowledge errors, 8 "irrelevant to the question",
  2 reasoning errors

> Worth knowing: 8 of the "false" segments are false because they're **off-topic**, not
> because they're factually wrong. A fact-checker can rate those as true and still be "wrong"
> on this benchmark.

## Problems we found by reading the data

| Problem | Example | What we did |
|---|---|---|
| Non-claims split out as segments | `"2."`, `"Sure!"`, `"Regenerate response"`, `"Mud."` — 64 segments under 4 words, all labelled true | Removed (segments under 4 words) |
| Same answer stored twice | Nuclear-fusion answer and "where does the sun get oxygen" answer, each twice with identical labels | Kept one copy of each |
| Some segments are several sentences glued together | A segment containing "…\n\nHowever, eating a large amount…" | Left as is — that's how the benchmark defines a unit |

After cleaning: **123 answers, 610 segments, 102 false** (every false segment kept).

## How we split it

| Half | Answers | Segments | False | Used for |
|---|---|---|---|---|
| Fit | ~half | 300 | 53 | Building the calibration table only |
| Test | ~half | 310 | 49 | Measuring accuracy only |

- **Split by whole answer.** Two segments from the same answer never land in different halves.
- **Stratified.** Answers with and without errors are split separately, so both halves get a
  fair share of false examples.
- **Fixed seed (42).** The split is the same every run.

Why this matters: see decision D18 in [03-decisions.md](03-decisions.md).
