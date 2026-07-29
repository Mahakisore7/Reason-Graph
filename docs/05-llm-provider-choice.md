# Which LLM Provider Should You Use?

You asked about Grok specifically, so this got checked properly rather than answered from a
general impression — the answer turns out to hinge on one very specific, easy-to-miss technical
requirement this project has.

## The Requirement That Decides This

BTProp's confidence-scoring step (`docs/04-research-notes-btprop.md`, Step 3) is not "ask the
model if a statement is true and read the words back." It prompts `"True or False? {statement}"`
and reads the model's own **token-level log-probabilities** for the words "True" and "False,"
then normalises them into a confidence score. This needs an API that exposes `logprobs` /
`top_logprobs` on chat completions — a feature most providers treat as optional, and not all of
them keep supported as their models evolve.

## What Was Actually Checked (as of this project's start)

| Provider | logprobs support | Verdict for this project |
|---|---|---|
| **OpenAI** (`gpt-3.5-turbo`, `gpt-4o-mini`, etc.) | Stable, documented, standard `logprobs`/`top_logprobs` parameters on chat completions | **Safe default.** Also one of the two backbones BTProp's own paper evaluates, so results are directly comparable to the numbers in `docs/04-research-notes-btprop.md`. |
| **xAI Grok** | Confirmed **not supported on current models** — Grok 4.20 and newer **silently ignore** `logprobs`/`top_logprobs` rather than erroring. Some older models (`grok-2-1212`) show `top_logprobs` in SDK signatures, but that's a deprecated path for a project starting now. | **Don't use for the confidence-scoring step.** The silent-ignore behaviour is the dangerous part — code would run, return *something*, and you might not notice it isn't real log-probabilities until your numbers look strange. Fine for the free-text generation strategies (Decomposition/Premises/Correction, which don't need logprobs at all) if you want to mix providers, just not for `true_false_confidence()`. |
| **Google Gemini** | Inconsistent across models and API surfaces — works on some endpoints, explicitly rejected on others (`gemini-2.5-flash`, `gemini-3-pro-preview` have both returned "logprobs not supported/enabled" errors), and not available at all through the OpenAI-compatibility layer. | Workable if you pin to a specific model version known to support it at the time you check, but higher maintenance risk than OpenAI for this specific need. |
| **A local open-weight model** (Llama-3-8B-Instruct via Hugging Face `transformers` or vLLM) | You have the raw logits directly — no API limitation possible, because there's no API. | **Best choice if you have GPU access** (a free-tier Colab GPU is enough for 8B at reasonable speed). Also the *other* backbone BTProp's own paper evaluates, free to run as much as you want, and the most pedagogically honest option — you can actually inspect the full probability distribution over every token, not just whatever a hosted API chooses to expose. |

## Recommendation

**Start with OpenAI for the API-backed path, and add the local Llama-3-8B-Instruct path once
your pipeline works end to end.** That combination exactly matches the base paper's own two
backbones, which means your Phase 2 reproduction numbers are comparable to a real, published
result — a much stronger position than an unpublished third combination, however capable that
model might be for general chat use. Keep Grok (or any other model) in your back pocket for the
free-text-only strategies if you want variety or lower cost there specifically — just don't route
`true_false_confidence()` through anything you haven't personally confirmed returns real logprobs.

**How to confirm this yourself, for any provider, before trusting it:** send one request with
`logprobs=True, top_logprobs=5` and print the raw response. If the `logprobs` field in the
response is genuinely populated with per-token alternatives and probabilities, you're fine. If
it's `null`, empty, or the request errors — even if the SDK "accepted" the call — don't build on
it. Providers change this kind of support over time in both directions, so this is worth
re-checking once, right when you start Phase 1, rather than trusting this table forever.

## Other Suggestions Worth Knowing About

Since you asked what else you might not know to ask about:

- **Track your API spend from day one**, not just at the end — most providers have a usage
  dashboard; check it after your first real end-to-end run on a handful of statements so you know
  the actual cost-per-verification before running it across a whole dataset.
- **`tiktoken` (OpenAI) or the model's own tokenizer** is worth knowing how to use directly — being
  able to check "how many tokens is this prompt" before sending it avoids surprise truncation on
  longer belief trees.
- **Rate limits differ a lot by account tier** — a brand-new API key usually starts on a low tier;
  check your actual rate limit before assuming `tenacity`'s retry/backoff (already a listed
  dependency,
  `docs/02-tech-stack.md`) will be enough on its own for a large evaluation run.
- **Keep a "provider swap" mental model from day one**, since `LLMClient` is a protocol precisely
  so this is cheap later (`docs/01-architecture.md`) — don't hard-code assumptions about one
  provider's exact response shape anywhere outside the one class that talks to it.
