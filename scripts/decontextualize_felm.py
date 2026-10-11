"""Rewrites every FELM-Science segment so it stands alone ("They are located on the wings" ->
"Ground spoilers are located on the wings"), using the question and the full answer as context.

The paper evaluated on a hand-cleaned FELM-Science file (not public) whose sentences already
stand alone, and its confidence prompt says to treat statements that need outside context as
false. We approximate the hand-cleaning with the LLM (D30). Labels are never touched: each
rewrite keeps the human label of the segment it came from.

Safety check: a rewrite is kept only if the NLI model says it entails the original -- so it
still says everything the original said. A rewrite that drops a clause (perhaps the very part
that made the segment false) or swaps in a different sentence is rejected, and the original is
kept instead.

Writes results/felm_science_standalone_<model>.json; pass --standalone to the fit and
evaluation scripts to use it.

Usage: python scripts/decontextualize_felm.py --backend hf [--model NAME]
"""

from __future__ import annotations

import argparse
import json

from scripts.backends import add_backend_args, make_backend
from scripts.fit_emission_model import load_felm_science_answers, standalone_path
from src.llm.prompts import decontextualize_prompt
from src.tree.nli_labeler import CrossEncoderDirectionalNLI, DirectionalNLI

_LABEL = "standalone sentence:"


def parse_standalone(output: str, original: str) -> str:
    """The text after "Standalone sentence:"; the original segment if the model didn't follow
    the format, so a bad answer can never replace a segment with something unrelated."""
    for line in output.replace("*", "").split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith(_LABEL):
            rewritten = stripped[len(_LABEL) :].strip()
            return rewritten or original
    return original


def keep_if_faithful(rewritten: str, original: str, nli: DirectionalNLI) -> str:
    if rewritten == original or nli.label_direction(rewritten, original) == "entailment":
        return rewritten
    return original


def main() -> None:
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N answers (smoke test)"
    )
    args = parser.parse_args()
    backend = make_backend(args)
    llm = backend.cached("standalone")
    print(f"LLM: {backend.model}")

    answers = load_felm_science_answers()[: args.limit]
    total = sum(len(a.segments) for a in answers)
    nli = CrossEncoderDirectionalNLI()
    rewrites: dict[str, str] = {}
    rejected: dict[str, str] = {}
    done = 0
    for answer in answers:
        for segment, _ in answer.segments:
            output = llm.complete(
                decontextualize_prompt(answer.question, answer.response, segment), max_tokens=200
            )
            proposed = parse_standalone(output, segment)
            rewrites[segment] = keep_if_faithful(proposed, segment, nli)
            if rewrites[segment] != proposed:
                rejected[segment] = proposed
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  rewrote {done}/{total}")

    changed = {s: r for s, r in rewrites.items() if r != s}
    print(f"{len(changed)} of {len(rewrites)} segments rewritten; {len(rejected)} rewrites")
    print("rejected by the NLI check (original kept). Examples of kept rewrites:")
    for original, rewritten in list(changed.items())[:15]:
        print(f"  - {original}\n    -> {rewritten}")

    out_path = standalone_path(backend.run_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "llm": backend.model,
                "num_segments": len(rewrites),
                "num_changed": len(changed),
                "num_rejected": len(rejected),
                "rewrites": rewrites,
                # proposed rewrites that failed the entailment check, for inspection
                "rejected": rejected,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
