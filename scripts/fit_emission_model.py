"""Fits a real emission model from labelled data: for statements the LLM scored with a given
confidence, how often are they actually true vs. false? Mirrors
docs/04-research-notes-btprop.md Step 4, and the same idea as the official repo's
configs/hmm.yaml -- but fit fresh here, on our own dataset and LLM, not copied from theirs.

This is a *calibration* question about the LLM's confidence scores on their own (does 0.9
confidence mean ~90% likely true?) -- it has nothing to do with belief-tree structure or the
transition parameters in src/tree/relationships.py, which are a separate concept.

Dataset: FELM-Science (hkust-nlp/felm, "science" domain) -- 125 ChatGPT answers, split into 684
segments, each hand-labelled true/false. Only the "fit" half of the answers is used here (see
split_by_response); scripts/evaluate_felm.py tests on the other half. Note: the HuggingFace `datasets` library can't load this repo anymore
(it ships a now-unsupported loading script), so we pull the raw all.jsonl file directly instead.

Usage: uv run python scripts/fit_emission_model.py [--backend groq|hf] [--model NAME]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression

from scripts.backends import add_backend_args, make_backend
from src.inference.hmt import LogisticEmission

# Same 5 bins (4 edges) as the official repo's configs/hmm.yaml, so our numbers stay
# comparable to theirs even though we fit them ourselves.
BIN_EDGES = [0.19, 0.39, 0.69, 0.89]


def bin_index(score: float) -> int:
    for i, edge in enumerate(BIN_EDGES):
        if score <= edge:
            return i
    return len(BIN_EDGES)


Response = list[tuple[str, bool]]  # one AI answer = its labelled segments, in order


@dataclass(frozen=True)
class FelmAnswer:
    """One ChatGPT answer: the question, the full answer text, and its labelled segments.
    The question and full text are the context used to make each segment stand alone."""

    question: str
    response: str
    segments: Response


FIT_FRACTION = 0.5
SPLIT_SEED = 42
# FELM's segmenter sometimes emits non-claims as their own segment -- bare list numbers
# ("2."), filler ("Sure!", "Regenerate response"), single list items ("Mud."). On inspection
# all 64 segments under 4 words are like this, all labelled true; real claims start at ~5
# words ("This is known as unihemispheric sleep.").
MIN_WORDS = 4
# List openers: "Sure, here are some papers on summarization:\n\n1." -- an intro plus the first
# list number, not a claim. All 14 in the data are labelled true; the LLM rightly rates them as
# not-true, which taught the calibration table that low confidence means true. Some segments
# contain a literal backslash-n instead of a real newline, so both are matched. A line break
# before the number is required, so "The ratio is 3:1." is not mistaken for one.
_LIST_OPENER = re.compile(r":[ \t]*(\n|\\n)(\s|\\n)*\d+[.)]?\s*$")


def is_claim(segment: str) -> bool:
    """False for segments that aren't checkable statements: too short, or a list opener."""
    return len(segment.split()) >= MIN_WORDS and not _LIST_OPENER.search(segment)


def fit_logistic(confidences: list[float], labels: list[bool]) -> LogisticEmission:
    """Platt scaling: logistic regression of the true/false label on logit(confidence).

    Two parameters instead of a 5-bucket table, so it stays stable with few false examples
    (53 in our fit half) and can't flip direction from bucket to bucket.
    """
    eps = LogisticEmission.eps
    clipped = [min(max(c, eps), 1 - eps) for c in confidences]
    x = [[math.log(c / (1 - c))] for c in clipped]
    model = LogisticRegression().fit(x, labels)
    return LogisticEmission(
        slope=float(model.coef_[0][0]),
        intercept=float(model.intercept_[0]),
        base_rate_true=sum(labels) / len(labels),
    )


def load_felm_science_answers() -> list[FelmAnswer]:
    path = hf_hub_download(repo_id="hkust-nlp/felm", filename="all.jsonl", repo_type="dataset")
    answers: list[FelmAnswer] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("domain") != "science":
                continue
            # Two science answers appear twice, word for word with identical labels -- keep
            # one copy, or the duplicate can land in the opposite fit/test half.
            if record["response"].strip() in seen:
                continue
            seen.add(record["response"].strip())
            sentences = record["segmented_response"]
            labels = record["labels"]
            segments = [
                (s, bool(label)) for s, label in zip(sentences, labels, strict=True) if is_claim(s)
            ]
            if segments:
                answers.append(FelmAnswer(record["prompt"], record["response"], segments))
    return answers


def load_felm_science_responses() -> list[Response]:
    return [a.segments for a in load_felm_science_answers()]


def split_answers(
    responses: list[Response], fit_fraction: float, seed: int
) -> tuple[list[Response], list[Response]]:
    """Splits whole answers (not individual segments) into a fit set and a test set.

    Segments from the same answer share topic and wording, so splitting segment-by-segment
    would still let the test set "see" its own answers during fitting. Answers with and
    without any false segment are split separately, so both halves get a fair share of
    false examples instead of whatever one shuffle happens to give.
    """
    rng = random.Random(seed)
    fit: list[Response] = []
    test: list[Response] = []
    has_error = [r for r in responses if not all(label for _, label in r)]
    error_free = [r for r in responses if all(label for _, label in r)]
    for group in (has_error, error_free):
        shuffled = group[:]
        rng.shuffle(shuffled)
        num_fit = round(len(shuffled) * fit_fraction)
        fit += shuffled[:num_fit]
        test += shuffled[num_fit:]
    return fit, test


def split_by_response(
    responses: list[Response], fit_fraction: float, seed: int
) -> tuple[list[tuple[str, bool]], list[tuple[str, bool]]]:
    """split_answers, flattened to segments."""
    fit, test = split_answers(responses, fit_fraction, seed)
    return [seg for r in fit for seg in r], [seg for r in test for seg in r]


def load_felm_science_test_answers(standalone: dict[str, str] | None = None) -> list[Response]:
    """The test half grouped by answer (same split as load_felm_science_split), for the
    answer-level graph (Phase 3)."""
    _, test = split_answers(load_felm_science_responses(), FIT_FRACTION, SPLIT_SEED)
    standalone = standalone or {}
    return [[(standalone.get(s, s), label) for s, label in r] for r in test]


def standalone_path(run_name: str) -> Path:
    return Path(f"results/felm_science_standalone_{run_name}.json")


def load_felm_science_split(
    standalone: dict[str, str] | None = None,
) -> tuple[list[tuple[str, bool]], list[tuple[str, bool]]]:
    """Fit and test halves. With `standalone` (original segment -> rewritten stand-alone
    version, from scripts/decontextualize_felm.py), each segment is replaced by its rewrite;
    the label always stays the original segment's human label."""
    fit, test = split_by_response(load_felm_science_responses(), FIT_FRACTION, SPLIT_SEED)
    if standalone is None:
        return fit, test
    return (
        [(standalone.get(s, s), label) for s, label in fit],
        [(standalone.get(s, s), label) for s, label in test],
    )


def result_name(args: argparse.Namespace, run_name: str) -> str:
    """Result-file name: one per model, plus a separate one for the stand-alone rewrites."""
    return run_name + ("-standalone" if args.standalone else "")


def add_standalone_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="use the stand-alone rewrites from scripts/decontextualize_felm.py",
    )


def load_standalone(args: argparse.Namespace, run_name: str) -> dict[str, str] | None:
    if not args.standalone:
        return None
    data: dict[str, str] = json.loads(standalone_path(run_name).read_text(encoding="utf-8"))[
        "rewrites"
    ]
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    add_standalone_arg(parser)
    args = parser.parse_args()
    backend = make_backend(args)

    examples, _ = load_felm_science_split(load_standalone(args, backend.run_name))
    num_false = sum(1 for _, is_true in examples if not is_true)
    print(
        f"Fitting on {len(examples)} FELM-Science segments ({num_false} false) -- "
        "the fit half only; the test half is never seen here."
    )

    print(f"LLM: {backend.model} ({backend.confidence_method})")
    llm = backend.cached("emission-fit")

    num_bins = len(BIN_EDGES) + 1
    counts_true = [0] * num_bins
    counts_false = [0] * num_bins
    confidences: list[float] = []

    for i, (sentence, is_true) in enumerate(examples):
        confidence = llm.true_false_confidence(sentence)
        confidences.append(confidence)
        b = bin_index(confidence)
        if is_true:
            counts_true[b] += 1
        else:
            counts_false[b] += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            print(f"  scored {i + 1}/{len(examples)}")

    total_true = sum(counts_true)
    total_false = sum(counts_false)
    emission_probs_true = [c / total_true if total_true else 0.0 for c in counts_true]
    emission_probs_false = [c / total_false if total_false else 0.0 for c in counts_false]

    print("counts_true:", counts_true)
    print("counts_false:", counts_false)
    print("emission_probs_true:", emission_probs_true)
    print("emission_probs_false:", emission_probs_false)
    logistic = fit_logistic(confidences, [is_true for _, is_true in examples])
    print(f"logistic: slope={logistic.slope:.3f} intercept={logistic.intercept:.3f}")

    result = {
        "dataset": "FELM-Science",
        "llm": backend.model,
        "confidence_method": backend.confidence_method,
        "split": {"part": "fit", "fit_fraction": FIT_FRACTION, "seed": SPLIT_SEED},
        "standalone_rewrites": bool(args.standalone),
        "num_examples": len(examples),
        "bins_sep": BIN_EDGES,
        "counts_true": counts_true,
        "counts_false": counts_false,
        "emission_probs_true": emission_probs_true,
        "emission_probs_false": emission_probs_false,
        # Ready to build src.inference.hmt.LookupEmission(table=...) from directly:
        # bin index -> {is_true: probability}.
        "lookup_table": {
            str(b): {"true": emission_probs_true[b], "false": emission_probs_false[b]}
            for b in range(num_bins)
        },
        # Smooth alternative to the table (src.inference.hmt.LogisticEmission).
        "logistic": {
            "slope": logistic.slope,
            "intercept": logistic.intercept,
            "base_rate_true": logistic.base_rate_true,
        },
    }

    out_path = Path(
        f"results/emission_model_felm_science_{result_name(args, backend.run_name)}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
