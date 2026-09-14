"""Fits a real emission model from labelled data: for statements the LLM scored with a given
confidence, how often are they actually true vs. false? Mirrors
docs/04-research-notes-btprop.md Step 4, and the same idea as the official repo's
configs/hmm.yaml -- but fit fresh here, on our own dataset and LLM, not copied from theirs.

This is a *calibration* question about the LLM's confidence scores on their own (does 0.9
confidence mean ~90% likely true?) -- it has nothing to do with belief-tree structure or the
transition parameters in src/tree/relationships.py, which are a separate concept.

Dataset: FELM-Science (hkust-nlp/felm, "science" domain) -- 125 real sentences, each already
hand-labelled true/false. Note: the HuggingFace `datasets` library can't load this repo anymore
(it ships a now-unsupported loading script), so we pull the raw all.jsonl file directly instead.

Usage: uv run python scripts/fit_emission_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.llm.cache import CachingLLMClient
from src.llm.groq_client import GroqClient

# Same 5 bins (4 edges) as the official repo's configs/hmm.yaml, so our numbers stay
# comparable to theirs even though we fit them ourselves.
BIN_EDGES = [0.19, 0.39, 0.69, 0.89]


def bin_index(score: float) -> int:
    for i, edge in enumerate(BIN_EDGES):
        if score <= edge:
            return i
    return len(BIN_EDGES)


def load_felm_science() -> list[tuple[str, bool]]:
    path = hf_hub_download(repo_id="hkust-nlp/felm", filename="all.jsonl", repo_type="dataset")
    examples: list[tuple[str, bool]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("domain") != "science":
                continue
            sentences = record["segmented_response"]
            labels = record["labels"]
            for sentence, label in zip(sentences, labels, strict=True):
                examples.append((sentence, bool(label)))
    return examples


def main() -> None:
    examples = load_felm_science()
    print(f"Loaded {len(examples)} labelled sentences from FELM-Science.")

    # Groq, not OpenAI: free, but text-based self-reported confidence (see
    # src/llm/groq_client.py) -- so this emission table is a provisional fit, not the paper's
    # faithful reproduction. Re-fit with OpenAIClient once billing credit is available.
    llm = CachingLLMClient(
        inner=GroqClient(), cache_dir=Path("llm_cache"), model_tag="groq-emission-fit"
    )

    num_bins = len(BIN_EDGES) + 1
    counts_true = [0] * num_bins
    counts_false = [0] * num_bins

    for i, (sentence, is_true) in enumerate(examples):
        confidence = llm.true_false_confidence(sentence)
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

    result = {
        "dataset": "FELM-Science",
        "llm": "groq/" + llm.inner.model if isinstance(llm.inner, GroqClient) else "unknown",
        "confidence_method": "text-based self-report (provisional, see src/llm/groq_client.py)",
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
    }

    out_path = Path("results/emission_model_felm_science_groq.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
