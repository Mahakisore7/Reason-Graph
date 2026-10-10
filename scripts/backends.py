"""Which LLM the experiment scripts use, chosen on the command line:

    --backend groq                 free hosted API, typed-out confidence (provisional)
    --backend hf [--model NAME]    open-source model run ourselves, real token probabilities

Each backend gets its own cache tag and result filenames, so runs never mix or overwrite.
--offline replays saved answers only (e.g. re-scoring Kaggle's Llama run on a laptop with no
GPU) and stops if an answer is missing, instead of trying to run the model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.llm.base import LLMClient
from src.llm.cache import CachingLLMClient
from src.llm.groq_client import GroqClient
from src.llm.hf_client import DEFAULT_MODEL, HuggingFaceClient


class _CacheOnlyClient:
    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        raise RuntimeError(f"--offline: no saved answer for prompt {prompt[:80]!r}")

    def true_false_confidence(self, statement: str) -> float:
        raise RuntimeError(f"--offline: no saved confidence for {statement[:80]!r}")


@dataclass(frozen=True)
class Backend:
    run_name: str  # used in cache tags and result filenames, e.g. "groq"
    model: str
    confidence_method: str
    inner: LLMClient

    def cached(self, stage: str) -> CachingLLMClient:
        return CachingLLMClient(
            inner=self.inner, cache_dir=Path("llm_cache"), model_tag=f"{self.run_name}-{stage}"
        )


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("groq", "hf"), default="groq")
    parser.add_argument("--model", default=None, help=f"hf backend only (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--offline", action="store_true", help="replay saved answers only; never call the model"
    )


def make_backend(args: argparse.Namespace) -> Backend:
    backend = _make_backend(args)
    if args.offline:
        return Backend(
            backend.run_name, backend.model, backend.confidence_method, _CacheOnlyClient()
        )
    return backend


def _make_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "groq":
        groq = GroqClient()
        # run_name stays plain "groq" so existing cached Groq answers keep being reused.
        return Backend("groq", groq.model, "typed-out self-report (provisional)", groq)
    model = args.model or DEFAULT_MODEL
    return Backend(
        run_name=model.split("/")[-1].lower(),
        model=model,
        confidence_method="next-token probability of True vs False (paper's method)",
        inner=HuggingFaceClient(model_name=model),
    )
