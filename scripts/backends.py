"""Which LLM the experiment scripts use, chosen on the command line:

    --backend groq                 free hosted API, typed-out confidence (provisional)
    --backend hf [--model NAME]    open-source model run ourselves, real token probabilities

Each backend gets its own cache tag and result filenames, so runs never mix or overwrite.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.llm.base import LLMClient
from src.llm.cache import CachingLLMClient
from src.llm.groq_client import GroqClient
from src.llm.hf_client import DEFAULT_MODEL, HuggingFaceClient


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


def make_backend(args: argparse.Namespace) -> Backend:
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
