"""Runs an open-source model ourselves (Hugging Face `transformers`), so we can read the
model's real next-token probabilities -- the base paper's confidence method -- for free.

Hosted free APIs (Groq) refuse to return these probabilities, and paid ones (OpenAI) cost
money; running the weights directly gives them to us exactly. Meant for a GPU machine
(Kaggle's free 2xT4 for Llama-3-8B); small models also run on CPU for testing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from src.llm.prompts import true_false_confidence_prompt

# The model the BTProp paper itself evaluates. The official meta-llama/ repo needs a manual
# licence approval; this is an ungated copy of the same weights (same Llama 3 licence).
DEFAULT_MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"

_TRUE_VARIANTS = ("True", " True", "true", " true", "TRUE")
_FALSE_VARIANTS = ("False", " False", "false", " false", "FALSE")


def true_probability(
    next_token_probs: Sequence[float], true_ids: set[int], false_ids: set[int]
) -> float:
    """P(True) / (P(True) + P(False)) from a full next-token distribution. 0.5 if the model
    puts no mass on either word -- same fallback as OpenAIClient."""
    p_true = sum(next_token_probs[i] for i in true_ids)
    p_false = sum(next_token_probs[i] for i in false_ids)
    total = p_true + p_false
    return 0.5 if total == 0.0 else p_true / total


@dataclass
class HuggingFaceClient:
    """Implements the LLMClient protocol by running the model locally.

    device_map: "auto" spreads the model across every visible GPU (needs `accelerate`; how
        Llama-3-8B fits on Kaggle's 2xT4). Ignored without a GPU, where "auto" would offload
        weights to disk. None puts the model on one device -- GPU if available, else CPU.
    """

    model_name: str = DEFAULT_MODEL
    device_map: str | None = "auto"

    @cached_property
    def _tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.model_name)

    @cached_property
    def _model(self) -> Any:
        import torch
        from transformers import AutoModelForCausalLM

        has_gpu = torch.cuda.is_available()
        # float16, not bfloat16: Kaggle's T4s don't support bfloat16 natively.
        dtype = torch.float16 if has_gpu else torch.float32
        model: Any
        if self.device_map is not None and has_gpu:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=dtype, device_map=self.device_map
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
            model.to("cuda" if has_gpu else "cpu")
        model.eval()
        return model

    @cached_property
    def _true_false_ids(self) -> tuple[set[int], set[int]]:
        # Only variants that are a single token can be the model's very next token.
        def single_token_ids(variants: tuple[str, ...]) -> set[int]:
            ids = set()
            for v in variants:
                encoded = self._tokenizer.encode(v, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(encoded[0])
            return ids

        return single_token_ids(_TRUE_VARIANTS), single_token_ids(_FALSE_VARIANTS)

    def _chat_input_ids(self, prompt: str) -> Any:
        # Render the chat template to text first, then tokenize: apply_chat_template's own
        # return type differs between transformers versions (Kaggle's vs. ours).
        text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        encoded = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"].to(self._model.device)

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        import torch

        input_ids = self._chat_input_ids(prompt)
        sampling: dict[str, Any] = (
            {"do_sample": True, "temperature": temperature}
            if temperature > 0
            else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        )
        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
                **sampling,
            )
        new_tokens = output[0, input_ids.shape[1] :]
        text: str = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    def true_false_confidence(self, statement: str) -> float:
        import torch

        input_ids = self._chat_input_ids(true_false_confidence_prompt(statement))
        with torch.no_grad():
            logits = self._model(input_ids).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1).tolist()
        true_ids, false_ids = self._true_false_ids
        return true_probability(probs, true_ids, false_ids)
