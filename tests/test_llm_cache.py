"""Tests for the disk-based LLM response cache (Phase 1). FakeLLMClient makes these fully
deterministic and free -- no network, no API key, ever.
"""

from pathlib import Path

from src.llm.cache import CachingLLMClient
from src.llm.fake import FakeLLMClient


def test_same_arguments_are_served_from_cache_not_called_twice(tmp_path: Path) -> None:
    inner = FakeLLMClient(completions={"hello": "world"})
    client = CachingLLMClient(inner=inner, cache_dir=tmp_path)

    first = client.complete("hello")
    second = client.complete("hello")

    assert first == second == "world"
    assert inner.complete_call_count == 1


def test_different_temperature_is_a_cache_miss(tmp_path: Path) -> None:
    inner = FakeLLMClient(completions={"hello": "world"})
    client = CachingLLMClient(inner=inner, cache_dir=tmp_path)

    client.complete("hello", temperature=0.0)
    client.complete("hello", temperature=0.7)

    assert inner.complete_call_count == 2


def test_cache_survives_a_fresh_client_pointed_at_the_same_directory(tmp_path: Path) -> None:
    inner_a = FakeLLMClient(completions={"hello": "world"})
    client_a = CachingLLMClient(inner=inner_a, cache_dir=tmp_path)
    client_a.complete("hello")

    # A brand new client, new inner fake with NO canned completions at all. If caching
    # is broken, this would raise KeyError instead of returning "world".
    inner_b = FakeLLMClient()
    client_b = CachingLLMClient(inner=inner_b, cache_dir=tmp_path)
    result = client_b.complete("hello")

    assert result == "world"
    assert inner_b.complete_call_count == 0


def test_confidence_scores_are_cached_too(tmp_path: Path) -> None:
    inner = FakeLLMClient(confidences={"the sky is blue": 0.95})
    client = CachingLLMClient(inner=inner, cache_dir=tmp_path)

    first = client.true_false_confidence("the sky is blue")
    second = client.true_false_confidence("the sky is blue")

    assert first == second == 0.95
    assert inner.confidence_call_count == 1
