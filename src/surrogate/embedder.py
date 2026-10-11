"""Text embedding for the surrogate confidence scorer (Our Novelty, Extension 2).

HashingEmbedder maps words and word pairs to vector positions with a hash function: no learned
vocabulary, no downloaded weights, nothing to train -- deliberately cheap. A heavier embedding
(sentence-transformers) can be swapped in through the Embedder protocol if the numbers show the
cheap one isn't good enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class HashingEmbedder:
    # 4096 rather than the plan's 256: with words AND word pairs, 256 slots would make many
    # unrelated words share a slot (hash collisions). Still tiny and free to compute.
    n_features: int = 4096
    _vectorizer: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=self.n_features,
            ngram_range=(1, 2),
            # Counts stay non-negative: a word is either present or not, it never "cancels" one
            # that happens to hash to the same slot with the opposite sign.
            alternate_sign=False,
            # Unit length, so a long sentence doesn't get bigger feature values than a short
            # one just for having more words.
            norm="l2",
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._vectorizer.transform(texts).toarray())
