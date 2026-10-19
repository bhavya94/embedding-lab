"""Smaller vectors: keep fewer dimensions, by cutting or by projecting onto the main directions of the corpus."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class Reducer:
    """Maps vectors to `dimensions` dimensions, then back to unit length.

    - `"truncate"` keeps the first `dimensions` coordinates.
    - `"pca"` subtracts the corpus mean, then projects onto the top principal directions. Subtracting the mean
      moves the origin, so it changes cosine similarity even when no dimension is dropped.
    - `"svd"` projects onto the top singular directions without subtracting the mean (truncated SVD). With every
      dimension kept it is a rotation, so cosine similarity is unchanged.
    """

    dimensions: int
    method: Literal["truncate", "pca", "svd"] = "svd"
    mean: np.ndarray | None = field(default=None, init=False, repr=False)
    components: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.method not in ("truncate", "pca", "svd"):
            raise ValueError(f"method must be 'truncate', 'pca' or 'svd', got {self.method!r}")
        if self.dimensions < 1:
            raise ValueError(f"dimensions must be at least 1, got {self.dimensions}")

    def fit(self, vectors: np.ndarray) -> "Reducer":
        if self.method != "truncate":
            self.mean = vectors.mean(axis=0) if self.method == "pca" else np.zeros(vectors.shape[1])
            # Rows of vt are the directions, the one with most (centered or raw) energy first.
            _, _, vt = np.linalg.svd(vectors - self.mean, full_matrices=False)
            self.components = vt[: self.dimensions]
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        if self.method == "truncate":
            reduced = vectors[:, : self.dimensions]
        else:
            if self.components is None:
                raise ValueError(f"call fit() before transform() for {self.method}")
            reduced = (vectors - self.mean) @ self.components.T
        return reduced / np.linalg.norm(reduced, axis=1, keepdims=True)
