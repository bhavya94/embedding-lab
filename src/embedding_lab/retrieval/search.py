"""Exact nearest-neighbour search over an embedding matrix: the baseline every approximate index is measured against."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from embedding_lab.retrieval.metrics import Ranked


@dataclass(frozen=True)
class ExactSearch:
    """Scores every document for every query, so the top k is exact. Cost grows with documents times queries."""

    doc_ids: list[str]
    embeddings: np.ndarray  # (documents, dimensions)
    similarity: Literal["cosine", "dot"] = "cosine"

    def __post_init__(self):
        if len(self.doc_ids) != len(self.embeddings):
            raise ValueError(f"{len(self.doc_ids)} ids for {len(self.embeddings)} embeddings")
        if self.similarity not in ("cosine", "dot"):
            raise ValueError(f"similarity must be 'cosine' or 'dot', got {self.similarity!r}")

    def scores(self, query_embeddings: np.ndarray) -> np.ndarray:
        """(queries, documents) similarity matrix."""
        docs, queries = self.embeddings, query_embeddings
        if self.similarity == "cosine":
            docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        return queries @ docs.T

    def search(self, query_ids: list[str], query_embeddings: np.ndarray, k: int) -> Ranked:
        scores = self.scores(query_embeddings)
        # A full stable sort: NFCorpus holds duplicate abstracts with equal scores, and ties must break the same
        # way (corpus order) whatever k is. Sorting every score is fine at this corpus size.
        ranked = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return Ranked(query_ids=list(query_ids), doc_ids=[[self.doc_ids[i] for i in row] for row in ranked])
