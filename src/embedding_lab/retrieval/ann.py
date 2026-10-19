"""FAISS indexes: exact (flat) and approximate (HNSW) nearest-neighbour search on unit vectors.

Both use inner product, which equals cosine similarity when every vector has unit length.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

import faiss
import numpy as np

from embedding_lab.retrieval.metrics import Ranked

T = TypeVar("T")


def timed(run: Callable[[], T], warm_up: bool = False) -> tuple[T, float]:
    """Run once and return (result, seconds). `warm_up` runs it once untimed first, so caches are filled."""
    if warm_up:
        run()
    start = time.perf_counter()
    result = run()
    return result, time.perf_counter() - start


def overlap_at_k(ranked: Ranked, exact: Ranked, k: int) -> np.ndarray:
    """Per query: the share of the exact top k that the approximate top k also found."""
    return np.array([len(set(got[:k]) & set(want[:k])) / k for got, want in zip(ranked.doc_ids, exact.doc_ids)])


@dataclass
class FlatIndex:
    """Exact search: compares the query with every vector. The baseline for recall and latency."""

    doc_ids: list[str]
    embeddings: np.ndarray
    index: faiss.Index | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if len(self.doc_ids) != len(self.embeddings):
            raise ValueError(f"{len(self.doc_ids)} ids for {len(self.embeddings)} embeddings")

    def _new_index(self, dimensions: int) -> faiss.Index:
        return faiss.IndexFlatIP(dimensions)

    def build(self) -> None:
        vectors = np.ascontiguousarray(self.embeddings, dtype=np.float32)
        self.index = self._new_index(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query_ids: list[str], query_embeddings: np.ndarray, k: int) -> Ranked:
        if self.index is None:
            self.build()
        _, rows = self.index.search(np.ascontiguousarray(query_embeddings, dtype=np.float32), k)
        # FAISS pads with -1 when it finds fewer than k results.
        return Ranked(query_ids=list(query_ids), doc_ids=[[self.doc_ids[i] for i in row if i >= 0] for row in rows])


@dataclass
class HNSWIndex(FlatIndex):
    """Approximate search on a layered proximity graph (HNSW).

    `neighbors` (M) sets the links per node: more links, better recall, more memory and build time.
    `ef_search` sets how many candidates a query explores: higher means better recall and slower queries.
    """

    neighbors: int = 32
    ef_construction: int = 200
    ef_search: int = 64

    def _new_index(self, dimensions: int) -> faiss.Index:
        index = faiss.IndexHNSWFlat(dimensions, self.neighbors, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        return index

    def set_ef_search(self, ef_search: int) -> None:
        """Change how many candidates a query explores, without rebuilding the graph."""
        self.ef_search = ef_search
        if self.index is not None:
            self.index.hnsw.efSearch = ef_search
