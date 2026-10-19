"""Ranking metrics: how good a list of retrieved documents is, per query, against graded relevance labels."""

from dataclasses import dataclass

import numpy as np

from embedding_lab.data.nfcorpus import Relevance


@dataclass(frozen=True)
class Ranked:
    """Retrieval output: for each query, its retrieved document ids, best first."""

    query_ids: list[str]
    doc_ids: list[list[str]]

    def __post_init__(self):
        if len(self.query_ids) != len(self.doc_ids):
            raise ValueError(f"{len(self.query_ids)} queries but {len(self.doc_ids)} rankings")


def _per_query(ranked: Ranked, relevance: Relevance, score) -> np.ndarray:
    """Apply `score(retrieved_grades, all_relevant_grades)` to each query; nan when a query has no relevant doc."""
    values = []
    for query_id, doc_ids in zip(ranked.query_ids, ranked.doc_ids):
        relevant = relevance.relevant(query_id)
        if not relevant:
            values.append(np.nan)
            continue
        retrieved = np.array([relevant.get(doc_id, 0) for doc_id in doc_ids], dtype=float)
        values.append(score(retrieved, np.array(sorted(relevant.values(), reverse=True), dtype=float)))
    return np.array(values)


def recall_at_k(ranked: Ranked, relevance: Relevance, k: int) -> np.ndarray:
    """Share of a query's relevant documents (any grade) that appear in its top k."""
    return _per_query(ranked, relevance, lambda retrieved, ideal: np.count_nonzero(retrieved[:k]) / len(ideal))


def ndcg_at_k(ranked: Ranked, relevance: Relevance, k: int) -> np.ndarray:
    """Normalized discounted cumulative gain: graded relevance, discounted by log2(rank + 1), over the ideal order.

    Linear gain (the grade itself), the same convention as sklearn and pytrec_eval.
    """
    discount = 1 / np.log2(np.arange(2, k + 2))

    def score(retrieved: np.ndarray, ideal: np.ndarray) -> float:
        dcg = float(np.sum(retrieved[:k] * discount[: len(retrieved[:k])]))
        return dcg / float(np.sum(ideal[:k] * discount[: len(ideal[:k])]))

    return _per_query(ranked, relevance, score)


def mrr(ranked: Ranked, relevance: Relevance) -> np.ndarray:
    """One over the rank of the first relevant document; 0 when none was retrieved."""
    def score(retrieved: np.ndarray, ideal: np.ndarray) -> float:
        hits = np.flatnonzero(retrieved)
        return 1 / (hits[0] + 1) if hits.size else 0.0

    return _per_query(ranked, relevance, score)
