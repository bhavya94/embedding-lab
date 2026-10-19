import numpy as np
import pytest

from embedding_lab.retrieval.search import ExactSearch

rng = np.random.default_rng(0)
DOCS = rng.normal(size=(50, 8))
QUERIES = rng.normal(size=(4, 8))
IDS = [f"d{i}" for i in range(50)]


def test_top_k_matches_a_full_sort_of_cosine_similarity():
    ranked = ExactSearch(IDS, DOCS).search(["a", "b", "c", "d"], QUERIES, k=5)
    unit_docs = DOCS / np.linalg.norm(DOCS, axis=1, keepdims=True)
    unit_queries = QUERIES / np.linalg.norm(QUERIES, axis=1, keepdims=True)
    for row, doc_ids in zip(unit_queries @ unit_docs.T, ranked.doc_ids):
        assert doc_ids == [IDS[i] for i in np.argsort(-row)[:5]]


def test_dot_and_cosine_agree_on_unit_vectors_and_differ_otherwise():
    unit = DOCS / np.linalg.norm(DOCS, axis=1, keepdims=True)
    cosine = ExactSearch(IDS, unit, "cosine").search(["q"], QUERIES[:1], k=10)
    dot = ExactSearch(IDS, unit, "dot").search(["q"], QUERIES[:1], k=10)
    assert cosine.doc_ids == dot.doc_ids
    stretched = DOCS * np.linspace(0.1, 5, 50)[:, None]  # same directions, very different lengths
    assert ExactSearch(IDS, stretched, "dot").search(["q"], QUERIES[:1], k=10).doc_ids != cosine.doc_ids


def test_k_larger_than_the_corpus_returns_every_doc():
    assert len(ExactSearch(IDS, DOCS).search(["q"], QUERIES[:1], k=500).doc_ids[0]) == 50


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="ids for"):
        ExactSearch(IDS[:3], DOCS)
    with pytest.raises(ValueError, match="similarity"):
        ExactSearch(IDS, DOCS, "euclid")


def test_tied_scores_rank_in_corpus_order_whatever_k_is():
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])  # d0, d2, d3 tie for the query
    search = ExactSearch(["d0", "d1", "d2", "d3"], docs)
    query = np.array([[1.0, 0.0]])
    for k in [1, 2, 3]:
        assert search.search(["q"], query, k=k).doc_ids == [["d0", "d2", "d3"][:k]]
    assert search.search(["q"], query, k=4).doc_ids == [["d0", "d2", "d3", "d1"]]
