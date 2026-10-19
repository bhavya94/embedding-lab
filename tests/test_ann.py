import numpy as np
import pytest

from embedding_lab.retrieval.ann import FlatIndex, HNSWIndex, overlap_at_k
from embedding_lab.retrieval.metrics import Ranked
from embedding_lab.retrieval.search import ExactSearch

rng = np.random.default_rng(0)
DOCS = rng.normal(size=(2_000, 32)).astype(np.float32)
DOCS /= np.linalg.norm(DOCS, axis=1, keepdims=True)
QUERIES = rng.normal(size=(50, 32)).astype(np.float32)
QUERIES /= np.linalg.norm(QUERIES, axis=1, keepdims=True)
IDS = [f"d{i}" for i in range(len(DOCS))]
QIDS = [f"q{i}" for i in range(len(QUERIES))]


def test_flat_index_returns_the_exact_cosine_top_k():
    flat = FlatIndex(IDS, DOCS).search(QIDS, QUERIES, 10)
    exact = ExactSearch(IDS, DOCS, "cosine").search(QIDS, QUERIES, 10)
    assert np.mean(overlap_at_k(flat, exact, 10)) == 1.0


def test_hnsw_recall_rises_with_ef_search():
    exact = FlatIndex(IDS, DOCS).search(QIDS, QUERIES, 10)
    low = np.mean(overlap_at_k(HNSWIndex(IDS, DOCS, neighbors=4, ef_search=10).search(QIDS, QUERIES, 10), exact, 10))
    high = np.mean(overlap_at_k(HNSWIndex(IDS, DOCS, ef_search=256).search(QIDS, QUERIES, 10), exact, 10))
    assert high > 0.95 and high >= low


def test_overlap_counts_shared_ids_in_the_top_k():
    got = Ranked(["q"], [["a", "b", "x"]])
    want = Ranked(["q"], [["a", "c", "b"]])
    assert overlap_at_k(got, want, 3).tolist() == [pytest.approx(2 / 3)]


def test_mismatched_ids_are_rejected():
    with pytest.raises(ValueError, match="ids for"):
        FlatIndex(IDS[:5], DOCS)
