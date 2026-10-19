import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from embedding_lab.data.nfcorpus import Relevance
from embedding_lab.retrieval.metrics import Ranked, mrr, ndcg_at_k, recall_at_k

# One query with three relevant docs (d1 highly relevant). The ranking finds d2 first, d1 third, and misses d3.
RELEVANCE = Relevance(grades={"q": {"d1": 2, "d2": 1, "d3": 1}})
RANKED = Ranked(query_ids=["q"], doc_ids=[["d2", "x", "d1", "y", "z"]])


def test_recall_counts_relevant_docs_in_the_top_k():
    assert recall_at_k(RANKED, RELEVANCE, k=1).tolist() == [pytest.approx(1 / 3)]
    assert recall_at_k(RANKED, RELEVANCE, k=3).tolist() == [pytest.approx(2 / 3)]
    assert recall_at_k(RANKED, RELEVANCE, k=5).tolist() == [pytest.approx(2 / 3)]  # d3 was never retrieved


def test_ndcg_matches_hand_computation_and_sklearn():
    dcg = 1 / np.log2(2) + 2 / np.log2(4)  # d2 at rank 1 (grade 1), d1 at rank 3 (grade 2)
    ideal = 2 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)
    assert ndcg_at_k(RANKED, RELEVANCE, k=5)[0] == pytest.approx(dcg / ideal)
    # sklearn scores every candidate at once; give it the same five retrieved docs plus the missed d3 at the end.
    true = np.array([[1, 0, 2, 0, 0, 1]], dtype=float)
    predicted = np.array([[6, 5, 4, 3, 2, 1]], dtype=float)
    assert ndcg_at_k(RANKED, RELEVANCE, k=5)[0] == pytest.approx(ndcg_score(true, predicted, k=5))


def test_mrr_is_one_over_the_first_relevant_rank():
    assert mrr(RANKED, RELEVANCE).tolist() == [1.0]
    assert mrr(Ranked(["q"], [["x", "y", "d1"]]), RELEVANCE).tolist() == [pytest.approx(1 / 3)]
    assert mrr(Ranked(["q"], [["x", "y"]]), RELEVANCE).tolist() == [0.0]


def test_query_without_labels_scores_nan_not_zero():
    scores = ndcg_at_k(Ranked(["q", "unlabeled"], [["d1"], ["d1"]]), RELEVANCE, k=1)
    assert scores[0] == 1.0 and np.isnan(scores[1])


def test_mismatched_rankings_are_rejected():
    with pytest.raises(ValueError, match="2 queries but 1 rankings"):
        Ranked(["a", "b"], [["d1"]])
