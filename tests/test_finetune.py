import numpy as np

from embedding_lab.data.nfcorpus import Corpus, Queries, Relevance
from embedding_lab.encode.encoder import Encoder
from embedding_lab.finetune.contrastive import (ContrastiveFineTune, false_negative_rate, query_document_ids,
                                                query_document_pairs, title_abstract_pairs)

CORPUS = Corpus(ids=["a", "b", "c", "d"], titles=["Statins and survival", "", "Fibre intake", "Sleep and memory"],
                texts=["Statin users lived longer.", "No title here.", "", "Short sleep hurt recall."])


def test_pairs_skip_documents_missing_a_title_or_an_abstract():
    pairs = title_abstract_pairs(CORPUS)
    assert pairs["anchor"] == ["Statins and survival", "Sleep and memory"]
    assert pairs["positive"] == ["Statin users lived longer.", "Short sleep hurt recall."]


def test_a_run_changes_the_model_and_a_second_run_reuses_it(tmp_path):
    # Three epochs: with one step, learning-rate warmup makes that step's rate zero.
    recipe = ContrastiveFineTune(name="test", epochs=3, batch_size=2, learning_rate=5e-4, max_length=32,
                                 directory=tmp_path)
    pairs = title_abstract_pairs(CORPUS)
    path = recipe.run(pairs)
    assert (path / "config.json").exists()
    texts = ["Statins and survival", "Sleep and memory"]
    before = Encoder(device="cpu").encode(texts)
    after = Encoder(model_name=str(path), device="cpu").encode(texts)
    assert not np.allclose(before, after, atol=1e-3)
    modified = (path / "config.json").stat().st_mtime
    assert recipe.run(pairs) == path and (path / "config.json").stat().st_mtime == modified
    assert recipe.output(pairs.select([0])) != path  # different pairs, different model


def test_query_pairs_cap_documents_per_query_skip_shared_texts_and_use_whole_passages():
    queries = Queries(ids=["q1", "q2", "q3"], texts=["statins?", "sleep?", "fibre?"])
    relevance = Relevance(grades={"q1": {"a": 1, "c": 1, "d": 1}, "q2": {"d": 1}, "q3": {"c": 1}})
    ids = query_document_ids(queries, relevance, per_query=2, exclude_texts=frozenset({"fibre?"}))
    assert [q for q, _ in ids] == ["q1", "q1", "q2"] and ids[2] == ("q2", "d")
    pairs = query_document_pairs(CORPUS, queries, ids)
    assert pairs["anchor"] == ["statins?", "statins?", "sleep?"]
    assert pairs["positive"][2] == "Sleep and memory. Short sleep hurt recall."


def test_false_negative_rate_counts_queries_whose_negatives_include_a_relevant_doc():
    relevance = Relevance(grades={"q1": {"a": 1, "b": 1}, "q2": {"b": 1}})
    # q1's negative is b, which is relevant to q1; q2's negative is a, which is not relevant to q2.
    assert false_negative_rate([("q1", "a"), ("q2", "b")], relevance, batch_size=2, batches=5) == 0.5


def test_false_negative_batches_never_repeat_a_query():
    relevance = Relevance(grades={"q1": {"a": 1, "b": 1}})
    # Both rows belong to q1, so a batch holds only one of them and q1 has no negatives at all.
    assert false_negative_rate([("q1", "a"), ("q1", "b")], relevance, batch_size=2, batches=5) == 0.0
