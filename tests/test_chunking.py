import pytest

from embedding_lab.data.long_docs import LongDocRecipe, sentences
from embedding_lab.data.nfcorpus import Corpus, Queries, Relevance
from embedding_lab.retrieval.chunking import (Chunk, auto_merging_chunks, fixed_chunks, merge_hits,
                                              sentence_windows)

DOC = [f"Sentence {i}." for i in range(10)]


def test_sentences_split_on_end_punctuation():
    assert sentences("One. Two? Three! Four") == ["One.", "Two?", "Three!", "Four"]


def test_fixed_chunks_cover_the_doc_without_overlap():
    chunks = fixed_chunks("d", DOC, 4)
    assert [(c.start, c.end) for c in chunks] == [(0, 4), (4, 8), (8, 10)]
    assert chunks[0].embed_text == "Sentence 0. Sentence 1. Sentence 2. Sentence 3."


def test_sentence_windows_embed_one_sentence_and_return_its_neighbours():
    chunks = sentence_windows("d", DOC, window=2)
    assert len(chunks) == 10 and chunks[5].embed_text == "Sentence 5."
    assert (chunks[5].start, chunks[5].end) == (3, 8)
    assert (chunks[0].start, chunks[0].end) == (0, 3)  # clipped at the edge


def test_merge_returns_the_parent_once_most_children_hit():
    children = auto_merging_chunks("d", DOC, child=2, parent=4)
    assert [c.parent for c in children] == [0, 0, 1, 1, 2]
    # Both children of parent 0 hit, one child of parent 1: parent 0 merges, the lone child stays.
    hits = [children[1], children[2], children[0]]
    assert merge_hits(hits, children) == [("d", 0, 4), ("d", 4, 6)]


def test_a_short_last_parent_merges_and_ends_at_the_document_end():
    children = auto_merging_chunks("d", DOC, child=2, parent=4)  # parent 2 holds one child: sentences 8-9
    assert merge_hits([children[4]], children) == [("d", 8, 10)]


def test_parent_size_must_be_a_multiple_of_the_child_size():
    with pytest.raises(ValueError, match="multiple"):
        auto_merging_chunks("d", DOC, child=3, parent=4)


def _toy_set(n_docs=40):
    ids = [f"c{i}" for i in range(n_docs)]
    corpus = Corpus(ids=ids, titles=[f"T{i}" for i in range(n_docs)],
                    texts=[f"Body {i}. More {i}." for i in range(n_docs)])
    queries = Queries(ids=["q0", "q1", "q2"], texts=["first", "second", "third"])
    # q2's gold (c0) is q0's gold too, and q0 is relevant to c2 (q1's gold): only one of each pair can be kept.
    relevance = Relevance(grades={"q0": {"c0": 2, "c1": 1, "c2": 1}, "q1": {"c2": 1}, "q2": {"c0": 1}})
    return corpus, queries, relevance


def test_each_long_doc_has_every_part_and_one_gold_abstract():
    corpus, queries, relevance = _toy_set()
    docs = LongDocRecipe(abstracts_per_doc=5, queries=2).build(corpus, queries, relevance)
    assert len(docs.ids) == 2
    for n, query_id in enumerate(docs.query_ids):
        assert len(docs.sentences[n]) == 5 * 3  # five abstracts of three sentences each
        start, end = docs.gold_spans[docs.ids.index(docs.gold_doc[query_id])]
        assert end - start == 3


def test_no_chosen_query_is_relevant_to_another_query_gold_or_to_filler():
    corpus, queries, relevance = _toy_set()
    docs = LongDocRecipe(abstracts_per_doc=5, queries=2).build(corpus, queries, relevance)
    for n, query_id in enumerate(docs.query_ids):
        text = " ".join(docs.sentences[n])
        start, end = docs.gold_spans[n]
        gold_text = " ".join(docs.sentences[n][start:end])
        for other in docs.query_ids:
            for doc_id in relevance.relevant(other):
                body = f"Body {doc_id[1:]}."
                if body in text:
                    assert body in gold_text and other == query_id


def test_too_few_queries_or_filler_is_an_error_not_a_short_set():
    corpus, queries, relevance = _toy_set()
    with pytest.raises(ValueError, match="only 2 queries"):
        LongDocRecipe(abstracts_per_doc=5, queries=3).build(corpus, queries, relevance)
    small, _, _ = _toy_set(n_docs=8)
    with pytest.raises(ValueError, match="filler"):
        LongDocRecipe(abstracts_per_doc=5, queries=2).build(small, queries, relevance)
