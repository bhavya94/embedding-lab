import pytest

from embedding_lab.data.nfcorpus import NFCorpus


@pytest.fixture(scope="module")
def nfcorpus():
    return NFCorpus().load()


def test_test_split_has_the_published_sizes(nfcorpus):
    corpus, queries, relevance = nfcorpus
    assert len(corpus) == 3_633 and len(queries) == 323
    assert set(queries.ids) == set(relevance.grades)


def test_every_labeled_doc_is_in_the_corpus_with_a_known_grade(nfcorpus):
    corpus, _, relevance = nfcorpus
    ids = set(corpus.ids)
    labeled = {doc_id: grade for grades in relevance.grades.values() for doc_id, grade in grades.items()}
    assert set(labeled) <= ids and set(labeled.values()) == {1, 2}


def test_passages_prefix_the_title_when_there_is_one(nfcorpus):
    corpus, _, _ = nfcorpus
    assert corpus.passages()[0].startswith(corpus.titles[0] + ". ")
