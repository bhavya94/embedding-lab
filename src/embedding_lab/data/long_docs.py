"""Long documents with a known answer inside, for testing how chunking affects retrieval.

NFCorpus abstracts are short enough to embed whole. To see chunking matter, stitch 10 abstracts into one long
"document". A query is relevant to the long document that contains its best abstract, and the passage it
should land on is that abstract. The other 9 abstracts are filler that no chosen query is relevant to. Queries
are chosen so that none is relevant to another query's gold abstract, so each query has exactly one right
long document.
"""

import re
from dataclasses import dataclass

import numpy as np

from embedding_lab.data.nfcorpus import Corpus, Queries, Relevance


def sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace. Good enough for abstracts."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


@dataclass(frozen=True)
class LongDocs:
    """Each long document is a list of sentences, plus the sentence range of its gold abstract."""

    ids: list[str]
    sentences: list[list[str]]
    gold_spans: list[tuple[int, int] | None]  # [start, end) of the gold abstract; None for filler-only docs
    query_ids: list[str]
    query_texts: list[str]
    gold_doc: dict[str, str]  # query id -> long document id


@dataclass(frozen=True)
class LongDocRecipe:
    abstracts_per_doc: int = 10
    queries: int = 100
    seed: int = 0

    def build(self, corpus: Corpus, queries: Queries, relevance: Relevance) -> LongDocs:
        rng = np.random.default_rng(self.seed)
        position = {doc_id: i for i, doc_id in enumerate(corpus.ids)}
        # One query per long doc. Its gold is its best abstract (highest grade, then first listed). Skip a query
        # if its gold is relevant to an already chosen query, or if it is relevant to an already chosen gold.
        chosen, golds, relevant_anywhere = [], set(), set()
        for i in rng.permutation(len(queries)):
            query_id = queries.ids[i]
            relevant = relevance.relevant(query_id)
            best = max(relevant.items(), key=lambda item: item[1])[0]
            if best in relevant_anywhere or golds & set(relevant):
                continue
            chosen.append((query_id, queries.texts[i], best))
            golds.add(best)
            relevant_anywhere |= set(relevant)
            if len(chosen) == self.queries:
                break
        if len(chosen) < self.queries:
            raise ValueError(f"only {len(chosen)} queries have a gold abstract no other chosen query shares")
        # Filler must not be relevant to any chosen query, or "wrong" chunks could be right.
        filler = [doc_id for doc_id in corpus.ids if doc_id not in relevant_anywhere]
        needed = self.queries * (self.abstracts_per_doc - 1)
        if len(filler) < needed:
            raise ValueError(f"{len(filler)} filler abstracts for {needed} slots; lower `queries`")
        rng.shuffle(filler)
        passages = corpus.passages()
        ids, all_sentences, spans, gold_doc = [], [], [], {}
        for n, (query_id, _, best) in enumerate(chosen):
            parts = filler[n * (self.abstracts_per_doc - 1):(n + 1) * (self.abstracts_per_doc - 1)]
            slot = int(rng.integers(self.abstracts_per_doc))
            parts = parts[:slot] + [best] + parts[slot:]
            doc_sentences, span = [], None
            for doc_id in parts:
                split = sentences(passages[position[doc_id]])
                if doc_id == best:
                    span = (len(doc_sentences), len(doc_sentences) + len(split))
                doc_sentences += split
            ids.append(f"long-{n}")
            all_sentences.append(doc_sentences)
            spans.append(span)
            gold_doc[query_id] = f"long-{n}"
        return LongDocs(ids=ids, sentences=all_sentences, gold_spans=spans,
                        query_ids=[query_id for query_id, _, _ in chosen],
                        query_texts=[text for _, text, _ in chosen], gold_doc=gold_doc)
