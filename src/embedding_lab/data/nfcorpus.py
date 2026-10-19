"""NFCorpus: 3,633 medical abstracts and 323 test queries with graded relevance, from the BEIR benchmark."""

from dataclasses import dataclass

from datasets import load_dataset


@dataclass(frozen=True)
class Corpus:
    """The documents to search, in one fixed order so row i of any embedding matrix is `ids[i]`."""

    ids: list[str]
    titles: list[str]
    texts: list[str]

    def __len__(self) -> int:
        return len(self.ids)

    def passages(self) -> list[str]:
        """What gets embedded: the title and the abstract as one string."""
        return [f"{title}. {text}" if title else text for title, text in zip(self.titles, self.texts)]


@dataclass(frozen=True)
class Queries:
    ids: list[str]
    texts: list[str]

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class Relevance:
    """Graded relevance labels: `grades[query_id][doc_id]` is 1 (relevant) or 2 (highly relevant).

    A document missing from a query's labels counts as not relevant.
    """

    grades: dict[str, dict[str, int]]

    def relevant(self, query_id: str) -> dict[str, int]:
        return self.grades.get(query_id, {})


@dataclass(frozen=True)
class NFCorpus:
    """A recipe for loading one split's queries with the full corpus. The Hugging Face cache keeps the download."""

    split: str = "test"

    def load(self) -> tuple[Corpus, Queries, Relevance]:
        corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
        queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
        qrels = load_dataset("BeIR/nfcorpus-qrels", split=self.split)
        grades: dict[str, dict[str, int]] = {}
        for row in qrels:
            grades.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        # Only the queries this split labels; the queries file holds every split's.
        labeled = [row for row in queries if row["_id"] in grades]
        return (Corpus(ids=list(corpus["_id"]), titles=list(corpus["title"]), texts=list(corpus["text"])),
                Queries(ids=[row["_id"] for row in labeled], texts=[row["text"] for row in labeled]),
                Relevance(grades=grades))
