"""Domain adaptation by contrastive fine-tuning, from two kinds of (anchor, positive) pairs.

1. Title -> abstract: no labels needed, only the corpus. A paper's title should match its own abstract.
2. Train query -> relevant document: NFCorpus's train split, whose queries never appear in the test split.

Each batch holds pairs; for an anchor, its own positive is the target and every other positive in the batch is
a negative (MultipleNegativesRankingLoss, the in-batch negatives trick).
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss

from embedding_lab.data.nfcorpus import Corpus, Queries, Relevance
from embedding_lab.encode.encoder import MINILM, default_device


def title_abstract_pairs(corpus: Corpus) -> Dataset:
    """One (anchor, positive) row per document that has both a title and an abstract."""
    rows = [(title, text) for title, text in zip(corpus.titles, corpus.texts) if title and text]
    return Dataset.from_dict({"anchor": [title for title, _ in rows], "positive": [text for _, text in rows]})


def query_document_ids(queries: Queries, relevance: Relevance, per_query: int = 4, seed: int = 0,
                       exclude_texts: frozenset[str] = frozenset()) -> list[tuple[str, str]]:
    """Up to `per_query` (query id, relevant doc id) pairs per query, sampled so common queries don't dominate.

    Queries whose text is in `exclude_texts` are skipped: NFCorpus repeats a few question texts across splits.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for query_id, text in zip(queries.ids, queries.texts):
        if text in exclude_texts:
            continue
        relevant = sorted(relevance.relevant(query_id))
        pairs += [(query_id, str(doc_id)) for doc_id in rng.permutation(relevant)[:per_query]]
    return pairs


def query_document_pairs(corpus: Corpus, queries: Queries, id_pairs: list[tuple[str, str]]) -> Dataset:
    """The texts for (query id, doc id) pairs: the query text as anchor, the doc passage as positive."""
    query_text = dict(zip(queries.ids, queries.texts))
    passage = dict(zip(corpus.ids, corpus.passages()))
    return Dataset.from_dict({"anchor": [query_text[q] for q, _ in id_pairs],
                              "positive": [passage[d] for _, d in id_pairs]})


def false_negative_rate(id_pairs: list[tuple[str, str]], relevance: Relevance, batch_size: int = 32,
                        batches: int = 300, seed: int = 0) -> float:
    """The share of queries, in training-like batches, that have a labeled-relevant document among their negatives.

    In-batch negatives assume every other positive in the batch is wrong for this query. When queries share
    many relevant documents, that assumption fails and training pushes relevant documents away. Batches are
    filled the way the `no_duplicates` sampler fills them: no query and no document appears twice in one batch.
    """
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(batches):
        batch, seen = [], set()
        for i in rng.permutation(len(id_pairs)):
            query_id, doc_id = id_pairs[i]
            if query_id in seen or doc_id in seen:
                continue
            batch.append(id_pairs[i])
            seen |= {query_id, doc_id}
            if len(batch) == batch_size:
                break
        for query_id, doc_id in batch:
            relevant = relevance.relevant(query_id)
            hits.append(any(other in relevant for _, other in batch if other != doc_id))
    return float(np.mean(hits))


@dataclass(frozen=True)
class ContrastiveFineTune:
    """A recipe for one fine-tuning run on `pairs`, named `name`. `run` saves the model and returns its directory.

    A run with the same settings and the same pairs reuses the saved model instead of training again, so commands
    that compare before and after stay fast. Changed pairs get a new directory. Delete a directory to retrain it.
    """

    name: str
    base_model: str = MINILM
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 2e-5
    # Tokens per text while training. At 256, MPS training at batch 32 ran an 8 GB Mac out of memory; 128 fits.
    max_length: int = 128
    seed: int = 0
    device: str = default_device()
    directory: Path = Path(".cache") / "models"

    def output(self, pairs: Dataset) -> Path:
        name = self.base_model.split("/")[-1]
        digest = hashlib.sha256(json.dumps([list(pairs["anchor"]), list(pairs["positive"])]).encode()).hexdigest()[:8]
        return self.directory / (f"{name}-{self.name}-e{self.epochs}-b{self.batch_size}-lr{self.learning_rate}"
                                 f"-len{self.max_length}-s{self.seed}-{digest}")

    def run(self, pairs: Dataset) -> Path:
        output = self.output(pairs)
        if (output / "config.json").exists():
            return output
        model = SentenceTransformer(self.base_model, device=self.device)
        model.max_seq_length = self.max_length
        arguments = SentenceTransformerTrainingArguments(
            output_dir=str(output / "checkpoints"), num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size, learning_rate=self.learning_rate,
            warmup_steps=0.1,  # below 1, a share of all steps: warm up over the first 10%
            use_cpu=self.device == "cpu", seed=self.seed, save_strategy="no", logging_steps=20, report_to="none",
            # In-batch negatives are wrong if a batch holds two copies of one title.
            batch_sampler="no_duplicates")
        SentenceTransformerTrainer(model=model, args=arguments, train_dataset=pairs,
                                   loss=MultipleNegativesRankingLoss(model)).train()
        model.save(str(output))
        return output
