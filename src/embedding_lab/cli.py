"""The `el` command: run the experiments and print the results."""

import argparse
import json
from pathlib import Path

import numpy as np

from embedding_lab.data.nfcorpus import NFCorpus, Relevance
from embedding_lab.encode.cache import EmbeddingCache
from embedding_lab.encode.encoder import MINILM, Encoder
from embedding_lab.retrieval.metrics import Ranked, mrr, ndcg_at_k, recall_at_k
from embedding_lab.retrieval.search import ExactSearch

HEADER = f"  {'setting':<22} {'NDCG@10':>8} {'Recall@100':>11} {'MRR':>7}"


def _metrics(ranked: Ranked, relevance: Relevance) -> dict:
    """Mean of each metric over the queries that have labels."""
    return {"ndcg@10": float(np.nanmean(ndcg_at_k(ranked, relevance, 10))),
            "recall@100": float(np.nanmean(recall_at_k(ranked, relevance, 100))),
            "mrr": float(np.nanmean(mrr(ranked, relevance)))}


def _row(label: str, metrics: dict) -> str:
    return f"  {label:<22} {metrics['ndcg@10']:>8.4f} {metrics['recall@100']:>11.4f} {metrics['mrr']:>7.4f}"


def _save(args: argparse.Namespace, name: str, results: dict) -> None:
    """With `--results-dir`, also write the command's numbers as JSON; the results page is built from them."""
    if args.results_dir:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        (args.results_dir / f"{name}.json").write_text(json.dumps(results, indent=1) + "\n")


def _search(encoder: Encoder, similarity: str, k: int = 100):
    """Encode NFCorpus with `encoder` (cached), run exact search, and return the ranking with the labels."""
    corpus, queries, relevance = NFCorpus().load()
    cache = EmbeddingCache()
    docs = cache.encode(encoder, "nfcorpus-corpus", corpus.passages())
    query_vectors = cache.encode(encoder, "nfcorpus-queries", queries.texts)
    return ExactSearch(corpus.ids, docs, similarity).search(queries.ids, query_vectors, k), relevance


def _data(args: argparse.Namespace) -> None:
    corpus, queries, relevance = NFCorpus(split=args.split).load()
    labels = [len(relevance.relevant(query_id)) for query_id in queries.ids]
    print(f"NFCorpus {args.split}: {len(corpus)} documents, {len(queries)} queries, "
          f"{sum(labels)} relevance labels ({sum(labels) / len(labels):.1f} per query)")
    print(f"  example query: {queries.texts[0]!r}")
    doc_id, grade = max(relevance.relevant(queries.ids[0]).items(), key=lambda item: item[1])
    print(f"  a grade-{grade} match: {corpus.titles[corpus.ids.index(doc_id)]!r}")


def _pooling(args: argparse.Namespace) -> None:
    print(f"{args.model}, unit-length vectors, exact cosine search on NFCorpus test")
    print(HEADER)
    rows = []
    for pooling in ["mean", "cls"]:
        ranked, relevance = _search(Encoder(model_name=args.model, pooling=pooling, normalize=True), "cosine")
        rows.append({"setting": f"{pooling} pooling", **_metrics(ranked, relevance)})
        print(_row(rows[-1]["setting"], rows[-1]))
    _save(args, "pooling", {"model": args.model, "rows": rows})


def _similarity(args: argparse.Namespace) -> None:
    print(f"{args.model}, mean pooling, raw (not unit-length) vectors, exact search on NFCorpus test")
    print(HEADER)
    rows = []
    for similarity in ["cosine", "dot"]:
        ranked, relevance = _search(Encoder(model_name=args.model, pooling="mean", normalize=False), similarity)
        rows.append({"setting": f"{similarity} similarity", **_metrics(ranked, relevance)})
        print(_row(rows[-1]["setting"], rows[-1]))
    _save(args, "similarity", {"model": args.model, "rows": rows})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="el", description=__doc__)
    parser.add_argument("--results-dir", type=Path, help="also save each command's numbers here as JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="what the dataset looks like")
    data.add_argument("--split", default="test", choices=["train", "validation", "test"])
    data.set_defaults(run=_data)

    pooling = commands.add_parser("pooling", help="CLS vs mean pooling on the same model")
    pooling.add_argument("--model", default=MINILM)
    pooling.set_defaults(run=_pooling)

    similarity = commands.add_parser("similarity", help="cosine vs dot product on the same vectors")
    similarity.add_argument("--model", default=MINILM)
    similarity.set_defaults(run=_similarity)

    args = parser.parse_args(argv)
    args.run(args)
