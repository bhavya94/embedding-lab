"""The `el` command: run the experiments and print the results."""

import argparse
import json
from pathlib import Path

from metrics_from_scratch.stats.bootstrap import Bootstrap
import numpy as np

from embedding_lab.data.nfcorpus import NFCorpus, Relevance
from embedding_lab.encode.cache import EmbeddingCache
from embedding_lab.encode.encoder import MINILM, MPNET, Encoder
from embedding_lab.encode.reduce import Reducer
from embedding_lab.finetune.contrastive import (ContrastiveFineTune, false_negative_rate, query_document_ids,
                                                query_document_pairs, title_abstract_pairs)
from embedding_lab.retrieval.ann import FlatIndex, HNSWIndex, overlap_at_k, timed
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


def _ann(args: argparse.Namespace) -> None:
    corpus, queries, relevance = NFCorpus().load()
    encoder = Encoder(normalize=True)
    cache = EmbeddingCache()
    real = cache.encode(encoder, "nfcorpus-corpus", corpus.passages())
    query_vectors = cache.encode(encoder, "nfcorpus-queries", queries.texts)
    rng = np.random.default_rng(args.seed)
    print(f"exact (flat) vs HNSW, {len(queries)} real queries, top 10; past {len(real)} docs the corpus is padded "
          "with random unit vectors; one CPU thread")
    print(f"  {'docs':>7} {'index':<14} {'build s':>8} {'ms/query':>9} {'overlap with exact':>19} {'NDCG@10':>8}")
    results = []
    for size in args.sizes:
        padding = rng.normal(size=(max(size - len(real), 0), real.shape[1])).astype(np.float32)
        padding /= np.linalg.norm(padding, axis=1, keepdims=True)
        docs = np.concatenate([real, padding])[:size]
        doc_ids = (corpus.ids + [f"random-{i}" for i in range(len(padding))])[:size]
        flat, hnsw = FlatIndex(doc_ids, docs), HNSWIndex(doc_ids, docs)
        _, flat_build = timed(flat.build)
        _, hnsw_build = timed(hnsw.build)
        exact, seconds = timed(lambda: flat.search(queries.ids, query_vectors, 10), warm_up=True)
        rows = [("flat (exact)", flat_build, exact, seconds)]
        for ef in args.ef_search:  # one graph per size; ef only changes the search
            hnsw.set_ef_search(ef)
            ranked, seconds = timed(lambda: hnsw.search(queries.ids, query_vectors, 10), warm_up=True)
            rows.append((f"HNSW ef={ef}", hnsw_build, ranked, seconds))
        for label, build, ranked, seconds in rows:
            ndcg = float(np.nanmean(ndcg_at_k(ranked, relevance, 10))) if size == len(real) else None
            results.append({"docs": size, "index": label, "build_s": build,
                            "ms_per_query": 1000 * seconds / len(queries),
                            "overlap": float(np.mean(overlap_at_k(ranked, exact, 10))), "ndcg@10": ndcg})
            shown = f"{ndcg:>8.4f}" if ndcg is not None else f"{'-':>8}"
            print(f"  {size:>7} {label:<14} {build:>8.2f} {results[-1]['ms_per_query']:>9.3f} "
                  f"{results[-1]['overlap']:>19.3f} {shown}")
    _save(args, "ann", {"real_docs": len(real), "rows": results})


def _finetune(args: argparse.Namespace) -> None:
    corpus, queries, relevance = NFCorpus().load()
    _, train_queries, train_relevance = NFCorpus(split="train").load()

    def query_pairs(seed: int) -> list[tuple[str, str]]:
        # NFCorpus repeats a few question texts across splits; train on none of the test texts.
        return query_document_ids(train_queries, train_relevance, seed=seed, exclude_texts=frozenset(queries.texts))

    runs = {"title -> abstract": ("nfcorpus-titles", lambda seed: title_abstract_pairs(corpus)),
            "train queries": ("nfcorpus-train-queries",
                              lambda seed: query_document_pairs(corpus, train_queries, query_pairs(seed)))}
    print(f"MiniLM before and after {args.epochs} epoch(s) of contrastive training, {len(args.seeds)} seeds per run; "
          f"scored on the {len(queries)} test queries, whose texts no training run sees")
    before_ranked, _ = _search(Encoder(), "cosine")
    before = {"ndcg@10": ndcg_at_k(before_ranked, relevance, 10),
              "recall@100": recall_at_k(before_ranked, relevance, 100)}
    labeled = ~np.isnan(before["ndcg@10"])
    rows = [{"setting": "no fine-tune", **_metrics(before_ranked, relevance)}]
    gains = []
    bootstrap = Bootstrap(n_resamples=args.resamples, seed=0)
    for label, (name, make_pairs) in runs.items():
        per_seed = {"ndcg@10": [], "recall@100": [], "mrr": []}
        for seed in args.seeds:
            path = ContrastiveFineTune(name, epochs=args.epochs, seed=seed).run(make_pairs(seed))
            ranked, _ = _search(Encoder(model_name=str(path)), "cosine")
            per_seed["ndcg@10"].append(ndcg_at_k(ranked, relevance, 10))
            per_seed["recall@100"].append(recall_at_k(ranked, relevance, 100))
            per_seed["mrr"].append(mrr(ranked, relevance))
        # Average each query's score over the seeds, so the interval reflects queries, not one lucky run.
        after = {metric: np.mean(values, axis=0) for metric, values in per_seed.items()}
        rows.append({"setting": label, **{metric: float(np.nanmean(v)) for metric, v in after.items()}})
        for metric in ["ndcg@10", "recall@100"]:
            gap = bootstrap.interval(lambda b, a: float(np.mean(a - b)), before[metric][labeled],
                                     after[metric][labeled])
            gains.append({"setting": label, "metric": metric, "estimate": gap.estimate, "low": gap.low,
                          "high": gap.high,
                          "per_seed": [float(np.nanmean(v - before[metric])) for v in per_seed[metric]]})
    print(HEADER)
    for row in rows:
        print(_row(row["setting"], row))
    print("gain over no fine-tune, seed-averaged per query, paired 95% bootstrap CI")
    for gain in gains:
        seeds = ", ".join(f"{value:+.4f}" for value in gain["per_seed"])
        print(f"  {gain['setting']:<18} {gain['metric']:<10} {gain['estimate']:+.4f} [{gain['low']:+.4f}, "
              f"{gain['high']:+.4f}]  per seed: {seeds}")
    rate = false_negative_rate(query_pairs(args.seeds[0]), train_relevance, seed=args.seeds[0])
    print(f"train-query batches of 32: {rate:.0%} of queries have a labeled-relevant doc among their negatives")
    _save(args, "finetune", {"seeds": args.seeds, "rows": rows, "gains": gains, "false_negative_rate": rate})


def _dims(args: argparse.Namespace) -> None:
    corpus, queries, relevance = NFCorpus().load()
    cache = EmbeddingCache()
    print("NDCG@10 on NFCorpus test as vectors shrink; SVD and PCA are fit on the corpus vectors only")
    print(f"  {'model':<10} {'dims':>5} {'method':<9} {'NDCG@10':>8} {'bytes/doc':>10}")
    rows = []
    for name, model in [("MiniLM", MINILM), ("mpnet", MPNET)]:
        encoder = Encoder(model_name=model)
        docs = cache.encode(encoder, "nfcorpus-corpus", corpus.passages())
        query_vectors = cache.encode(encoder, "nfcorpus-queries", queries.texts)
        full = docs.shape[1]
        for dims in [d for d in args.dims if d < full] + [full]:
            for method in (["svd", "pca", "truncate"] if dims < full else ["full", "pca"]):
                if method == "full":
                    d, q = docs, query_vectors
                else:
                    reducer = Reducer(dims, method).fit(docs)
                    d, q = reducer.transform(docs), reducer.transform(query_vectors)
                ranked = ExactSearch(corpus.ids, d).search(queries.ids, q, 10)
                rows.append({"model": name, "dims": dims, "method": method,
                             "ndcg@10": float(np.nanmean(ndcg_at_k(ranked, relevance, 10))), "bytes": 4 * dims})
                print(f"  {name:<10} {dims:>5} {method:<9} {rows[-1]['ndcg@10']:>8.4f} {4 * dims:>10}")
    _save(args, "dims", {"rows": rows})


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

    ann = commands.add_parser("ann", help="FAISS HNSW vs exact search: latency and recall as the corpus grows")
    ann.add_argument("--sizes", type=int, nargs="+", default=[3_633, 50_000, 200_000])
    ann.add_argument("--ef-search", type=int, nargs="+", default=[16, 64, 256])
    ann.add_argument("--seed", type=int, default=0)
    ann.set_defaults(run=_ann)

    finetune = commands.add_parser("finetune", help="NDCG before and after an unsupervised in-domain fine-tune")
    finetune.add_argument("--epochs", type=int, default=1)
    finetune.add_argument("--resamples", type=int, default=2_000)
    finetune.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    finetune.set_defaults(run=_finetune)

    dims = commands.add_parser("dims", help="retrieval quality vs vector size: PCA vs truncation, 384 vs 768")
    dims.add_argument("--dims", type=int, nargs="+", default=[32, 64, 128, 256])
    dims.set_defaults(run=_dims)

    args = parser.parse_args(argv)
    args.run(args)
