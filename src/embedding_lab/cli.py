"""The `el` command: run the experiments and print the results."""

import argparse

from embedding_lab.data.nfcorpus import NFCorpus


def _data(args: argparse.Namespace) -> None:
    corpus, queries, relevance = NFCorpus(split=args.split).load()
    labels = [len(relevance.relevant(query_id)) for query_id in queries.ids]
    print(f"NFCorpus {args.split}: {len(corpus)} documents, {len(queries)} queries, "
          f"{sum(labels)} relevance labels ({sum(labels) / len(labels):.1f} per query)")
    print(f"  example query: {queries.texts[0]!r}")
    doc_id, grade = max(relevance.relevant(queries.ids[0]).items(), key=lambda item: item[1])
    print(f"  a grade-{grade} match: {corpus.titles[corpus.ids.index(doc_id)]!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="el", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="what the dataset looks like")
    data.add_argument("--split", default="test", choices=["train", "validation", "test"])
    data.set_defaults(run=_data)
    args = parser.parse_args(argv)
    args.run(args)
