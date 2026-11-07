"""Command line interface for VectorLite."""

import argparse
import json
import sys

from .exact import ExactIndex
from .lsh import LSHIndex
from .recall import evaluate_recall
from .vectors import load_vectors


def _cmd_query(args):
    items, dims = load_vectors(args.vectors)
    query = [float(x) for x in args.query.split(",")]
    if len(query) != dims:
        print("query dim %d != index dim %d" % (len(query), dims), file=sys.stderr)
        return 2
    if args.approximate:
        index = LSHIndex(items, num_bits=args.bits, seed=args.seed)
    else:
        index = ExactIndex(items, metric=args.metric)
    for vid, score in index.query(query, k=args.k):
        print("%8.4f  %s" % (score, vid))
    return 0


def _cmd_benchmark(args):
    items, dims = load_vectors(args.vectors)
    exact = ExactIndex(items, metric="cosine")
    approx = LSHIndex(items, num_bits=args.bits, seed=args.seed)
    queries = [item["vector"] for item in items]
    result = evaluate_recall(exact, approx, queries, k=args.k)
    result["bits"] = args.bits
    result["seed"] = args.seed
    result["vectors"] = len(items)
    print(
        "vectors %d, dims %d, bits %d, k %d: mean recall@k %.4f over %d self-queries"
        % (len(items), dims, args.bits, args.k, result["mean_recall"], result["queries"])
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
    if args.min_recall is not None and result["mean_recall"] < args.min_recall:
        print(
            "FAIL mean recall %.4f below gate %.4f" % (result["mean_recall"], args.min_recall),
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="vectorlite",
        description="Exact and approximate kNN vector search from scratch.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("query", help="find the k nearest neighbours of a query vector")
    p.add_argument("--vectors", required=True, help="JSONL vectors ({id, vector} per line)")
    p.add_argument("--query", required=True, help="comma-separated query vector")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    p.add_argument("--approximate", action="store_true", help="use the LSH index")
    p.add_argument("--bits", type=int, default=64, help="LSH signature bits")
    p.add_argument("--seed", type=int, default=17, help="LSH hyperplane seed")
    p.set_defaults(func=_cmd_query)

    p = sub.add_parser("benchmark", help="measure LSH recall against the exact baseline")
    p.add_argument("--vectors", required=True)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--bits", type=int, default=64)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--json", help="write JSON report")
    p.add_argument("--min-recall", type=float, default=None, help="fail below this mean recall")
    p.set_defaults(func=_cmd_benchmark)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)
