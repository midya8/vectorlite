"""Recall measurement: approximate results against the exact baseline."""


def recall_at_k(approx_results, exact_results):
    """Fraction of the exact top-k ids present in the approximate result.

    approx_results and exact_results are lists of (id, score) pairs; the
    exact list defines the ground truth and k.
    """
    truth = {vid for vid, _ in exact_results}
    if not truth:
        return 1.0
    found = {vid for vid, _ in approx_results} & truth
    return len(found) / len(truth)


def evaluate_recall(exact_index, approx_index, queries, k=10):
    """Mean recall@k of approx against exact across queries."""
    per_query = []
    for query in queries:
        exact = exact_index.query(query, k=k)
        approx = approx_index.query(query, k=k)
        per_query.append(recall_at_k(approx, exact))
    mean = sum(per_query) / len(per_query) if per_query else 1.0
    return {"queries": len(queries), "k": k, "mean_recall": round(mean, 4), "recalls": per_query}
