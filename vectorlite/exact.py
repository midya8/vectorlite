"""Canonical brute-force exact kNN, score descending then id ascending."""
from .vectors import checked_items, checked_vector, positive_integer, cosine_similarity, l2_distance

class ExactIndex:
    def __init__(self, items, metric="cosine"):
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.items = checked_items(items)
        self.metric = metric
        self.dims = len(self.items[0]["vector"]) if self.items else 0

    def _score(self, query, vector):
        return cosine_similarity(query, vector) if self.metric == "cosine" else -l2_distance(query, vector)

    def query(self, query, k=10, *, exclude_ids=()):
        positive_integer(k, "k", zero=True)
        query = checked_vector(query, self.dims, empty=not self.items)
        excluded = set(exclude_ids)
        scored = [(x["id"], self._score(query,x["vector"])) for x in self.items if x["id"] not in excluded]
        return sorted(scored, key=lambda x: (-x[1],x[0]))[:k]
