"""Exact brute-force k-nearest-neighbour search."""

from .vectors import cosine_similarity, l2_distance


class ExactIndex:
    """Brute-force kNN: scans every vector, so results are exact."""

    def __init__(self, items, metric="cosine"):
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.items = list(items)
        self.metric = metric
        self.dims = len(self.items[0]["vector"]) if self.items else 0

    def _score(self, query, vector):
        if len(query) != self.dims:
            raise ValueError("query dim %d != index dim %d" % (len(query), self.dims))
        if self.metric == "cosine":
            return cosine_similarity(query, vector)
        return -l2_distance(query, vector)

    def query(self, query, k=10):
        """Return up to k (id, score) pairs, best first. Ties break on id."""
        scored = [(item["id"], self._score(query, item["vector"])) for item in self.items]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]
