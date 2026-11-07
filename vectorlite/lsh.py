"""Random-hyperplane LSH index for approximate cosine kNN.

Hyperplanes are drawn from a seeded PRNG, so an index built with the same
seed and the same data is bit-for-bit reproducible. Each vector hashes to a
bit signature; candidates are vectors sharing at least `min_shared_bits`
with the query, rescored exactly.
"""

import random

from .vectors import cosine_similarity, dot


class LSHIndex:
    def __init__(self, items, num_bits=64, seed=17, min_shared_bits=None):
        if num_bits < 8:
            raise ValueError("num_bits must be at least 8")
        self.items = list(items)
        self.dims = len(self.items[0]["vector"]) if self.items else 0
        self.num_bits = num_bits
        self.seed = seed
        rng = random.Random(seed)
        self.hyperplanes = [
            [rng.gauss(0.0, 1.0) for _ in range(self.dims)] for _ in range(num_bits)
        ]
        self.min_shared_bits = (
            min_shared_bits if min_shared_bits is not None else (num_bits * 3) // 4
        )
        self.signatures = [self._signature(item["vector"]) for item in self.items]

    def _signature(self, vector):
        if len(vector) != self.dims:
            raise ValueError("vector dim %d != index dim %d" % (len(vector), self.dims))
        bits = 0
        for i, plane in enumerate(self.hyperplanes):
            if dot(plane, vector) >= 0.0:
                bits |= 1 << i
        return bits

    def query(self, query, k=10):
        """Approximate kNN: bucket by signature, rescore candidates exactly."""
        qsig = self._signature(query)
        candidates = []
        for item, sig in zip(self.items, self.signatures):
            shared = self.num_bits - bin(qsig ^ sig).count("1")
            if shared >= self.min_shared_bits:
                candidates.append(item)
        scored = [(item["id"], cosine_similarity(query, item["vector"])) for item in candidates]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]
