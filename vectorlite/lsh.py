"""Random-hyperplane signature scan with exact cosine candidate rescoring.

This implementation scans every stored signature, not hash buckets. It can
reduce cosine rescoring; candidate counts are not latency measurements.
"""
import random
from .vectors import checked_items, checked_vector, positive_integer, cosine_similarity, dot, unit

class LSHIndex:
    def __init__(self, items, num_bits=64, seed=17, min_shared_bits=None, *, hyperplanes=None):
        positive_integer(num_bits, "num_bits")
        if not 8 <= num_bits <= 4096:
            raise ValueError("num_bits must be in 8..4096")
        if isinstance(seed,bool) or not isinstance(seed,int):
            raise ValueError("seed must be an integer")
        threshold = (num_bits*3)//4 if min_shared_bits is None else min_shared_bits
        positive_integer(threshold, "min_shared_bits", zero=True)
        if threshold > num_bits:
            raise ValueError("min_shared_bits exceeds num_bits")
        self.items = checked_items(items)
        self.dims = len(self.items[0]["vector"]) if self.items else 0
        self.num_bits, self.seed, self.min_shared_bits = num_bits, seed, threshold
        rng = random.Random(seed)
        planes = hyperplanes if hyperplanes is not None else [
            [rng.gauss(0,1) for _ in range(self.dims)] for _ in range(num_bits)]
        if not isinstance(planes,(list,tuple)) or len(planes) != num_bits:
            raise ValueError("hyperplane count must match num_bits")
        self.hyperplanes = [checked_vector(x,self.dims,empty=not self.items) for x in planes]
        if self.items and any(not any(x) for x in self.hyperplanes):
            raise ValueError("hyperplane must be nonzero")
        self.signatures = [self._signature(x["vector"]) for x in self.items]

    def projections(self, vector):
        vector = checked_vector(vector,self.dims,empty=not self.items)
        if not self.items:
            return [0.0]*self.num_bits
        # Positive scaling preserves halfspace except floating-point boundary ties.
        return [dot(unit(plane),unit(vector)) for plane in self.hyperplanes]

    def _signature(self, vector):
        return sum(1 << i for i, value in enumerate(self.projections(vector)) if value >= 0)

    def query(self, query, k=10, *, exclude_ids=()):
        positive_integer(k,"k",zero=True)
        qsig = self._signature(query)
        excluded = set(exclude_ids)
        candidates = [x for x,s in zip(self.items,self.signatures)
                      if x["id"] not in excluded and self.num_bits-(qsig^s).bit_count() >= self.min_shared_bits]
        scored = [(x["id"],cosine_similarity(query,x["vector"])) for x in candidates]
        return sorted(scored,key=lambda x:(-x[1],x[0]))[:k]
