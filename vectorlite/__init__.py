"""VectorLite: exact and approximate kNN vector search from scratch."""

from .exact import ExactIndex
from .lsh import LSHIndex
from .buckets import HammingBuckets, BucketLSHIndex
from .vectors import cosine_similarity, l2_distance, load_vectors

__version__ = "0.3.0"
__all__ = ["HammingBuckets", "BucketLSHIndex", "ExactIndex", "LSHIndex", "cosine_similarity", "l2_distance", "load_vectors", "__version__"]
