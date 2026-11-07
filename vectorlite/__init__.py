"""VectorLite: exact and approximate kNN vector search from scratch."""

from .exact import ExactIndex
from .lsh import LSHIndex
from .vectors import cosine_similarity, l2_distance, load_vectors

__version__ = "0.1.0"
__all__ = ["ExactIndex", "LSHIndex", "cosine_similarity", "l2_distance", "load_vectors", "__version__"]
