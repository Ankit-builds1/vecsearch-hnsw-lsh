"""VecSearch - vector similarity search from scratch (Python + NumPy).

Indexes:
    BruteForceIndex  exact search, used as ground truth
    LSHIndex         random hyperplane locality sensitive hashing
    HNSWIndex        hierarchical navigable small world graph
"""

from .base import BaseIndex
from .brute_force import BruteForceIndex
from .heap import MaxHeap, MinHeap
from .hnsw import HNSWIndex
from .lsh import LSHIndex

__all__ = ["BaseIndex", "BruteForceIndex", "HNSWIndex", "LSHIndex", "MaxHeap", "MinHeap"]
__version__ = "0.1.0"
