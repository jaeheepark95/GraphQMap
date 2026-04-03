"""
kmw1: canonical-hardware v1.4 branch.

This package removes the learned reindexer and replaces it with a fixed,
deterministic hardware canonicalizer. The circuit stays in native logical order;
the hardware is canonicalized once in preprocessing; the mapper predicts in the
canonical hardware frame; logits are decoded back to native hardware order
before the authoritative v1.4.1 execution-surrogate loss is computed.
"""

__all__ = [
    "utils",
]
