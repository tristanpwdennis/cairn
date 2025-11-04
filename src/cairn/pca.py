"""
pca.py
-----

Principal component analysis (PCA)

This module provides modules for computing and plotting
PCA on genomic data.

"""

from __future__ import annotations


import json
import hashlib


# Hashing functions
def hash_params(*args, **kwargs):
    """Helper function to hash analysis parameters."""
    o = {"args": args, "kwargs": kwargs}
    s = json.dumps(o, sort_keys=True).encode()
    h = hashlib.md5(s).hexdigest()
    return h
