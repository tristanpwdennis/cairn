"""
io.py
-----

input/output utilities

This module provides convenient, typed, and extensible I/O helpers
for extracting and subsetting genotype data from Zarr arrays.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import allel
import pandas as pd
import numpy as np
import xarray as xr
import zarr


import numpy as np
from typing import Union



# To-Do
# - Add region support


def locate_region(region: tuple,  # Parse this
                  pos: np.ndarray) -> slice:
    """Get array slice and a parsed genomic region.

    Parameters
    ----------
    region : Region
        The region to locate.
    pos : array-like
        Positions to be searched.

    Returns
    -------
    loc_region : slice

    """
    pos_idx = allel.SortedIndex(pos)
    try:
        loc_region = pos_idx.locate_range(region.start, region.end)
    except KeyError:
        # There are no data within the requested region, return a zero-length slice.
        loc_region = slice(0, 0)
    return loc_region

# TODO - move this to util?
def _select_random_elements_sorted(
    arr: Union[np.ndarray],
    n: int,
    replace: bool = False,
    seed: int | None = None,
    return_indices: bool = False,
):
    """
    Select random rows from a 2D array (or xarray), returned in sorted order.

    Parameters
    ----------
    arr : array-like of shape (n_rows, n_features)
        Input array or matrix from which to sample rows.
    n : int
        Number of rows to select.
    replace : bool, optional
        Whether to sample with replacement. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    return_indices : bool, optional
        If True, also return the selected indices. Default is False.
    """

    rng = np.random.default_rng(seed)
    n_rows = arr.shape[0]

    if not replace and n > n_rows:
        raise ValueError(
            f"Cannot sample {n} rows without replacement from {n_rows} total."
        )

    indices = np.sort(rng.choice(n_rows, size=n, replace=replace))

    sampled = arr[indices]
    return (sampled, indices) if return_indices else sampled

# Define helper functions
def load_genotype_array(
    zarr_base_path : str, 
    contig: str, 
    df_samples: pd.DataFrame, 
    range: str = None,
    sample_query: str = None, 
    n_snps: int = None
    ):

    """
    Load of genotypes from a zarr store. Optionally apply queries, randomly downsample, or select a range. Returns a scikit-allel GenotypeArray.

    Parameters
    ----------
    zarr_base_path : str
        Path to input zarr store.
    contig : str
        Contig or chromosome identifier in the reference genome.
    df_samples : pandas DataFrame
        Sample metadata.
    range : str, optional
        Genomic range (start:end) to apply to the data. Optional.
    sample_query : str, optional
        Pandas-style query statement to subset samples. Optional.
    n_snps : int, optional
        Randomly downsample variants to this value. Optional.

    Returns
    -------
    allel.GenotypeArray (n_sites, n_samples, n_ploidy)
    
    Raises
    ------
    # Error if the data aren't found or the queries are insensible.

    Examples
    --------
    >>> 
    >>>
    >>>
    """
    
    # Open zarr
    z = zarr.open(zarr_base_path.format(contig=contig))
    
    # Variant-level mask: punctulatus_group_filter_pass
    filter_mask = z[f"variants/filter_pass"][:]
    
    # Apply combined variant mask
    gt = allel.GenotypeChunkedArray(z[f"calldata/GT"])
    gt = gt.compress(filter_mask, axis=0)    # Filter variants
    
    # If an additional mask is supplied to subset the data from the finished metadata, apply, else return all samples
    if sample_query is not None:
        bool_query = np.array(df_samples.eval(sample_query))
        gt = gt.compress(bool_query, axis=1)
    if n_snps is not None:
            gt = _select_random_elements_sorted(gt, n_snps)

    return gt