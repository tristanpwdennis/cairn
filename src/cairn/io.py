"""
io.py
-----

input/output utilities

This module provides convenient, typed, and extensible I/O helpers
for extracting and subsetting genotype data from Zarr arrays.

"""

from __future__ import annotations

from typing import Union

import allel
import pandas as pd
import numpy as np
import zarr


# To-Do
# - Add region support
# - Add gcs support


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


def _parse_region(region_str: str) -> tuple:
    """
    Parse a genomic region string of the form 'chrom', or 'chrom:start-end'.

    Examples
    --------
    'CM023248' -> ('CM023248', None, None)
    'CM023248:1000' -> ('CM023248', 1000, None)
    'CM023248:1000-2000' -> ('CM023248', 1000, 2000)
    """
    # Strip whitespace
    region_str = region_str.strip()
    chrom = region_str
    start = end = None

    # If contains colon, split into chrom and coords and strip whitespace
    if ":" in region_str:
        chrom_part, coords = region_str.split(":", 1)
        chrom = chrom_part.strip()

        # Cmomplain if the coords aren't formatted correctly
        if "-" not in coords:
            raise ValueError(
                f"Region must include both start and end positions, e.g. '2RL:1000-2000', got '{region_str}'"
            )

        # Strip whitespace again
        start_str, end_str = coords.split("-", 1)

        # Remove any commas from numbers. If the numbers aren't integers, complain.
        try:
            start = int(start_str.replace(",", "").strip())
            end = int(end_str.replace(",", "").strip())
        except ValueError as err:
            raise ValueError(
                f"Start and end positions must be integers, got '{coords}'"
            ) from err

        # Make sure that the start and end are oriented correctly
        if start >= end:
            raise ValueError(
                f"Start position must be less than end position in '{region_str}'"
            )

    return (chrom, start, end)  # Return parsed region


def _locate_region(region: tuple, pos: np.ndarray) -> slice:
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
        loc_region = pos_idx.locate_range(
            region[1], region[2]
        )  # use start and end (1st and 2nd elements of the region tuple)
    except KeyError:
        # There are no data within the requested region, return a zero-length slice.
        loc_region = slice(0, 0)
    return loc_region


def load_genotype_array(
    zarr_base_path: str,
    region: str,
    df_samples: pd.DataFrame,
    genotype_var="calldata/GT",
    pos_var="variants/POS",
    thin_offset: int = 0,
    n_snps: int = None,
    sample_query: str = None,
    filter_mask: str = None,
) -> allel.GenotypeArray:
    """
    Load of genotypes from a zarr store. Optionally apply queries, randomly downsample, or select a range. Returns a scikit-allel GenotypeArray.

    Parameters
    ----------
    zarr_base_path : str
        Path to input zarr store.
    region : str
        Genomic region to select. Can be a whole contig (e.g. 2RL), or a region (e.g. 2RL:1000-2000). Regions must be in the format <contig>:<start>-<end>.
    df_samples : pandas DataFrame
        Sample metadata.
    genotype_var : str
        Path to the genotype data within the zarr store.
    pos_var : str
        Path to the variant position within the zarr store. Defaults to 'variants/POS'.
    thin_offset: int
        Starting index for SNP thinning. Change this to repeat the analysis using a different set of SNPs.
    sample_query : str, optional
        Pandas-style query statement to subset samples. Optional.
    n_snps : int, optional
        Randomly downsample variants to this value. Optional.
    filter_mask : str, optional
        Path of a boolean filter mask variable in the zarr store. Optional.
    Returns
    -------
    allel.GenotypeArray (n_sites, n_samples, n_ploidy)

    Raises
    ------
    # Errors if the GT and POS arrays are mismatched, if there aren't enough SNPs after filtering to downsample, or if the range queries are insensible.

    Examples
    --------
    >>>
    >>>
    >>>
    """
    # Parse region
    genome_location = _parse_region(region)

    # Open zarr
    z = zarr.open(zarr_base_path.format(contig=genome_location[0]))

    # Load genotype data
    gt = allel.GenotypeChunkedArray(z[f"{genotype_var}"])

    # Subset by query, if applicable
    if sample_query is not None:
        bool_query = np.array(df_samples.eval(sample_query))
        gt = gt.compress(bool_query, axis=1)

    # Subset to range if coordinates are passed in the region
    if region[2] is not None and region[1] is not None:
        pos = z[f"{pos_var}"]
        posx = _locate_region(region=genome_location, pos=pos)
        if gt[posx].shape[0] != pos[posx].shape[0]:
            raise Exception("Conflicting GT and POS array lengths")
        gt = gt[posx]

    # Apply filter mask
    if filter_mask is not None:
        flt = z[f"{filter_mask}"]  # Load mask
        if posx is not None:  # Filter by position if applicable
            flt = flt[posx]
        gt = gt.compress(flt, axis=0)

    # Thin
    if n_snps is not None:
        # Try to meet target number of SNPs.
        if gt.shape[0] > (n_snps):
            # Apply thinning.
            thin_step = gt.shape[0] // n_snps
            loc_thin = slice(thin_offset, None, thin_step)
            gt = gt[loc_thin]

        elif gt.shape[0] < n_snps:
            raise ValueError("Not enough SNPs.")

    return gt
