"""
utils.py
-----

Useful bits and pieces

"""

from __future__ import annotations

from typing import Union

import allel
import numpy as np


def select_random_elements_sorted(
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


def parse_region(region_str: str) -> tuple:
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


def locate_region(region: tuple, pos: np.ndarray) -> slice:
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
