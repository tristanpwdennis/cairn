"""
utils.py
-----

Useful bits and pieces

"""

from __future__ import annotations


import allel
import hashlib
from itertools import cycle
import json
import numba
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns


def thin_array(arr, n_snps: int, thin_offset: int = 0):
    """
    Thin an array to the approximate number of SNPs. Adding a thin offset allows you to to repeat the analysis
    using a different set of SNPs.
    """
    if n_snps is None:
        return arr
    if arr.shape[0] < n_snps:
        raise ValueError("Not enough SNPs.")
    if arr.shape[0] == n_snps:
        return arr
    step = arr.shape[0] // n_snps

    return arr[slice(thin_offset, None, step)]


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
    """
    Get array slice and a parsed genomic region.

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


def prepare_discrete_colour_palette(data: pd.DataFrame, color: str) -> dict:
    """
    Prepare a dictionary of colors based on a dataframe and a column value.

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe containing the values to be plotted / coloured.
    color : str
        The name of the column containing the values to be mapped to colours.
    """

    # Throw error if we can't find the data in the metadata.
    if color not in data.columns:
        raise ValueError(f"{color!r} is not a known column in the data.")

    # Get factor levels (ever the R programmer) of colour col.
    color_data_unique_values = data[color].unique()

    # Now set up color choices.
    if len(color_data_unique_values) <= 12:
        color_discrete_map = list(
            sns.color_palette("Paired", n_colors=len(color_data_unique_values)).as_hex()
        )
    else:
        color_discrete_map = px.colors.qualitative.Alphabet

    # Map values to colors.
    color_discrete_map_prepped = {
        v: c
        for v, c in zip(
            color_data_unique_values, cycle(color_discrete_map), strict=False
        )
    }

    return color_discrete_map_prepped


def hash_params(*args, **kwargs):
    """Helper function to hash analysis parameters."""
    o = {"args": args, "kwargs": kwargs}
    s = json.dumps(o, sort_keys=True).encode()
    h = hashlib.md5(s).hexdigest()
    return h


@numba.njit
def hash_columns(x):
    # Here we want to compute a hash for each column in the
    # input array. However, we assume the input array is in
    # C contiguous order, and therefore we scan the array
    # and perform the computation in this order for more
    # efficient memory access.
    #
    # This function uses the DJBX33A hash function which
    # is much faster than computing Python hashes of
    # bytes, as discovered by Tom White in work on sgkit.
    m = x.shape[0]
    n = x.shape[1]
    out = np.empty(n, dtype=np.int64)
    out[:] = 5381
    for i in range(m):
        for j in range(n):
            v = x[i, j]
            out[j] = out[j] * 33 + v
    return out
