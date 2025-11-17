"""
pca.py
-----

Neighbour joining trees (NJT)

This module provides modules for computing and plotting
NJTs on genomic data.

"""
import allel
import anjl
import hashlib
from itertools import cycle
import json
import os
import numpy as np
import numba
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import squareform  # type: ignore

from cairn.snps import load_genotype_array

from typing import Union

from yaspin import yaspin

# Hashing function
def hash_params(*args, **kwargs):
    """Helper function to hash analysis parameters."""
    o = {"args": args, "kwargs": kwargs}
    s = json.dumps(o, sort_keys=True).encode()
    h = hashlib.md5(s).hexdigest()
    return h

@numba.njit
def square_to_condensed(i, j, n):
    """Convert distance matrix coordinates from square form (i, j) to condensed form."""

    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return n * j - j * (j + 1) // 2 + i - 1 - j

@numba.njit
def genotype_cityblock(x, y):
    n_sites = x.shape[0]
    distance = np.float32(0)

    # Loop over sites.
    for i in range(n_sites):
        # Compute cityblock distance (absolute difference).
        d = np.fabs(x[i] - y[i])

        # Accumulate distance for the current pair.
        distance += d

    return distance

@numba.njit(parallel=True)
def biallelic_diplotype_pdist(X):
    n_samples = X.shape[0]
    n_pairs = (n_samples * (n_samples - 1)) // 2
    out = np.zeros(n_pairs, dtype=np.float32)

    # Loop over samples, first in pair.
    for i in range(n_samples):
        x = X[i, :]

        # Loop over observations again, second in pair.
        for j in numba.prange(i + 1, n_samples):
            y = X[j, :]

            # Compute distance for the current pair.
            d = genotype_cityblock(x, y)

            # Store result for the current pair.
            k = square_to_condensed(i, j, n_samples)
            out[k] = d

    return out


def infer_njt(
    zarr_base_path: str,
    region: str,
    df_samples: pd.DataFrame,
    genotype_var: str = "calldata/GT",
    pos_var: str = "variants/POS",
    cache_path: str = os.getcwd(),
    thin_offset: int = 0,
    n_snps: int = None,
    sample_query: str = None,
    filter_mask: str = None,
    analysis_name: str = None,
    overwrite: bool = False,
):
    """
    Compute a neighbour joining tree (NJT) on a chromosome or genomic region. Optionally apply random SNP thinning and subsetting by samples. Returns an numpy array of the inferred tree.

    Parameters
    ----------
    zarr_base_path : str
        Path to input zarr store.
    region : str
        Genomic region to select. Can be a whole contig (e.g. 2RL), or a region (e.g. 2RL:1000-2000). Regions must be in the format <contig>:<start>-<end>.
    df_samples : pandas DataFrame
        Sample metadata.
    genotype_var : str
        Path to the genotype data within the zarr store. Defaults to 'calldata/GT'.
    pos_var : str
        Path to the variant position within the zarr store. Defaults to 'variants/POS'.
    cache_path : str
        Path to cache for storing results hashes.
    n_components: int
        Number of principal components to compute.
    thin_offset: int
        Starting index for SNP thinning. Change this to repeat the analysis using a different set of SNPs. Optional.
    sample_query : str
        Pandas-style query statement to subset samples. Optional.
    n_snps : int
        Randomly downsample variants to this value. Optional.
    filter_mask : str
        Path of a boolean filter mask variable in the zarr store. Optional.
    analysis_name : str
        Name of analysis if you want to save separate analysis caches. Optional.
    overwrite : bool
        Overwrite existing hash (e.g. if underlying data have changed). Optional.
        
    Returns
    -------
    "njt"

    Raises
    ------
    # Errors if the GT and POS arrays are mismatched, if there aren't enough SNPs after filtering to downsample, or if the range queries are insensible.

    Examples
    --------
    >>>
    >>>
    >>>
    """

    # Construct a key to save the results under.
    results_key = hash_params(
        contig=region,
        sample_query=sample_query,
        n_snps=n_snps,
        analysis_name=analysis_name,
        thin_offset=thin_offset,
    )

    # Define paths for distance matrices.
    njt_path = f"{cache_path}/njt_v1/{results_key}-njt.npy"

    # Subset sample df
    if sample_query is not None:
        df = df_samples.query(sample_query)
    else:
        df = df_samples

    if overwrite is False:
        try:
            njt = np.load(njt_path)
            return njt, df
        except FileNotFoundError:
            pass  # fall through to running analysis

    # Prepare inputs.
    with yaspin("Preparing inputs..."):
        g = load_genotype_array(
            zarr_base_path=zarr_base_path,
            region=region,
            df_samples=df_samples,
            genotype_var=genotype_var,
            pos_var=pos_var,
            n_snps = n_snps,
            thin_offset = thin_offset,
            sample_query=sample_query,
            filter_mask=filter_mask,
        )


    # Prepare distance matrix
    gn = g.to_n_alt()

    X = np.ascontiguousarray(gn.T)

    dist = biallelic_diplotype_pdist(X)

    #Coerce to square matrix.
    D = squareform(dist)
    
    # Build the NJ tree.
    progress_options = dict(desc="Construct neighbour-joining tree", leave=False)
    
    njt = anjl.canonical_nj(
                    D=D, progress_options=progress_options
    )

    # Write output to cache
    os.makedirs(os.path.dirname(njt_path), exist_ok=True)
    np.save(njt_path, njt)
    print(f"saved results: {results_key}")

    return njt, df


def plot_njt(
        njt : np.array,
        metadata : pd.DataFrame,
        hover_data : list = None,
        colour : str = None,
        width: int = 800,
        height: int = 600,
        colour_map : dict = None,
        output_path : str = None,
        color_discrete_map : dict = None,
    ):

    # TO-DO: abstract common plotting functionality into utils
    
    # Set up colour palette
    # Check for no color.
    if colour is None:
        # Bail out early.
        return None, None, None

    # Throw error if we can't find the data in the metadata.
    if colour not in metadata.columns:
        raise ValueError(f"{colour!r} is not a known column in the data.")

    # Get factor levels (ever the R programmer) of colour col.
    color_data_unique_values = metadata[colour].unique()

    # Now set up color choices.
    if color_discrete_map is None:
        if len(color_data_unique_values) <= 10:
            color_discrete_sequence = px.colors.qualitative.Plotly
        else:
            color_discrete_sequence = px.colors.qualitative.Alphabet

    # Map values to colors.
    color_discrete_map_prepped = {
        v: c
        for v, c in zip(
            color_data_unique_values, cycle(color_discrete_sequence), strict=False
        )
    }

    # Set up plot kwargs
    plot_kwargs = dict(
        width=width,
        height=height,
        color=colour,
        color_discrete_map=color_discrete_map_prepped,
        hover_data=hover_data,
        #opacity=alpha,
    )
    
    # Plot
    nj_2 = anjl.plot(
                Z=Z,
                leaf_data=metadata,
                **plot_kwargs
                )
    # Save as SVG if you like
    if output_path is not None:
        fig.write_image(f"{output_path}.svg")  # This should work with kaleido
        
    nj_2.show()
    

    