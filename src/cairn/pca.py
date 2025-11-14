"""
pca.py
-----

Principal component analysis (PCA)

This module provides modules for computing and plotting
PCA on genomic data.

"""

from __future__ import annotations


import allel
import json
import hashlib
from itertools import cycle
import os
import numpy as np
import pandas as pd
import plotly.express as px

from cairn.snps import load_biallelic_snp_calls

from typing import Union

from yaspin import yaspin


# Hashing function
def hash_params(*args, **kwargs):
    """Helper function to hash analysis parameters."""
    o = {"args": args, "kwargs": kwargs}
    s = json.dumps(o, sort_keys=True).encode()
    h = hashlib.md5(s).hexdigest()
    return h


def run_pca(
    zarr_base_path: str,
    region: str,
    df_samples: pd.DataFrame,
    genotype_var: str = "calldata/GT",
    pos_var: str = "variants/POS",
    cache_path: str = os.getcwd(),
    n_components: int = 10,
    min_minor_ac: Union[int, float] = 1,
    max_missing_an: int = 1,
    thin_offset: int = 0,
    n_snps: int = None,
    sample_query: str = None,
    filter_mask: str = None,
    analysis_name: str = None,
):
    """
    Compute principal component analysis (PCA) on a chromosome or genomic region. Optionally apply random SNP thinning and subsetting by samples. Returns a pd DataFrame of PCA and metadata, and an np.array of the explained variance ratio (evr) of each PC.

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
    cache_path : str
        Path to cache for storing results hashes.
    n_components: int
        Number of principal components to compute.
    is_biallelic: bool
        Toggle whether to select biallelic only sites. Optional. Defaults to True.
    is_segregating: bool
        Toggle whether to select segregating only sites. Optional. Defaults to True.
    min_minor_ac: int / float.
        Filter on minor allele count. If float, will filter on frequency (fraction). Defaults to 1.
    max_missing_an: int
        Min number of missing individuals at a given site. Defaults to 1.
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
    Returns
    -------
    "df_pca", "evr"

    Raises
    ------
    # Errors if the GT and POS arrays are mismatched, if there aren't enough SNPs after filtering to downsample, or if the range queries are insensible.

    Examples
    --------
    >>>
    >>>
    >>>
    """

    # Construct a key to save the results under
    results_key = hash_params(
        contig=region,
        sample_query=sample_query,
        min_minor_ac=min_minor_ac,
        n_snps=n_snps,
        n_components=n_components,
        analysis_name=analysis_name,
        thin_offset=thin_offset,
    )

    # Define paths for results files
    data_path = f"{cache_path}/pca_v1/{results_key}-data.csv"
    evr_path = f"{cache_path}/pca_v1/{results_key}-evr.npy"

    try:
        # Try to load previously generated results
        data = pd.read_csv(data_path)
        evr = np.load(evr_path)
        return data, evr
    except FileNotFoundError:
        # No previous results available, need to run analysis
        print(f"running analysis: {results_key}")

    # Prepare inputs
    with yaspin("Preparing input matrix..."):
        g = load_biallelic_snp_calls(
            zarr_base_path=zarr_base_path,
            region=region,
            df_samples=df_samples,
            genotype_var=genotype_var,
            pos_var=pos_var,
            sample_query=sample_query,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            n_snps=n_snps,
            filter_mask=filter_mask,
        )

        gn = g.to_n_alt()

    # Subset sample df
    if sample_query is not None:
        df = df_samples.query(sample_query)
    else:
        df = df_samples

    # Run PCA
    with yaspin("Computing PCA..."):
        coords, model = allel.pca(
            gn, n_components=n_components, scaler="patterson"
        )  # Run PCA
        df_coords = pd.DataFrame(
            {f"PC{i + 1}": coords[:, i] for i in range(coords.shape[1])}
        )
        data = pd.concat([df.reset_index(), df_coords.reset_index(drop=True)], axis=1)

        # Save output
        evr = model.explained_variance_ratio_
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        data.to_csv(data_path, index=False)
        np.save(evr_path, evr)
        print(f"saved results: {results_key}")

        return data, evr


def plot_pca(
    data: pd.DataFrame,
    evr: np.array,
    i: str = "PC1",
    j: str = "PC2",
    width: int = 500,
    height: int = 600,
    alpha: Union[int, float] = 0.8,
    hover_data: list = None,
    colour: str = None,
    color_discrete_map: dict = None,
    plotpath: str = None,
    **kwargs,  # Additional plotting params if you like for plotly express.
):
    """
    Plot an interactive PCA scatterplot. Optionally supply custom colouring factors and palettes

    Parameters
    ----------
    data : pd.DataFrame
        pd DataFrame, output of run_pca above (data)
    evr : np.array
        np array of explained variance ratio (evr) values. Essentially the % of total variance explained by each PC.
    i : str
        To be plotted on x axis. Defaults to PC1.
    j : str
        To be plotted on y axis. Defaults to PC2.
    width : int
        Plot width in pixels. Defaults to 500.
    height : int
        Plot height in pixels. Defaults to 600.
    hover_data : list
        List of columns to include in point tooltips. Values must be present as column headers in sample metadata. Optional.
    alpha : float or int
        Point opacity. Defaults to 0.9.
    colour : str
        Factor used to colour points. Must be a column in the sample metadata. Optional.
    color_discrete_map : dict
        A dictionary mapping colour factor values to hex colours. Optional.
    plotpath : str
        Output SVG path and name. Optional.
    """

    # Set up colour palette
    # Check for no color.
    if colour is None:
        # Bail out early.
        return None, None, None

    # Throw error if we can't find the data in the metadata.
    if colour not in data.columns:
        raise ValueError(f"{colour!r} is not a known column in the data.")

    # Get factor levels (ever the R programmer) of colour col.
    color_data_unique_values = data[colour].unique()

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
        template="simple_white",
        hover_name="sample_id",
        hover_data=hover_data,
        opacity=alpha,
    )

    # Set up PCA data
    pc_list = [f"PC{i+1}" for i in range(0, len(evr))]
    pc_dict = dict(zip(pc_list, evr.tolist(), strict=True))

    # Apply any user overrides.
    plot_kwargs.update(kwargs)

    # 2D scatter plot.
    fig = px.scatter(data, x=i, y=j, **plot_kwargs)

    # Set up labels
    xlab = f"{i}, {pc_dict[i] * 100:.2f}%"
    ylab = f"{j}, {pc_dict[j] * 100:.2f}%"

    # Add axis titles
    fig.update_layout(xaxis_title=xlab, yaxis_title=ylab)

    fig.update_traces(marker=dict(size=10, opacity=0.7))

    fig.show()

    # Save as SVG if you like
    if plotpath is not None:
        fig.write_image(f"{plotpath}.svg")  # This should work with kaleido
