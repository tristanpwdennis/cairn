"""
snps.py
-----

snp loading utilities

This module provides convenient, typed, and extensible I/O helpers
for extracting and subsetting genotype data from Zarr arrays.

"""

from __future__ import annotations

from typing import Union

import allel
import pandas as pd
import numpy as np
import zarr

from yaspin import yaspin

from cairn.utils import locate_region, parse_region


# To-Do
# - Add gcs support


@yaspin(text="Loading genotypes...")
def load_genotype_array(
    zarr_base_path: str,
    region: str,
    df_samples: pd.DataFrame,
    genotype_var: str = "calldata/GT",
    pos_var: str = "variants/POS",
    thin_offset: int = 0,
    n_snps: int = None,
    sample_query: str = None,
    filter_mask: str = None,
) -> allel.GenotypeArray:
    """
    Load genotypes from a zarr store. Optionally apply queries, randomly downsample, or select a range. Returns a scikit-allel GenotypeArray.

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
    genome_location = parse_region(region)

    # Open zarr
    z = zarr.open(zarr_base_path.format(contig=genome_location[0]))

    # Load genotype data
    gt = allel.GenotypeArray(z[f"{genotype_var}"])

    # Subset by query, if applicable
    if sample_query is not None:
        bool_query = np.array(df_samples.eval(sample_query))
        gt = gt.compress(bool_query, axis=1)

    # Subset to range if coordinates are passed in the region
    if region[2] is not None and region[1] is not None:
        pos = z[f"{pos_var}"]
        posx = locate_region(region=genome_location, pos=pos)
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


@yaspin(text="Computing allele counts...")
def compute_ac(
    zarr_base_path: str,
    region: str,
    df_samples: pd.DataFrame,
    genotype_var: str = "calldata/GT",
    pos_var: str = "variants/POS",
    is_biallelic: bool = True,
    is_segregating: bool = True,
    min_minor_ac: Union[int, float] = 1,
    thin_offset: int = 0,
    n_snps: int = None,
    sample_query: str = None,
    filter_mask: str = None,
) -> allel.AlleleCountsArray:
    """
    Compute allele counts. Subset by region. Optionally apply queries, randomly downsample, or select a range. Returns a scikit-allel AlleleCountsArray.

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
    is_biallelic: bool
        Toggle whether to select biallelic only sites. Optional. Defaults to True.
    is_segregating: bool
        Toggle whether to select segregating only sites. Optional. Defaults to True.
    min_minor_ac: int / float.
        Filter on minor allele count. If float, will filter on frequency (fraction). Defaults to 1. Optional.
    thin_offset: int
        Starting index for SNP thinning. Change this to repeat the analysis using a different set of SNPs. Optional.
    sample_query : str
        Pandas-style query statement to subset samples. Optional.
    n_snps : int
        Randomly downsample variants to this value. Optional.
    filter_mask : str
        Path of a boolean filter mask variable in the zarr store. Optional.
    Returns
    -------
    allel.AlleleCountsArray (n_sites, n_ploidy)

    Raises
    ------
    # Errors if the GT and POS arrays are mismatched, if there aren't enough SNPs after filtering to downsample, or if the range queries are insensible.

    Examples
    --------
    >>>
    >>>
    >>>
    """

    g = load_genotype_array(
        zarr_base_path=zarr_base_path,
        region=region,
        df_samples=df_samples,
        genotype_var=genotype_var,
        pos_var=pos_var,
        filter_mask=filter_mask,
        thin_offset=thin_offset,
        sample_query=sample_query,
    )

    ac = g.count_alleles()

    mask = None  # Start with initial mask of none, then build filter mask as optional conditions are applied.

    # Apply biallelic filter
    if is_biallelic is not None:
        biallelic_mask = ac.is_biallelic()
        mask = biallelic_mask if mask is None else mask & biallelic_mask

    # Apply segregating filter
    if is_segregating is not None:
        segregating_mask = ac.is_segregating()
        mask = segregating_mask if mask is None else mask & segregating_mask

    # Apply minor allele count filter
    if min_minor_ac is not None:
        an = ac.sum(axis=1)
        # Apply minor allele count condition.
        ac_minor = ac[:, 1:].sum(axis=1)
        if isinstance(min_minor_ac, float):
            ac_minor_frac = ac_minor / an
            loc_minor_mask = ac_minor_frac >= min_minor_ac
        else:
            loc_minor_mask = ac_minor >= min_minor_ac
        mask = loc_minor_mask if mask is None else mask & loc_minor_mask

    # Apply all filters at once
    if mask is not None:
        gt = g.compress(mask)

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

    return gt.count_alleles()


def load_biallelic_snp_calls(
    zarr_base_path: str,
    region: str,
    df_samples: pd.DataFrame,
    genotype_var: str = "calldata/GT",
    pos_var: str = "variants/POS",
    thin_offset: int = 0,
    min_minor_ac: int = 1,
    max_missing_an: int = 1,
    n_snps: int = None,
    sample_query: str = None,
    filter_mask: str = None,
):
    """ """

    # Set up genotypes
    gt = load_genotype_array(
        zarr_base_path=zarr_base_path,
        region=region,
        df_samples=df_samples,
        genotype_var=genotype_var,
        pos_var=pos_var,
        sample_query=sample_query,
        filter_mask=filter_mask,
    )

    ac = gt.count_alleles()

    mask = None  # Start with initial mask of none, then build filter mask as optional conditions are applied.

    # Apply biallelic filter
    biallelic_mask = ac.is_biallelic()
    mask = biallelic_mask if mask is None else mask & biallelic_mask

    # Apply segregating filter
    segregating_mask = ac.is_segregating()
    mask = segregating_mask if mask is None else mask & segregating_mask

    # Apply minor allele count filter
    if min_minor_ac is not None:
        an = ac.sum(axis=1)
        # Apply minor allele count condition.
        ac_minor = ac[:, 1:].sum(axis=1)
        if isinstance(min_minor_ac, float):
            ac_minor_frac = ac_minor / an
            loc_minor_mask = ac_minor_frac >= min_minor_ac
        else:
            loc_minor_mask = ac_minor >= min_minor_ac
        mask = loc_minor_mask if mask is None else mask & loc_minor_mask

    # Apply all filters at once
    if mask is not None:
        gt = gt.compress(mask)

    # Try to meet target number of SNPs.
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
