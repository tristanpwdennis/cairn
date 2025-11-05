"""
frq.py
-----

SNP frequencies

This module provides modules for computing allele frequencies and
site-frequency spectra.

"""

import allel
import pandas as pd

from cairn.io import load_genotype_array, _select_random_elements_sorted

from typing import Union

from yaspin import yaspin


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

    # Random selection
    if n_snps is not None:
        gt = _select_random_elements_sorted(gt, n_snps)

    return gt.count_alleles()
