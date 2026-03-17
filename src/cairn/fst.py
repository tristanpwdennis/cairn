import allel
import numpy as np
import pandas as pd
import os
import zarr

from cairn.utils import parse_region, locate_region, hash_params, hash_columns
from cairn.snps import load_genotype_array

from collections import Counter

from yaspin import yaspin


# Mapper class for plotting contigs in a GWSS
class GenomePositionMapper:
    def __init__(self, contig_lengths_dict):
        """
        contig_lengths_dict: dict {contig_id: length}, already sorted longest first
        """
        self.sorted_contigs = list(contig_lengths_dict.keys())
        sorted_lengths = [contig_lengths_dict[c] for c in self.sorted_contigs]

        # compute offsets
        contig_offsets = np.cumsum([0] + sorted_lengths[:-1])
        self.contig_offsets = dict(zip(self.sorted_contigs, contig_offsets))

        # zebra striping
        self.color_map = {
            contig: ("lightblue" if i % 2 == 0 else "steelblue")
            for i, contig in enumerate(self.sorted_contigs)
        }

    def _apply(self, df, coord):
        df = df.copy()
        df["contig"] = pd.Categorical(df["contig"], categories=self.sorted_contigs, ordered=True)
        df = df.sort_values(["contig", coord])
        df["contig_offset"] = df["contig"].map(self.contig_offsets).astype(float)
        df["genome_position"] = df[coord] + df["contig_offset"]
        df["color"] = df["contig"].map(self.color_map)
        return df

    def transform_positions(self, df):
        return self._apply(df, "pos")

    def transform_annotations(self, df):
        return self._apply(df, "start")

# Main fst function for windowed Fst
def fst_gwss(
    sample_query_a: str,
    sample_query_b: str,
    window_size: int,
    contig: str,
    zarr_base_path: str, 
    df_samples: pd.DataFrame, 
    clip_min : float | int = 0,
    genotype_var: str = "calldata/GT",
    pos_var: str = "variants/POS",
    filter_mask: str = 'variants/filter_pass',
    results_dir: str ='results_cache/results_fst_v1',
    overwrite: bool = False,
    ) -> pd.DataFrame: 

    """"
    Perform genome scan of windowed Hudson's Fst across a specific genomic region (contig or region).
    """

    # Set up params for hashing
    params = dict(
            sample_query_a = sample_query_a,
            sample_query_b = sample_query_b,
            contig=contig,
            window_size=window_size,
        )

    results_key = hash_params(
        params
    )

    # define paths for results files
    fst_path = f'{results_dir}/{results_key}-fst.npy'
    x_path = f'{results_dir}/{results_key}-x.npy'


    # Hashing or not
    if overwrite is False:
        try:
            # try to load previously generated results
            fst = np.load(fst_path)
            x = np.load(x_path)
            return (fst, x)
        except FileNotFoundError:
            # no previous results available, need to run analysis
            print(f'running analysis: {results_key}')

    # Load genos for query a
    ac_a = load_genotype_array(
            zarr_base_path=zarr_base_path,
            region=contig,
            df_samples=df_samples,
            genotype_var=genotype_var,
            pos_var=pos_var,
            sample_query=sample_query_a,
            filter_mask=filter_mask,
        ).count_alleles()

    # Load genos for query b
    ac_b = load_genotype_array(
            zarr_base_path=zarr_base_path,
            region=contig,
            df_samples=df_samples,
            genotype_var=genotype_var,
            pos_var=pos_var,
            sample_query=sample_query_b,
            filter_mask=filter_mask,
        ).count_alleles()

    
    # Get pos data: TODO -refactor this into utils
    z = zarr.open(zarr_base_path.format(contig=contig))

    # Get variant position array
    pos = z["variants/POS"]
    flt = z["variants/filter_pass"][:]
    pos = pos[flt]

    with yaspin("Compute Fst..."):
        with np.errstate(divide="ignore", invalid="ignore"):
            fst = allel.moving_hudson_fst(ac_a, ac_b, size=window_size)
            # Sometimes Fst can be very slightly below zero, clip for simplicity.
            fst = np.clip(fst, a_min=clip_min, a_max=1)
            x = allel.moving_statistic(pos, statistic=np.mean, size=window_size)

    # Save outputs
    os.makedirs(results_dir, exist_ok=True)
    np.save(fst_path, fst)
    np.save(x_path, x)

    print(f'saved results: {results_key}')

    return (fst, x)


# Main fst function for windowed Fst
def fst_average(
    sample_query_a: str,
    sample_query_b: str,
    region: str,
    zarr_base_path: str, 
    df_samples: pd.DataFrame, 
    block_length: int  = 2000,
    clip_min : float | int = 0,
    genotype_var: str = "calldata/GT",
    pos_var: str = "variants/POS",
    filter_mask: str = 'variants/filter_pass',
    results_dir: str ='results_cache/results_fst_av_v1',
    overwrite: bool = False,
    )-> tuple: 

    """"
    Compute the average Hudson's Fst and standard error over a genomic region. Returns a tuple of Fst and SE.
    """

    # Set up params for hashing
    params = dict(
            sample_query_a = sample_query_a,
            sample_query_b = sample_query_b,
            contig=region,
        )

    results_key = hash_params(
        params
    )

    # define paths for results files
    fst_path = f'{results_dir}/{results_key}-fst.npy'

    # Hashing or not
    if overwrite is False:
        try:
            # try to load previously generated results
            fst, se = np.load(fst_path)
            return (fst, se)
        except FileNotFoundError:
            # no previous results available, need to run analysis
            print(f'running analysis: {results_key}')

    # Load genos for query a
    ac_a = load_genotype_array(
            zarr_base_path=zarr_base_path,
            region=region,
            df_samples=df_samples,
            genotype_var=genotype_var,
            pos_var=pos_var,
            sample_query=sample_query_a,
            filter_mask=filter_mask,
        ).count_alleles()

    # Load genos for query b
    ac_b = load_genotype_array(
            zarr_base_path=zarr_base_path,
            region=region,
            df_samples=df_samples,
            genotype_var=genotype_var,
            pos_var=pos_var,
            sample_query=sample_query_b,
            filter_mask=filter_mask,
        ).count_alleles()

    with yaspin("Compute Fst..."):
        with np.errstate(divide="ignore", invalid="ignore"):
            fst, se, vb, vj = allel.average_hudson_fst(ac_a, ac_b, blen=block_length)
            # Sometimes Fst can be very slightly below zero, clip for simplicity.

    
    # Save outputs
    os.makedirs(results_dir, exist_ok=True)
    np.save(fst_path, np.array([fst, se]))
    
    print(f'saved results: {results_key}')

    return (fst, se)