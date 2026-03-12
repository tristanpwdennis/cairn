import allel
import numpy as np
import pandas as pd
import os
import zarr

from cairn.utils import parse_region, locate_region, hash_params, hash_columns
from cairn.snps import load_genotype_array

from collections import Counter


def diplotype_frequencies(gt):
    """Compute diplotype frequencies, returning a dictionary that maps
    diplotype hash values to frequencies."""

    # Here are some optimisations to speed up the computation
    # of diplotype hashes. First we combine the two int8 alleles
    # in each genotype call into a single int16.
    m = gt.shape[0]
    n = gt.shape[1]
    x = np.asarray(gt).view(np.int16).reshape((m, n))

    # Now call optimised hashing function.
    hashes = hash_columns(x)

    # Now compute counts and frequencies of distinct haplotypes.
    counts = Counter(hashes)
    freqs = {key: count / n for key, count in counts.items()}

    return freqs
    
def garud_g123(gt):
    """Compute Garud's G123."""

    # compute diplotype frequencies
    frq_counter = diplotype_frequencies(gt)

    # convert to array of sorted frequencies
    f = np.sort(np.fromiter(frq_counter.values(), dtype=float))[::-1]

    # compute G123
    g123 = np.sum(f[:3]) ** 2 + np.sum(f[3:] ** 2)

    return g123

    
def g123_analysis(
              zarr_base_path: str, 
              df_samples: pd.DataFrame, 
              contig: str, 
              window_size: int, 
              sample_query: str = None,
              filter_mask: str = None,
              genotype_var: str = "calldata/GT",
              pos_var: str = "variants/POS",
              results_dir='results_g123_v1',
              overwrite=False
              ):
    
    params = dict(
            contig=contig,
            window_size=window_size,
            sample_query=sample_query,
        )

     # construct a key to save the results under
    results_key = hash_params(
        params
    )

    # define paths for results files
    g123_path = f'{results_dir}/{results_key}-g123.npy'
    x_path = f'{results_dir}/{results_key}-x.npy'

    # Hashing or not
    if overwrite is False:
        try:
            # try to load previously generated results
            g123 = np.load(g123_path)
            x = np.load(x_path)
            return (g123, x)
        except FileNotFoundError:
            # no previous results available, need to run analysis
            print(f'running analysis: {results_key}')
    
    gt = load_genotype_array(
            zarr_base_path=zarr_base_path,
            region=contig,
            df_samples=df_samples,
            genotype_var=genotype_var,
            pos_var=pos_var,
            sample_query=sample_query,
            filter_mask=filter_mask,
        )

    # Get pos data: TODO -refactor this into utils
    z = zarr.open(zarr_base_path.format(contig=contig))

    # Get variant position array
    pos = z["variants/POS"]
    flt = z["variants/filter_pass"][:]
    pos = pos[flt]

    g123 = allel.moving_statistic(gt, statistic=garud_g123, size=window_size)
    x = allel.moving_statistic(pos, statistic=np.mean, size=window_size)

    # Save outputs
    os.makedirs(results_dir, exist_ok=True)
    np.save(g123_path, g123)
    np.save(x_path, x)

    print(f'saved results: {results_key}')

    return (g123, x)


