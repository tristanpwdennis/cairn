# cairn

**cairn** is a collection of utility functions for loading, querying, and analysing Zarr arrays of genomic data.

Packages like [sgkit](https://pystatgen.github.io/sgkit/) and [scikit-allel](https://scikit-allel.readthedocs.io/) provide excellent functionality for genome analysis. *Cairn* adds lightweight utilities and convenience wrappers for my own workflows — typically involving large Zarr arrays of genotype and variant data.

All functionality currently runs locally, with planned extensions for parallel computation on **Dask clusters** managed by **SLURM** or **Kubernetes**.

TO-DO:
- Add support for GCS.
- " Dask.
- Add spinners for everything. I love spinners.


## Naming

- *Cairns* are piles of rocks, often left on ridges or mountaintops to guide walkers.
- Passersby add a stone as they pass.
- Likewise, this library is a small stack of utilities — pragmatic additions on top of the excellent Python population-genomics ecosystem.


## Installation

```bash
git clone https://github.com/tristanpwdennis/cairn.git
cd cairn
pip install -e .

For development:

pip install -e ".[dev]"
