# phangs_slug3

Bayesian inference of star cluster masses, ages, extinctions, and
metallicities for the PHANGS-HST DR5 star cluster catalog, using
[`cluster_slug`](https://markkrumholz.github.io/slug3/clusterslug.html)
from [slug3](https://github.com/markkrumholz/slug3) and its default
cluster library.

The analysis follows the `cluster_slug` example in slug3's
`examples/cluster_slug` directory, applied to the PHANGS DR5
machine-classified class 1+2 "primary" catalog
(`hlsp_phangs-cat_hst_acs-uvis_dr5-targets_v1_machine-class1+2-primary.fits`,
with its README `hlsp_phangs-cat_dr5-targets_v1_primary-readme.txt`).
Neither the catalog nor any output is included in this repository.

## Files

### `phangs_cluster_slug.py`

The main analysis script. For each cluster in the catalog it:

1. converts the apparent Vega magnitudes in the NUV, U, B, V, and I bands
   to absolute magnitudes using the catalog distance modulus
   (`PHANGS_DM`);
2. picks the library filter for each band from the instrument listed in
   `PHANGS_OBS_SUMMARY`: WFC3/UVIS (F275W, F336W, F438W, F555W, F814W)
   or ACS/WFC (F435W, F555W, F814W), since B, V, and I come from ACS for
   some galaxies. Bands with missing photometry are left out, and each
   distinct combination of filters gets its own `cluster_slug` filter
   set;
3. finds the 5 nearest neighbors of the cluster's photometry in the
   library, using both the `errors` norm (distance in units of the
   photometric errors) and the `Euclidean` norm (distance in mag);
4. draws the posterior sample points and weights with
   `conditional_sample` (default `massTol` = 1e-3);
5. builds the posterior KDE from that sample and computes the 5th, 16th,
   50th, 84th, and 95th percentiles of the marginal posterior of log
   mass, log age, A_V, and [Fe/H].

The prior in mass and age is d²N/dM dT ∝ M^α_M max(T, T_break)^α_T,
by default with α_M = −1.5, T_break = 10^6.5 yr, and α_T = −0.5
(`--alpha-m`, `--log-t-break`, `--alpha-t`). It is flat in A_V and
[Fe/H]. The prior is recorded in the text summary's header and in the
HDF5 file's root attributes.

Outputs, written next to the script:

- `phangs_cluster_slug_summary.txt`: one line per cluster giving the
  catalog index, galaxy, distance to the 5th nearest neighbor in each
  norm (`d5_err`, `d5_euc`), and the marginal percentiles. A commented
  header describes the columns.
- `phangs_cluster_slug_samples.h5`: one group per cluster, named by its
  catalog index, containing the posterior sample `points` (K × 4: log
  M/M☉, log T/yr, A_V, [Fe/H]) and `weights` (gzip-compressed float32).
  It also holds the distances to the 5 nearest neighbors in each norm
  (`dist_err`, `dist_euc`). Each group's attributes give the galaxy, the
  filters used, and the posterior kernel's `h2Inv`.

Usage:

```bash
python phangs_cluster_slug.py              # first 20 clusters (test run)
python phangs_cluster_slug.py --ncluster 0 # whole catalog
# every 25th cluster, with a steeper mass prior
python phangs_cluster_slug.py --ncluster 0 --skip 25 --alpha-m -2 \
    --out-text skip25_summary.txt --out-hdf5 skip25_samples.h5
```

`--library FILE` uses another `cluster_slug` library instead of the
default one; the library and the range of each quantity it covers are
recorded in the outputs (`plot_triangle.py` truncates its axes to that
range). `--skip N` analyzes only every Nth cluster of the catalog (applied before
`--ncluster`); `--index I [I ...]` analyzes only the clusters with those
catalog indices (and cannot be combined with `--skip` or `--ncluster`);
`--catalog` selects a different input catalog. Run `python phangs_cluster_slug.py --help` for all options. The full
catalog's HDF5 output is estimated at about 34 GB, far more than fits in
memory, so the script processes clusters in chunks (`--chunk`, default
500) and writes each chunk's posterior samples to the HDF5 file as it
goes. A run on a 500-cluster sample peaked at about 12 GB of memory.

### `phangs_kde.py`

Provides `load_kde(index, filename=None)`, which rebuilds the posterior
KDE (a `slugpy.kde.KDE`) of one cluster, or of each of a list of
clusters, from the HDF5 file. It returns a single KDE for a single index
and a list for several:

```python
from phangs_kde import load_kde
kde = load_kde(1)
kde.percentile("mass", [16, 50, 84])
kdes = load_kde(range(1, 6))
```

### `plot_triangle.py`

Makes a triangle (corner) plot of the posterior of a cluster, or of each
cluster in a range, using `npdfplot` from slug3's `examples/cluster_slug`.
Each axis spans the cluster's 1st to 99th percentile in that quantity,
widened by ±0.5 and truncated to the range covered by the library. The
plot of cluster N is written to `triangleN.pdf` in the current directory.

```bash
python plot_triangle.py 7
python plot_triangle.py 2 - 10
python plot_triangle.py 2-10 --file other_samples.h5
```

### `split_by_galaxy.py`

Splits the posterior sample file into one file per galaxy, named after
the input with the galaxy inserted after `phangs_` (so
`phangs_cluster_slug_samples.h5` gives
`phangs_GALAXY_cluster_slug_samples.h5`) and written next to the input
unless `--out-dir` is given. Each
has the same layout and root attributes as the input, plus a root
attribute `galaxy`, so `load_kde(index, filename)` can read it. The
compressed data are copied without decompressing them.

```bash
python split_by_galaxy.py                     # phangs_cluster_slug_samples.h5
python split_by_galaxy.py other_samples.h5 --out-dir split/
```

### `compare_thilker.py`

Plots the cluster_slug median log mass and log age (50th percentiles from
a text summary) against the Thilker+25 deterministic fits in the PHANGS
catalog, with the 1:1 line, writing `mcomparison<TAG>.pdf` and
`tcomparison<TAG>.pdf` in the current directory.

```bash
python compare_thilker.py phangs_cluster_slug_summary.txt
python compare_thilker.py other_summary.txt --tag _other
```

### `plot_distances.py`

Plots the distribution of clusters in the two nearest-neighbor distances
from a text summary, using `npdfplot` from slug3's `examples/cluster_slug`:
the 5th-neighbor distance in the errors norm (`d5_err`) on the x axis and
in the Euclidean norm (`d5_euc`, mag) on the y axis. By default it reads
`phangs_cluster_slug_phangslib_summary.txt`, writes the summary's name with
`_summary.txt` replaced by `_distances.pdf`, plots 0 - 4 in `d5_err` and
0 - 0.6 mag in `d5_euc` on a linear scale with 50 bins in each dimension
(`--nbin`), and draws the individually
shown points with alpha 0.5. Clusters outside the plotted range are left
out. Labels are rendered with LaTeX, so a LaTeX installation is needed.

```bash
python plot_distances.py
python plot_distances.py phangs_cluster_slug_summary.txt --err-range 0 5 \
    --abs-range 0 1 --log --scat-alpha 1
```

### `galaxy_distributions.py`

Computes the distribution of a cluster property ([Fe/H] by default) in
each galaxy: the average of the marginal posterior PDFs of the galaxy's
clusters. Each cluster's sample weights are normalized to sum to 1, so that
every cluster counts equally, and the samples are combined into one KDE;
the same percentiles as for single clusters are written to a text file
(by default the input's name with `_samples.h5` replaced by
`_galaxy_FeH.txt`).

```bash
python galaxy_distributions.py phangs_cluster_slug_samples.h5
python galaxy_distributions.py phangs_cluster_slug_samples.h5 --quantity age
```

## Requirements

- slug3, built with its Python bindings, with the default `cluster_slug`
  library downloaded (`python3 data/tools/download_data.py --subdir
  cluster_slug`, about 33 GB). The scripts expect the slug3 checkout at
  `~/Dropbox/Projects/slug3` (`SLUG_DIR` in `phangs_cluster_slug.py` and
  `phangs_kde.py`).
- Python ≥ 3.12 (required by slugpy), with numpy, scipy, astropy, h5py,
  matplotlib, and slugpy's other dependencies.

## License

BSD 3-Clause; see [LICENSE](LICENSE). This is the same license as slug3.
