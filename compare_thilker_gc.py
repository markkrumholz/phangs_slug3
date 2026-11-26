"""
Plot the joint distribution of cluster_slug and Thilker+25 ages for the
clusters most confidently identified as old globular clusters.

The clusters plotted are those that (1) lie in the old globular cluster
clump of the machine-classified color-color diagram
(PHANGS_CC_CLASS_ML = OGCC), (2) were fit by Thilker+25 with the old
globular cluster grid (PHANGS_DR5_THILKER25_GRID = OGC), and (3) have a
cluster_slug median mass (logM_50 in the text summary) of at least
--min-mass (default 5e4 Msun). Clusters without a Thilker+25 age are left
out.

The plot is made with npdfplot (from slug3's examples/cluster_slug), as a
heatmap with flanking histograms on linear scales, in the same style as
plot_distances.py, with the 1:1 line added to the heatmap. Its points are
the clusters' posterior samples from the HDF5 sample file: each sample
contributes its log age as the slug value and its weight, with each
cluster's weights normalized to sum to 1 so that every cluster counts
equally. For each sample, a Thilker+25 age is drawn from a Gaussian with
mean PHANGS_DR5_THILKER25_AGE and dispersion PHANGS_DR5_THILKER25_AGE_ERR
(both in Myr, so the Gaussian is in linear age), which marginalizes over
the Thilker+25 errors. Since that error often exceeds the age, the Gaussian
is truncated at zero age (negative draws are redrawn), so that every draw
has a logarithm.
"""

import argparse
import os.path as osp
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from astropy.io import fits  # noqa: E402

from compare_thilker import read_summary  # noqa: E402
from phangs_cluster_slug import CATALOG  # noqa: E402
from phangs_kde import SLUG_DIR  # noqa: E402

sys.path.insert(0, osp.join(SLUG_DIR, "examples", "cluster_slug"))
from npdfplot import npdfplot  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
SUMMARY_SUFFIX = "_summary.txt"


def select_gc(catalog, summary, min_mass):
    """
    Select the confident old globular clusters.

    Parameters
    ----------
    catalog : str
        The PHANGS FITS catalog.
    summary : numpy.ndarray
        The text summary, as returned by read_summary.
    min_mass : float
        Minimum cluster_slug median mass, in Msun.

    Returns
    -------
    rows : numpy.ndarray of int
        Rows of the summary of the selected clusters.
    thilker : dict of numpy.ndarray
        The selected clusters' Thilker+25 age ("age"), mean age error
        ("err"), and lower and upper 1-sigma limits ("lo", "hi"), all in yr.
    ncand : int
        Number of clusters meeting the catalog criteria (1) and (2).
    """
    with fits.open(catalog) as hdul:
        d = hdul[1].data

        def col(name):
            return np.char.strip(np.asarray(d[name]).astype(str))

        cand = (col("PHANGS_CC_CLASS_ML") == "OGCC") & (col("PHANGS_DR5_THILKER25_GRID") == "OGC")
        cat_index = np.asarray(d["INDEX"]).astype(int)[cand]
        ages = {key: np.asarray(d[f"PHANGS_DR5_THILKER25_{name}"], dtype=float)[cand] * 1e6
                for key, name in [("age", "AGE"), ("err", "AGE_ERR"),
                                  ("lo", "AGE_LIMLO"), ("hi", "AGE_LIMHI")]}
    pos = {idx: k for k, idx in enumerate(summary["index"])}
    in_summary = np.array([i in pos for i in cat_index])
    rows = np.array([pos.get(i, -1) for i in cat_index])
    keep = in_summary & np.isfinite(ages["age"]) & (ages["age"] > 0)
    keep[keep] &= summary["logM_50"][rows[keep]] >= np.log10(min_mass)
    return rows[keep], {k: v[keep] for k, v in ages.items()}, int(cand.sum())


def truncated_normal(rng, mean, sigma, n):
    """
    Draw from a Gaussian truncated to positive values.

    Parameters
    ----------
    rng : numpy.random.Generator
        The random number generator.
    mean, sigma : float
        Mean and dispersion of the Gaussian before truncation.
    n : int
        Number of draws.

    Returns
    -------
    numpy.ndarray, shape (n,)
        The draws, all > 0.
    """
    x = rng.normal(mean, sigma, n)
    bad = x <= 0
    while np.any(bad):
        x[bad] = rng.normal(mean, sigma, bad.sum())
        bad = x <= 0
    return x


def joint_samples(samples, indices, thilker, rng):
    """
    Joint slug and Thilker+25 log ages of the clusters' posterior samples.

    Parameters
    ----------
    samples : str
        The HDF5 sample file.
    indices : sequence of int
        Catalog indices of the clusters.
    thilker : dict of numpy.ndarray
        The clusters' Thilker+25 ages ("age") and errors ("err"), in yr.
    rng : numpy.random.Generator
        The random number generator for the Thilker+25 draws.

    Returns
    -------
    pts : numpy.ndarray, shape (N, 2)
        log T/yr from slug and from a draw from the Thilker+25 Gaussian, for
        each sample with positive weight.
    wgts : numpy.ndarray, shape (N,)
        The samples' weights, normalized to sum to 1 for each cluster.
    """
    with h5py.File(samples, "r") as f:
        iage = [str(q) for q in f.attrs["quantities"]].index("age")
        groups = [f[str(i)] for i in indices]
        npts = sum(g["weights"].shape[0] for g in groups)
        pts = np.empty((npts, 2))
        wgts = np.empty(npts)
        n = 0
        for g, age, err in zip(groups, thilker["age"], thilker["err"], strict=True):
            w = g["weights"][()].astype(np.float64)
            ok = w > 0
            k = int(ok.sum())
            if k == 0:
                continue
            pts[n:n + k, 0] = g["points"][:, iage][ok]
            pts[n:n + k, 1] = np.log10(truncated_normal(rng, age, err, k))
            wgts[n:n + k] = w[ok] / w[ok].sum()
            n += k
    return pts[:n], wgts[:n]


def default_samples(summary):
    """The sample file matching a summary: _summary.txt replaced by _samples.h5."""
    if summary.endswith(SUMMARY_SUFFIX):
        return summary[:-len(SUMMARY_SUFFIX)] + "_samples.h5"
    return osp.splitext(summary)[0] + "_samples.h5"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summary", nargs="?",
                        default=osp.join(HERE, "phangs_cluster_slug_phangslib_summary.txt"),
                        help="text summary from phangs_cluster_slug.py (default: "
                        "phangs_cluster_slug_phangslib_summary.txt next to this script)")
    parser.add_argument("--samples", default=None,
                        help="HDF5 sample file (default: the summary's name with _summary.txt "
                        "replaced by _samples.h5)")
    parser.add_argument("--catalog", default=CATALOG, help="PHANGS FITS catalog")
    parser.add_argument("--min-mass", type=float, default=5e4,
                        help="minimum cluster_slug median mass, in Msun (default 5e4)")
    parser.add_argument("--range", type=float, nargs=2, default=[7.0, 10.5],
                        metavar=("MIN", "MAX"),
                        help="range of log T/yr plotted on both axes (default 7 10.5)")
    parser.add_argument("--nbin", type=int, default=50,
                        help="number of bins in each dimension (default 50)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the Thilker+25 age draws (default 0)")
    parser.add_argument("--tag", default="",
                        help="appended to the output name: tcomparison_gc<TAG>.pdf")
    args = parser.parse_args()
    samples = args.samples or default_samples(args.summary)

    s = read_summary(args.summary)
    rows, th, ncand = select_gc(args.catalog, s, args.min_mass)
    print(f"{ncand} clusters with CC_CLASS_ML = OGCC and THILKER25_GRID = OGC; "
          f"{len(rows)} of them with a Thilker+25 age and slug M_50 >= {args.min_mass:g} Msun",
          flush=True)
    pts, wgts = joint_samples(samples, s["index"][rows], th, np.random.default_rng(args.seed))
    print(f"{len(pts)} posterior samples from {osp.basename(samples)}", flush=True)

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=12)
    lims = args.range
    # npdfplot shows the points in its lowest-density bins individually,
    # ignoring their weights, which for weighted posterior samples would be
    # misleading, so suppress them (scat_skip shows at most one point, and
    # scat_alpha makes it invisible); the contour still marks the region
    # holding 99% of the weight
    npdfplot(pts, lolim=[lims[0], lims[0]], hilim=[lims[1], lims[1]], log=False,
             nbin=args.nbin, weights=wgts, scat_alpha=0, scat_skip=len(pts),
             labels=[r"$\log T_\mathrm{slug}$ [yr]", r"$\log T_\mathrm{Thilker}$ [yr]"])
    fig = plt.gcf()
    # npdfplot makes the axes in the order: first histogram, heatmap,
    # second histogram (then the color bar)
    ax = fig.axes[1]
    ax.plot(lims, lims, "k--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fname = f"tcomparison_gc{args.tag}.pdf"
    fig.savefig(fname, bbox_inches="tight")
    print(f"Wrote {fname}")


if __name__ == "__main__":
    main()
