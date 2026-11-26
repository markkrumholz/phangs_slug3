"""
Plot the cluster_slug ages against the Thilker+25 ages, with error bars,
for the clusters most confidently identified as old globular clusters.

The clusters plotted are those that (1) lie in the old globular cluster
clump of the machine-classified color-color diagram
(PHANGS_CC_CLASS_ML = OGCC), (2) were fit by Thilker+25 with the old
globular cluster grid (PHANGS_DR5_THILKER25_GRID = OGC), and (3) have a
cluster_slug median mass (logM_50 in the text summary) of at least
--min-mass (default 5e4 Msun). Clusters without a Thilker+25 age are left
out.

The plot is in the same style as compare_thilker.py's tcomparison plot,
with error bars added: on the slug axis, the 16th to 84th percentile range
of the marginal posterior of log T; on the Thilker axis, the age
plus or minus PHANGS_DR5_THILKER25_AGE_ERR (the mean error, in Myr). Since
that error is symmetric in linear age and often exceeds the age, a lower
error bar that would reach zero age or below is drawn down to the bottom
of the plot. With --thilker-err limits, the Thilker error bars instead
span PHANGS_DR5_THILKER25_AGE_LIMLO to PHANGS_DR5_THILKER25_AGE_LIMHI.
"""

import argparse
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from compare_thilker import read_summary  # noqa: E402
from phangs_cluster_slug import CATALOG  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
LIMS = (5.5, 11)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summary", nargs="?",
                        default=osp.join(HERE, "phangs_cluster_slug_phangslib_summary.txt"),
                        help="text summary from phangs_cluster_slug.py (default: "
                        "phangs_cluster_slug_phangslib_summary.txt next to this script)")
    parser.add_argument("--catalog", default=CATALOG, help="PHANGS FITS catalog")
    parser.add_argument("--min-mass", type=float, default=5e4,
                        help="minimum cluster_slug median mass, in Msun (default 5e4)")
    parser.add_argument("--thilker-err", choices=["mean", "limits"], default="mean",
                        help="Thilker+25 error bars: the age plus or minus AGE_ERR (mean, the "
                        "default) or AGE_LIMLO to AGE_LIMHI (limits)")
    parser.add_argument("--tag", default="",
                        help="appended to the output name: tcomparison_gc<TAG>.pdf")
    args = parser.parse_args()

    s = read_summary(args.summary)
    rows, th, ncand = select_gc(args.catalog, s, args.min_mass)
    print(f"{ncand} clusters with CC_CLASS_ML = OGCC and THILKER25_GRID = OGC; "
          f"{len(rows)} of them with a Thilker+25 age and slug M_50 >= {args.min_mass:g} Msun")

    x = s["logT_50"][rows]
    xerr = [x - s["logT_16"][rows], s["logT_84"][rows] - x]
    y = np.log10(th["age"])
    if args.thilker_err == "mean":
        lo, hi = th["age"] - th["err"], th["age"] + th["err"]
    else:
        lo, hi = th["lo"], th["hi"]
    ylo = np.where(lo > 0, np.log10(np.where(lo > 0, lo, 1.0)), LIMS[0])
    yerr = [y - np.minimum(ylo, y), np.log10(hi) - y]

    fig, ax = plt.subplots()
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", ms=2, mew=0, elinewidth=0.5,
                alpha=0.3, capsize=0)
    ax.plot(LIMS, LIMS, "k--")
    ax.set_xlim(LIMS)
    ax.set_ylim(LIMS)
    ax.set_xlabel("log T [slug]")
    ax.set_ylabel("log T [Thilker+25]")
    ax.set_title(f"{len(rows)} OGCC/OGC clusters with slug $M_{{50}} \\geq$ "
                 f"{args.min_mass:g} M$_\\odot$; {osp.basename(args.summary)}", fontsize=9)
    fname = f"tcomparison_gc{args.tag}.pdf"
    fig.savefig(fname)
    plt.close(fig)
    print(f"Wrote {fname}")


if __name__ == "__main__":
    main()
