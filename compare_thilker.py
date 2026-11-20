"""
Plot the cluster_slug median masses and ages against the Thilker+25
deterministic fits listed in the PHANGS catalog.

Writes two plots: log M from cluster_slug (50th percentile of the marginal
posterior) against Thilker+25's log M, and the same for log age, each with
the 1:1 line. Clusters without a Thilker+25 fit are left out.
"""

import argparse
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from phangs_cluster_slug import CATALOG  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))


def read_summary(filename):
    """Read a phangs_cluster_slug.py text summary into a structured array."""
    with open(filename) as f:
        cols = [line for line in f if line.startswith("#")][-1][1:].split()
    return np.genfromtxt(filename, names=cols, dtype=None, encoding=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summary", nargs="?",
                        default=osp.join(HERE, "phangs_cluster_slug_summary.txt"),
                        help="text summary from phangs_cluster_slug.py")
    parser.add_argument("--catalog", default=CATALOG, help="PHANGS FITS catalog")
    parser.add_argument("--tag", default="",
                        help="appended to the output names: mcomparison<TAG>.pdf, "
                        "tcomparison<TAG>.pdf")
    args = parser.parse_args()

    s = read_summary(args.summary)
    with fits.open(args.catalog) as hdul:
        d = hdul[1].data
        cat_index = np.asarray(d["INDEX"]).astype(int)
        t_age = np.asarray(d["PHANGS_DR5_THILKER25_AGE"], dtype=float) * 1e6  # yr
        t_mass = np.asarray(d["PHANGS_DR5_THILKER25_MASS"], dtype=float)
    pos = {idx: k for k, idx in enumerate(cat_index)}
    rows = np.array([pos[i] for i in s["index"]])
    t_age, t_mass = t_age[rows], t_mass[rows]

    for qty, thilker, label, lims, out in [
            ("logM_50", t_mass, "log M", (2, 7), "mcomparison"),
            ("logT_50", t_age, "log T", (5.5, 11), "tcomparison")]:
        ok = np.isfinite(thilker) & (thilker > 0) & np.isfinite(s[qty])
        fig, ax = plt.subplots()
        ax.scatter(s[qty][ok], np.log10(thilker[ok]), s=2, alpha=0.3, lw=0)
        ax.plot(lims, lims, "k--")
        ax.set_xlabel(f"{label} [slug]")
        ax.set_ylabel(f"{label} [Thilker+25]")
        ax.set_title(f"{ok.sum()} clusters; {osp.basename(args.summary)}", fontsize=9)
        fname = f"{out}{args.tag}.pdf"
        fig.savefig(fname)
        plt.close(fig)
        print(f"Wrote {fname} ({ok.sum()} clusters)")


if __name__ == "__main__":
    main()
