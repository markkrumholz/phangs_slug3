"""
Plot the distribution of clusters in the two nearest-neighbor distances
from a phangs_cluster_slug.py text summary.

The plot is made with npdfplot (from slug3's examples/cluster_slug), with
the distance to the 5th nearest library neighbor in the "errors" norm
(d5_err, in units of the photometric errors) on the x axis and in the
Euclidean norm (d5_euc, in mag) on the y axis. By default the plot spans
0 - 4 in d5_err and 0 - 0.6 mag in d5_euc, the PDFs are shown on a linear
scale with 50 bins in each dimension, and the individually shown points
have alpha 0.5; all of these can be changed with command-line options. The output defaults to the
summary's name with its final "_summary.txt" replaced by "_distances.pdf".

Usage examples::

    python plot_distances.py
    python plot_distances.py phangs_cluster_slug_summary.txt
    python plot_distances.py phangs_cluster_slug_summary.txt --out distances.pdf
"""

import argparse
import os.path as osp
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from phangs_kde import SLUG_DIR  # noqa: E402

sys.path.insert(0, osp.join(SLUG_DIR, "examples", "cluster_slug"))
from npdfplot import npdfplot  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
SUMMARY_SUFFIX = "_summary.txt"


def read_summary(filename):
    """
    Read a phangs_cluster_slug.py text summary.

    Parameters
    ----------
    filename : str
        The summary file.

    Returns
    -------
    numpy.ndarray
        A structured array with one field per column, named as in the
        file's column header line.
    """
    with open(filename) as f:
        cols = [line for line in f if line.startswith("#")][-1][1:].split()
    return np.genfromtxt(filename, names=cols, dtype=None, encoding=None)


def default_output(summary):
    """
    Default output file name for a summary file.

    Parameters
    ----------
    summary : str
        The summary file.

    Returns
    -------
    str
        The summary's name with a final "_summary.txt" replaced by
        "_distances.pdf" (or, if it does not end in "_summary.txt", with
        "_distances.pdf" appended to its name without its extension).
    """
    if summary.endswith(SUMMARY_SUFFIX):
        prefix = summary[:-len(SUMMARY_SUFFIX)]
    else:
        prefix = osp.splitext(summary)[0]
    return prefix + "_distances.pdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summary", nargs="?",
                        default=osp.join(HERE, "phangs_cluster_slug_phangslib_summary.txt"),
                        help="text summary from phangs_cluster_slug.py (default: "
                        "phangs_cluster_slug_phangslib_summary.txt next to this script)")
    parser.add_argument("--out", default=None,
                        help="output file (default: the summary's name with _summary.txt "
                        "replaced by _distances.pdf)")
    parser.add_argument("--err-range", type=float, nargs=2, default=[0.0, 4.0],
                        metavar=("MIN", "MAX"),
                        help="range of the errors-norm distance plotted (default 0 4)")
    parser.add_argument("--abs-range", type=float, nargs=2, default=[0.0, 0.6],
                        metavar=("MIN", "MAX"),
                        help="range of the Euclidean-norm distance plotted, in mag "
                        "(default 0 0.6)")
    parser.add_argument("--log", action=argparse.BooleanOptionalAction, default=False,
                        help="show the PDFs on a logarithmic rather than linear scale "
                        "(default: linear)")
    parser.add_argument("--nbin", type=int, default=50,
                        help="number of bins in each dimension (default 50)")
    parser.add_argument("--scat-alpha", type=float, default=0.5,
                        help="alpha of the individually shown points (default 0.5)")
    args = parser.parse_args()
    out = args.out or default_output(args.summary)

    s = read_summary(args.summary)
    pts = np.column_stack([s["d5_err"], s["d5_euc"]])
    pts = pts[np.all(np.isfinite(pts), axis=1)]

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=12)
    npdfplot(pts,
             lolim=[args.err_range[0], args.abs_range[0]],
             hilim=[args.err_range[1], args.abs_range[1]],
             log=args.log,
             nbin=args.nbin,
             scat_alpha=args.scat_alpha,
             labels=[r"$D^5_\mathrm{err}$", r"$D^5_\mathrm{abs}$ [mag]"])
    plt.gcf().savefig(out, bbox_inches="tight")
    print(f"Wrote {out} ({len(pts)} clusters)")


if __name__ == "__main__":
    main()
