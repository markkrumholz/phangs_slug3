"""
Make triangle (corner) plots of the cluster_slug posteriors of PHANGS
clusters, from the HDF5 file written by phangs_cluster_slug.py.

The plot is made with npdfplot (from slug3's examples/cluster_slug), with
the same options as the cluster_slug_legus.py example, except that
scat_skip is 100 rather than 10, and the range plotted in each quantity
runs from the 1st percentile of that cluster's posterior in it minus 0.5
to the 99th percentile plus 0.5, truncated to the range covered by the
default cluster_slug library. The plot of cluster N is written to
triangleN.pdf in the current directory.

Usage examples::

    python plot_triangle.py 7
    python plot_triangle.py 2 - 10
    python plot_triangle.py 2-10 --file other_samples.h5
"""

import argparse
import os.path as osp
import re
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from phangs_kde import DEFAULT_FILE, SLUG_DIR, load_kde  # noqa: E402

sys.path.insert(0, osp.join(SLUG_DIR, "examples", "cluster_slug"))
from npdfplot import npdfplot  # noqa: E402

# Axis labels, as in cluster_slug_legus.py
LABELS = [r"$\log M$ [M$_\odot$]", r"$\log T$ [yr]", r"$A_V$ [mag]", "[Fe/H]"]

# Percentiles bounding the range plotted in each quantity, and the
# padding added on either side of them
PLOT_RANGE_PCT = [1, 99]
PLOT_RANGE_PAD = 0.5

# Range of each quantity covered by the default cluster_slug library
# (M = 10^2 - 10^7 Msun, T = 10^5 - 1.5 x 10^10 yr, A_V = 0 - 3 mag,
# [Fe/H] = -1 - +0.3); the plot range is truncated to the library's range,
# since the kernel's Gaussian broadening can carry percentiles slightly
# past it. This is used for sample files that do not record the range of
# the library they came from (in a "library_range" attribute).
LIBRARY_RANGE = {"mass": (2.0, 7.0),
                 "age": (5.0, float(np.log10(1.5e10))),
                 "A_V": (0.0, 3.0),
                 "metallicity": (-1.0, 0.3)}


def parse_indices(tokens):
    """
    Parse a cluster index or an inclusive range of them.

    Parameters
    ----------
    tokens : list of str
        The command-line arguments giving the index or range, e.g.
        ["7"], ["2-10"], or ["2", "-", "10"].

    Returns
    -------
    list of int
        The indices: [N] for a single index N, or N1, N1+1, ..., N2 for
        a range N1 - N2.

    Raises
    ------
    ValueError
        If the arguments are neither an index nor a range of them, or
        the range is empty.
    """
    text = " ".join(tokens)
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", text)
    if m is None:
        raise ValueError(f"expected an index or a range such as '2 - 10', not '{text}'")
    lo = int(m.group(1))
    hi = lo if m.group(2) is None else int(m.group(2))
    if hi < lo:
        raise ValueError(f"empty range '{text}'")
    return list(range(lo, hi + 1))


def plot_triangle(index, kde, galaxy=None, lib_range=None):
    """
    Make the triangle plot of one cluster's posterior and write it to file.

    Parameters
    ----------
    index : int
        The cluster's catalog index.
    kde : KDE
        Its posterior KDE.
    galaxy : str, optional
        Its galaxy, for the plot title.
    lib_range : dict, optional
        The (min, max) of each quantity covered by the library (default
        LIBRARY_RANGE).

    Returns
    -------
    str
        The name of the file written.
    """
    # Plot range in each quantity: the 1st to 99th percentiles of its
    # marginal posterior, padded by PLOT_RANGE_PAD on either side, and
    # truncated to the range the library covers
    lib_range = LIBRARY_RANGE if lib_range is None else lib_range
    lolim, hilim = [], []
    for i, name in enumerate(kde.names):
        lo, hi = kde.percentile(i, PLOT_RANGE_PCT)
        libmin, libmax = lib_range[name]
        lolim.append(max(float(lo) - PLOT_RANGE_PAD, libmin))
        hilim.append(min(float(hi) + PLOT_RANGE_PAD, libmax))

    # Plot, with the options of the cluster_slug_legus.py example
    npdfplot(kde.dataset,
             weights=kde.w,
             labels=LABELS,
             lolim=lolim,
             hilim=hilim,
             nbin=20,
             threshfrac=0.95,
             cbarmin=0,
             log=False,
             scat_skip=100,
             scat_alpha=0.25)
    fig = plt.gcf()
    fig.suptitle(f"Cluster {index}" + ("" if galaxy is None else f" ({galaxy})"))
    outname = f"triangle{index}.pdf"
    fig.savefig(outname, bbox_inches="tight")
    plt.close(fig)
    return outname


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("indices", nargs="+",
                        help="cluster index, or an inclusive range of them such as '2 - 10'")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help=f"HDF5 file of posterior samples (default {osp.basename(DEFAULT_FILE)})")
    args = parser.parse_args()
    try:
        indices = parse_indices(args.indices)
    except ValueError as err:
        parser.error(str(err))

    # Galaxy names, for the titles, and which indices the file holds
    with h5py.File(args.file, "r") as f:
        galaxies = {int(k): str(f[k].attrs.get("galaxy", "")) or None for k in f}
        lib_range = None
        if "library_range" in f.attrs:
            lib_range = {str(n): tuple(float(v) for v in r)
                         for n, r in zip(f.attrs["quantities"], f.attrs["library_range"])}

    nfail = 0
    for index in indices:
        if index not in galaxies:
            print(f"Cluster {index} is not in {args.file}; skipping it", file=sys.stderr)
            nfail += 1
            continue
        try:
            kde = load_kde(index, args.file)
        except ValueError as err:
            print(f"Skipping cluster {index}: {err}", file=sys.stderr)
            nfail += 1
            continue
        print(f"Wrote {plot_triangle(index, kde, galaxies[index], lib_range)}")
    if nfail:
        sys.exit(1)


if __name__ == "__main__":
    main()
