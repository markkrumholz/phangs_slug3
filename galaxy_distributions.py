"""
Distribution of a cluster property in each galaxy, from the average of its
clusters' marginal posterior PDFs.

For each galaxy, this script combines the posterior samples of all of the
galaxy's clusters into one sample, builds a KDE from it, and computes the
same percentiles as phangs_cluster_slug.py does for single clusters. The
KDE of the combined sample is the average of the clusters' posterior PDFs
with every cluster weighted equally: each cluster's weights are first
normalized to sum to 1 (conditional_sample's weights are not normalized,
and their sums differ between clusters by orders of magnitude), with no
other weighting applied.

Only the one-dimensional marginal of the chosen quantity is needed, so the
KDE is built directly from that quantity's column, with the marginal
kernel bandwidth; this gives the same result as building the full KDE and
marginalizing it, at much lower cost.
"""

import argparse
import os.path as osp
import sys
import time

import h5py
import numpy as np

from phangs_cluster_slug import PERCENTILES, QUANTITY_LABELS, SLUG_DIR

sys.path.insert(0, SLUG_DIR)
from slugpy.kde import KDE  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))


def marginal_h2inv(h2inv, i):
    """
    Inverse squared bandwidth of a KDE's one-dimensional marginal.

    Parameters
    ----------
    h2inv : numpy.ndarray, shape (nqty, nqty)
        The full KDE's inverse squared bandwidth matrix.
    i : int
        The dimension kept.

    Returns
    -------
    float
        The marginal's inverse squared bandwidth: the inverse of the
        kernel covariance's (i, i) element.
    """
    return 1.0 / np.linalg.inv(h2inv)[i, i]


def combined_percentiles(groups, qty):
    """
    Percentiles of the average of several clusters' marginal posteriors.

    Parameters
    ----------
    groups : list of h5py.Group
        The clusters' groups in the sample file.
    qty : int
        Column of the quantity in the clusters' points.

    Returns
    -------
    pcts : numpy.ndarray, shape (len(PERCENTILES),)
        The percentiles of the averaged marginal PDF (NaN if no cluster
        has a non-empty sample).
    nused : int
        Number of clusters with non-empty samples, which are averaged.
    npts : int
        Total number of sample points.

    Raises
    ------
    ValueError
        If the clusters' posterior kernels differ.
    """
    # Fill preallocated arrays rather than concatenating, since for the
    # largest galaxies the combined sample has hundreds of millions of
    # points
    npts = sum(g["weights"].shape[0] for g in groups)
    x = np.empty((npts, 1))
    w = np.empty(npts)
    n, nused, h2inv = 0, 0, None
    for g in groups:
        wgts = g["weights"][()].astype(np.float64)
        if len(wgts) == 0 or not np.any(wgts > 0):
            continue
        if h2inv is None:
            h2inv = g.attrs["h2Inv"]
        elif not np.array_equal(g.attrs["h2Inv"], h2inv):
            raise ValueError(f"cluster {g.name} has a different posterior kernel bandwidth; "
                             "the clusters' PDFs cannot be combined into a single KDE")
        x[n:n + len(wgts), 0] = g["points"][:, qty]
        w[n:n + len(wgts)] = wgts / wgts.sum()
        n += len(wgts)
        nused += 1
    if nused == 0:
        return np.full(len(PERCENTILES), np.nan), 0, 0
    x, w = x[:n], w[:n]
    kde = KDE(x, h2Inv=np.array([[marginal_h2inv(h2inv, qty)]]), w=w)
    return np.asarray(kde.percentile(0, PERCENTILES)), nused, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("infile", nargs="?",
                        default=osp.join(HERE, "phangs_cluster_slug_samples.h5"),
                        help="sample file from phangs_cluster_slug.py (default: "
                        "phangs_cluster_slug_samples.h5 next to this script)")
    parser.add_argument("--quantity", default="metallicity", choices=list(QUANTITY_LABELS),
                        help="the cluster property (default metallicity)")
    parser.add_argument("--out", default=None,
                        help="output text file (default: the input's name with _samples.h5 "
                        "replaced by _galaxy_<quantity>.txt)")
    args = parser.parse_args()
    label = QUANTITY_LABELS[args.quantity]
    out = args.out or (args.infile.removesuffix(".h5").removesuffix("_samples")
                       + f"_galaxy_{label}.txt")

    t0 = time.time()
    rows = []
    with h5py.File(args.infile, "r") as f:
        names = [str(n) for n in f.attrs["quantities"]]
        qty = names.index(args.quantity)
        by_galaxy = {}
        for key in f:
            by_galaxy.setdefault(str(f[key].attrs["galaxy"]), []).append(key)
        print(f"{sum(map(len, by_galaxy.values()))} clusters in {len(by_galaxy)} galaxies in "
              f"{osp.basename(args.infile)}", flush=True)
        for galaxy in sorted(by_galaxy):
            groups = [f[k] for k in by_galaxy[galaxy]]
            pcts, nused, npts = combined_percentiles(groups, qty)
            rows.append((galaxy, len(groups), nused, npts, pcts))
            print(f"  {galaxy}: {nused} clusters, {npts} points; {label} percentiles "
                  + " ".join(f"{p:+.3f}" for p in pcts)
                  + f" (elapsed {time.time() - t0:.0f} s)", flush=True)
        library = str(f.attrs.get("library", "cluster_slug default library"))
        prior = str(f.attrs.get("prior", ""))

    cols = ["galaxy", "ncluster"] + [f"{label}_{p:02d}" for p in PERCENTILES]
    with open(out, "w") as fo:
        fo.write(f"# Distribution of {args.quantity} ({label}) in each galaxy: percentiles of the "
                 "average of its clusters' marginal posterior PDFs, each cluster weighted equally\n")
        fo.write(f"# Samples: {osp.basename(args.infile)}; library: {library}\n")
        if prior:
            fo.write(f"# Prior: {prior}\n")
        fo.write("# ncluster = number of clusters averaged (those with non-empty posterior samples)\n")
        fo.write("#" + f"{cols[0]:>9s}{cols[1]:>9s}" + "".join(f"{c:>9s}" for c in cols[2:]) + "\n")
        for galaxy, _n, nused, _npts, pcts in rows:
            fo.write(f"{galaxy:>10s}{nused:9d}" + "".join(f"{p:9.3f}" for p in pcts) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
