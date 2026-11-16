"""
Infer star cluster masses, ages, extinctions, and metallicities for the
PHANGS-HST DR5 machine-classified class 1+2 cluster catalog with
cluster_slug.

For each cluster this script

1. converts the catalog's apparent Vega magnitudes to absolute magnitudes
   using the catalog's distance modulus;
2. finds the 5 nearest neighbors of its photometry in the cluster_slug
   default library twice: with distances in units of the photometric
   errors ("errors" norm), and in mag ("Euclidean" norm);
3. draws the sample points and weights of its posterior with
   conditional_sample;
4. builds a KDE from those points, marginalizes it down to each physical
   quantity in turn, and computes percentiles of each marginal PDF.

Outputs, written to the directory holding this script:

- a plain text summary, one line per cluster: index, galaxy, distance to
  the 5th nearest neighbor in each of the two norms, and the
  5/16/50/84/95th percentiles of log
  mass, log age, A_V, and [Fe/H];
- an HDF5 file with one group per cluster, named by its catalog index,
  holding the "points" and "weights" returned by conditional_sample, and
  the distances to its 5 nearest neighbors in each norm ("dist_err" and
  "dist_euc").

The prior is dN/dM ~ M^-1.5, and dN/dT ~ T^0 for T < 10^6.5 yr and
T^-0.5 for older ages (continuous at 10^6.5 yr); it is flat in A_V and
[Fe/H] (i.e. the library's own uniform distributions of these are kept).

Each band's library filter is chosen per cluster from the instrument
PHANGS_OBS_SUMMARY gives for it (WFC3/UVIS or ACS/WFC). Clusters with
photometry missing (NaN) in some bands are fit using only the bands they
have. Each distinct combination of filters gets its own filter set.
"""

import argparse
import os.path as osp
import sys
import time

import astropy.units as u
import h5py
import numpy as np
from astropy.io import fits

# Make slugpy importable from the slug3 checkout
SLUG_DIR = osp.expanduser("~/Dropbox/Projects/slug3")
sys.path.insert(0, SLUG_DIR)
from slugpy import cluster_slug  # noqa: E402
from slugpy.kde import KDE  # noqa: E402

# Paths
HERE = osp.dirname(osp.abspath(__file__))
CATALOG = osp.join(HERE, "hlsp_phangs-cat_hst_acs-uvis_dr5-targets_v1_machine-class1+2-primary.fits")

# Map from the catalog's band names, and the instrument each band was
# observed with, to the library's filter names. The instrument comes from
# the first five fields of PHANGS_OBS_SUMMARY ("UVIS" for WFC3/UVIS,
# "WFC" for ACS/WFC), in the order of BANDS. B is WFC3 F438W or ACS
# F435W; V and I are F555W and F814W on either instrument (despite the
# README, some galaxies have V and I from ACS); NUV and U are always WFC3.
BANDS = {"NUV": {"UVIS": "HST.WFC3_UVIS1.F275W"},
         "U": {"UVIS": "HST.WFC3_UVIS1.F336W"},
         "B": {"UVIS": "HST.WFC3_UVIS1.F438W", "WFC": "HST.ACS_WFC.F435W"},
         "V": {"UVIS": "HST.WFC3_UVIS1.F555W", "WFC": "HST.ACS_WFC.F555W"},
         "I": {"UVIS": "HST.WFC3_UVIS1.F814W", "WFC": "HST.ACS_WFC.F814W"}}

# Number of nearest neighbors to find
N_NEIGHBOR = 5

# Percentiles to compute, and the inference quantities with the labels
# used for them in the text output
PERCENTILES = [5, 16, 50, 84, 95]
QUANTITY_LABELS = {"mass": "logM", "age": "logT", "A_V": "AV", "metallicity": "FeH"}

# Age at which the age prior breaks from T^0 to T^-0.5
T_BREAK = 10.0**6.5  # yr


def prior(mass, age, av, feh):
    """
    Prior probability density of a library cluster's properties.

    Parameters
    ----------
    mass : astropy.units.Quantity
        Cluster masses.
    age : astropy.units.Quantity
        Cluster ages.
    av : astropy.units.Quantity or None
        Visual extinctions (unused: the prior is flat in A_V).
    feh : numpy.ndarray
        Metallicities [Fe/H] (unused: the prior is flat in [Fe/H]).

    Returns
    -------
    numpy.ndarray
        The prior density per unit (linear) mass and age, up to a
        normalization: M^-1.5 times 1 for T < 10^6.5 yr, or
        (T / 10^6.5 yr)^-0.5 for older T.
    """
    m = mass.to_value(u.Msun)
    t = age.to_value(u.yr)
    return m**-1.5 * np.where(t < T_BREAK, 1.0, (t / T_BREAK)**-0.5)


def read_catalog(filename, ncluster=None):
    """
    Read the PHANGS catalog, converting photometry to absolute magnitudes.

    Parameters
    ----------
    filename : str
        The catalog FITS file.
    ncluster : int, optional
        Read only the first ncluster clusters (default: all).

    Returns
    -------
    index : numpy.ndarray of int, shape (N,)
        Each cluster's catalog index.
    galaxy : numpy.ndarray of str, shape (N,)
        Each cluster's galaxy.
    phot : numpy.ndarray of float, shape (N, 5)
        Absolute Vega magnitudes in the bands of BANDS, NaN where
        missing.
    phot_err : numpy.ndarray of float, shape (N, 5)
        Their errors, in mag.
    filters : numpy.ndarray of str, shape (N, 5)
        The library filter name of each band for each cluster, chosen
        from the instrument PHANGS_OBS_SUMMARY gives for it.

    Raises
    ------
    ValueError
        If PHANGS_OBS_SUMMARY gives an instrument for a band that BANDS
        has no filter for.
    """
    with fits.open(filename) as hdul:
        data = hdul[1].data
        if ncluster is not None:
            data = data[:ncluster]
        index = np.asarray(data["INDEX"]).astype(int)
        galaxy = np.char.strip(np.asarray(data["PHANGS_GALAXY"]).astype(str))
        dist_mod = np.asarray(data["PHANGS_DM"], dtype=float)
        phot = np.column_stack([data[f"PHANGS_{b}_VEGA"] for b in BANDS]).astype(float) \
            - dist_mod[:, None]
        phot_err = np.column_stack([data[f"PHANGS_{b}_VEGA_ERR"] for b in BANDS]).astype(float)
        obs_summary = np.asarray(data["PHANGS_OBS_SUMMARY"]).astype(str)
    filters = np.empty(phot.shape, dtype=object)
    for i, summary in enumerate(obs_summary):
        instruments = [field.strip() for field in summary.split(";")[:len(BANDS)]]
        for j, (band, inst) in enumerate(zip(BANDS, instruments)):
            if inst not in BANDS[band]:
                raise ValueError(f"cluster {index[i]}: no library filter for band {band} "
                                 f"observed with instrument '{inst}'")
            filters[i, j] = BANDS[band][inst]
    return index, galaxy, phot, phot_err, filters


def marginal_percentiles(pts, wgts, h2inv, names):
    """
    Percentiles of each one-dimensional marginal of a posterior sample.

    Parameters
    ----------
    pts : numpy.ndarray of float, shape (K, nqty)
        Posterior sample points from conditional_sample.
    wgts : numpy.ndarray of float, shape (K,)
        Their weights.
    h2inv : numpy.ndarray of float, shape (nqty, nqty)
        The posterior kernel's inverse squared bandwidth matrix.
    names : list of str
        The names of the nqty quantities.

    Returns
    -------
    numpy.ndarray of float, shape (nqty, len(PERCENTILES))
        The percentiles of each quantity's marginal PDF; NaN if the
        sample is empty.
    """
    out = np.full((len(names), len(PERCENTILES)), np.nan)
    if len(wgts) == 0 or not np.any(wgts > 0):
        return out
    # The KDE reorders its data set and weights in place, so give it copies
    kde = KDE(np.ascontiguousarray(pts, dtype=np.float64).copy(), h2Inv=h2inv,
              w=np.ascontiguousarray(wgts, dtype=np.float64).copy(), names=list(names))
    for i, name in enumerate(names):
        marginal_kde = kde.marginalize([n for n in names if n != name])
        out[i] = marginal_kde.percentile(0, PERCENTILES)
    return out


def write_text(filename, index, galaxy, d5_err, d5_euc, pcts, names):
    """
    Write the plain text summary.

    Parameters
    ----------
    filename : str
        Output file name.
    index : numpy.ndarray of int, shape (N,)
        Catalog indices.
    galaxy : numpy.ndarray of str, shape (N,)
        Galaxy names.
    d5_err : numpy.ndarray of float, shape (N,)
        Distance to each cluster's 5th nearest neighbor with the "errors"
        norm, in units of the photometric errors.
    d5_euc : numpy.ndarray of float, shape (N,)
        Distance to each cluster's 5th nearest neighbor with the
        Euclidean norm, in mag.
    pcts : numpy.ndarray of float, shape (N, nqty, len(PERCENTILES))
        Marginal percentiles of each quantity.
    names : list of str
        The inference quantities, in the order of pcts' second axis.
    """
    cols = ["index", "galaxy", "d5_err", "d5_euc"] + [f"{QUANTITY_LABELS[n]}_{p:02d}"
                                                       for n in names for p in PERCENTILES]
    widths = [7, 9, 8, 8] + [8] * (len(cols) - 4)
    with open(filename, "w") as f:
        f.write("# cluster_slug posteriors for PHANGS-HST DR5 machine-classified class 1+2 clusters\n")
        f.write("# Columns: index = PHANGS catalog INDEX; galaxy = PHANGS_GALAXY;\n")
        f.write(f"#   d5_err = distance to the {N_NEIGHBOR}th nearest library neighbor with the errors\n")
        f.write("#        norm, in units of the photometric errors (RMS over bands of difference / error);\n")
        f.write(f"#   d5_euc = distance to the {N_NEIGHBOR}th nearest library neighbor with the Euclidean\n")
        f.write("#        norm, in mag (RMS over bands of the magnitude difference);\n")
        f.write("#   X_PP = PPth percentile of the marginal posterior of X, where\n")
        f.write("#        logM = log10(M / Msun), logT = log10(age / yr), AV = A_V [mag], FeH = [Fe/H]\n")
        f.write("#" + "".join(f"{c:>{w}s}" for c, w in zip(cols, widths))[1:] + "\n")
        for i in range(len(index)):
            line = (f"{index[i]:{widths[0]}d}{galaxy[i]:>{widths[1]}s}"
                    f"{d5_err[i]:{widths[2]}.3f}{d5_euc[i]:{widths[3]}.3f}")
            line += "".join(f"{v:8.3f}" for v in pcts[i].ravel())
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ncluster", type=int, default=20,
                        help="process only the first NCLUSTER clusters (default 20); 0 for all")
    parser.add_argument("--mass-tol", type=float, default=1e-3,
                        help="massTol for conditional_sample (default 1e-3)")
    parser.add_argument("--out-text", default=osp.join(HERE, "phangs_cluster_slug_summary.txt"),
                        help="plain text summary output file")
    parser.add_argument("--out-hdf5", default=osp.join(HERE, "phangs_cluster_slug_samples.h5"),
                        help="HDF5 conditional sample output file")
    args = parser.parse_args()

    # Read the data
    t0 = time.time()
    index, galaxy, phot, phot_err, band_filters = read_catalog(CATALOG, args.ncluster or None)
    ncl = len(index)
    print(f"Read {ncl} clusters from {osp.basename(CATALOG)}")

    # Load the default library, with our prior
    cs = cluster_slug(priors=prior)
    names = list(cs.inference_quantities)
    print(f"Loaded library ({cs.size} clusters) in {time.time() - t0:.1f} s; "
          f"inference quantities {names}")

    # Group clusters by their filter set: the filters of the bands they
    # have photometry in. Since every band's filters are distinct, the
    # filter set also fixes which bands those are.
    have = np.isfinite(phot) & np.isfinite(phot_err) & (phot_err > 0)
    groups = {}
    for j in range(ncl):
        groups.setdefault(tuple(band_filters[j, have[j]]), []).append(j)

    # Outputs
    dist_err = np.full((ncl, N_NEIGHBOR), np.nan)
    dist_euc = np.full((ncl, N_NEIGHBOR), np.nan)
    pcts = np.full((ncl, len(names), len(PERCENTILES)), np.nan)
    samples = [None] * ncl
    h2inv_by_group = {}

    for key, member_list in groups.items():
        members = np.array(member_list)
        pattern = have[members[0]]
        filters = [str(f) for f in key]
        if len(filters) == 0:
            print(f"  {len(members)} clusters have no usable photometry; skipping them")
            continue
        t1 = time.time()
        cs.add_filterset(filters)
        kde = cs.kdes[tuple(filters)]
        # Inverse squared bandwidth of the posterior: the default
        # bandwidth is diagonal and the errors only affect the photometric
        # dimensions, so it is the physical block of the joint one
        phys = [kde.names.index(n) for n in names]
        h2inv = np.ascontiguousarray(kde.h2Inv[np.ix_(phys, phys)])
        h2inv_by_group[tuple(filters)] = h2inv
        print(f"  Filter set {filters}: {len(members)} clusters "
              f"(KDE built in {time.time() - t1:.1f} s)")

        p = np.ascontiguousarray(phot[np.ix_(members, pattern)])
        e = np.ascontiguousarray(phot_err[np.ix_(members, pattern)])

        # Nearest neighbors, in units of the photometric errors and, with
        # the Euclidean norm, in mag
        t1 = time.time()
        dist, _ = cs.neighbors(p, N_NEIGHBOR, filterset=filters, errors=e, norm="errors")
        dist_err[members] = np.asarray(dist)
        dist, _ = cs.neighbors(p, N_NEIGHBOR, filterset=filters, norm="Euclidean")
        dist = dist.to_value(u.mag) if isinstance(dist, u.Quantity) else np.asarray(dist)
        dist_euc[members] = dist
        print(f"    neighbors: {time.time() - t1:.1f} s")

        # Posterior samples
        t1 = time.time()
        res = cs.conditional_sample(p, massTol=args.mass_tol, filterset=filters, errors=e)
        print(f"    conditional_sample: {time.time() - t1:.1f} s")

        # Marginal percentiles
        t1 = time.time()
        for j, (pts, wgts, _rows) in zip(members, res):
            samples[j] = (pts, wgts, filters)
            pcts[j] = marginal_percentiles(pts, wgts, h2inv, names)
        print(f"    marginal percentiles: {time.time() - t1:.1f} s")

        # Drop this filter set's KDE, which spans the whole library, so
        # that only one is held in memory at a time
        cs.remove_filterset(filters)

    # Write the plain text summary
    write_text(args.out_text, index, galaxy, dist_err[:, N_NEIGHBOR - 1],
               dist_euc[:, N_NEIGHBOR - 1], pcts, names)
    print(f"Wrote {args.out_text}")

    # Write the conditional samples
    with h5py.File(args.out_hdf5, "w") as f:
        f.attrs["catalog"] = osp.basename(CATALOG)
        f.attrs["quantities"] = names
        f.attrs["description"] = (
            "One group per cluster, named by its PHANGS catalog INDEX. 'points' (K x nqty) and "
            "'weights' (K) are the output of cluster_slug.conditional_sample; the columns of "
            "'points' are the quantities in the 'quantities' attribute (log10 M/Msun, "
            "log10 age/yr, A_V [mag], [Fe/H]). Both are stored as gzip-compressed float32. "
            "Weights are not normalized. A KDE of the posterior is "
            "KDE(points[()].astype(np.float64), h2Inv=h2Inv, w=weights[()].astype(np.float64)), "
            "with h2Inv the group attribute (KDE requires float64 inputs). 'dist_err' and "
            "'dist_euc' are the distances to the cluster's nearest library neighbors, nearest "
            "first, from cluster_slug.neighbors with the 'errors' norm (in units of the "
            "photometric errors) and the 'Euclidean' norm (in mag), respectively.")
        f.attrs["mass_tol"] = args.mass_tol
        for j in range(ncl):
            if samples[j] is None:
                continue
            pts, wgts, filters = samples[j]
            grp = f.create_group(str(index[j]))
            # Stored as float32, gzip-compressed (with the shuffle filter,
            # which improves gzip's compression of floating-point data);
            # HDF5 cannot chunk, and so cannot compress, an empty dataset
            comp = dict(compression="gzip", compression_opts=4, shuffle=True) \
                if len(wgts) > 0 else {}
            grp.create_dataset("points", data=pts.astype(np.float32), **comp)
            grp.create_dataset("weights", data=wgts.astype(np.float32), **comp)
            grp.create_dataset("dist_err", data=dist_err[j])
            grp.create_dataset("dist_euc", data=dist_euc[j])
            grp.attrs["galaxy"] = str(galaxy[j])
            grp.attrs["filters"] = filters
            grp.attrs["h2Inv"] = h2inv_by_group[tuple(filters)]
    print(f"Wrote {args.out_hdf5}")
    print(f"Total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
