"""
Reconstruct cluster_slug posterior KDEs from the HDF5 file written by
phangs_cluster_slug.py.

Each cluster's group in that file holds the "points" and "weights"
returned by conditional_sample, stored as float32, and the posterior
kernel's inverse squared bandwidth matrix as its "h2Inv" attribute; the
posterior is the KDE over those points with those weights and that
bandwidth.

Example
-------
>>> from phangs_kde import load_kde
>>> kde = load_kde(1)                 # a single KDE
>>> kdes = load_kde(range(1, 6))      # a list of 5 KDEs
>>> kde.percentile("mass", [16, 50, 84])
"""

import os.path as osp
import sys
from collections.abc import Iterable

import h5py
import numpy as np

# Make slugpy importable from the slug3 checkout
SLUG_DIR = osp.expanduser("~/Dropbox/Projects/slug3")
if SLUG_DIR not in sys.path:
    sys.path.insert(0, SLUG_DIR)
from slugpy.kde import KDE  # noqa: E402

# Default data file: the one phangs_cluster_slug.py writes by default
DEFAULT_FILE = osp.join(osp.dirname(osp.abspath(__file__)), "phangs_cluster_slug_samples.h5")


def _kde_from_group(grp, names):
    """
    Build the KDE stored in one cluster's group.

    Parameters
    ----------
    grp : h5py.Group
        The cluster's group.
    names : list of str
        The names of the inference quantities.

    Returns
    -------
    KDE
        The posterior KDE.

    Raises
    ------
    ValueError
        If the group's sample is empty or has no positive weight.
    """
    # KDE needs C-contiguous, writeable float64 arrays (it reorders them
    # in place), so convert from the stored float32
    pts = np.ascontiguousarray(grp["points"][()], dtype=np.float64)
    wgts = np.ascontiguousarray(grp["weights"][()], dtype=np.float64)
    if len(wgts) == 0 or not np.any(wgts > 0):
        raise ValueError(f"cluster {grp.name.lstrip('/')} has an empty posterior sample")
    h2inv = np.asarray(grp.attrs["h2Inv"], dtype=np.float64)
    return KDE(pts, h2Inv=h2inv, w=wgts, names=names)


def load_kde(index, filename=None):
    """
    Reconstruct the posterior KDE of one or more clusters.

    Parameters
    ----------
    index : int or iterable of int
        The PHANGS catalog INDEX of the cluster, or of each of several
        clusters.
    filename : str, optional
        The HDF5 file written by phangs_cluster_slug.py (default:
        phangs_cluster_slug_samples.h5 in this file's directory).

    Returns
    -------
    KDE or list of KDE
        The posterior KDE over (log10 M/Msun, log10 age/yr, A_V, [Fe/H]),
        with names "mass", "age", "A_V", "metallicity"; a single KDE for
        a single index, or a list of one per index, in the order given.

    Raises
    ------
    KeyError
        If a requested index is not in the file.
    ValueError
        If a requested cluster's stored sample is empty.
    """
    filename = DEFAULT_FILE if filename is None else filename
    single = np.ndim(index) == 0 and not isinstance(index, Iterable)
    indices = [index] if single else list(index)
    with h5py.File(filename, "r") as f:
        names = [str(n) for n in f.attrs["quantities"]]
        kdes = []
        for idx in indices:
            key = str(int(idx))
            if key not in f:
                raise KeyError(f"cluster {key} is not in {filename}")
            kdes.append(_kde_from_group(f[key], names))
    return kdes[0] if single else kdes
