"""
Split the cluster_slug posterior sample file written by
phangs_cluster_slug.py into one file per galaxy.

Each output file is named after the input, with the galaxy inserted after
its leading "phangs_" (e.g. phangs_cluster_slug_samples.h5 gives
phangs_GALAXY_cluster_slug_samples.h5, and
phangs_cluster_slug_hstlib_samples.h5 gives
phangs_GALAXY_cluster_slug_hstlib_samples.h5), and has the same layout as
the input: one group per cluster, named by its catalog index,
holding the same datasets and attributes, and the same root attributes,
plus a root attribute "galaxy". Groups are copied with HDF5's object copy,
so the compressed data are copied as they are, without decompressing them.
Since each file has the same layout as the input, phangs_kde.load_kde can
read any of them with its filename argument.
"""

import argparse
import os.path as osp
import time

import h5py

HERE = osp.dirname(osp.abspath(__file__))


def galaxy_filename(infile, galaxy):
    """
    Name of a galaxy's output file.

    Parameters
    ----------
    infile : str
        The input sample file.
    galaxy : str
        The galaxy.

    Returns
    -------
    str
        The input's base name with the galaxy inserted after its leading
        "phangs_" (or prepended, if it has none).
    """
    base = osp.basename(infile)
    if base.startswith("phangs_"):
        return f"phangs_{galaxy}_{base[len('phangs_'):]}"
    return f"{galaxy}_{base}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("infile", nargs="?",
                        default=osp.join(HERE, "phangs_cluster_slug_samples.h5"),
                        help="input sample file (default: phangs_cluster_slug_samples.h5 next "
                        "to this script)")
    parser.add_argument("--out-dir", default=None,
                        help="directory for the per-galaxy files (default: that of the input)")
    args = parser.parse_args()
    out_dir = args.out_dir or osp.dirname(osp.abspath(args.infile))

    t0 = time.time()
    with h5py.File(args.infile, "r") as fin:
        # Group the clusters by galaxy, keeping the input order
        by_galaxy = {}
        for name in fin:
            by_galaxy.setdefault(str(fin[name].attrs["galaxy"]), []).append(name)
        print(f"{sum(len(v) for v in by_galaxy.values())} clusters in {len(by_galaxy)} galaxies",
              flush=True)

        for galaxy in sorted(by_galaxy):
            outfile = osp.join(out_dir, galaxy_filename(args.infile, galaxy))
            with h5py.File(outfile, "w") as fout:
                for key, val in fin.attrs.items():
                    fout.attrs[key] = val
                fout.attrs["galaxy"] = galaxy
                for name in by_galaxy[galaxy]:
                    fin.copy(fin[name], fout, name=name)
            print(f"  {galaxy}: {len(by_galaxy[galaxy])} clusters -> {osp.basename(outfile)} "
                  f"({osp.getsize(outfile) / 1e9:.2f} GB; elapsed {time.time() - t0:.0f} s)",
                  flush=True)


if __name__ == "__main__":
    main()
