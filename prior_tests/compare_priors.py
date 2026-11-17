"""Compare the skip-25 prior-variant summaries with the fiducial full run."""
import os.path as osp
import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
QTY = ["logM", "logT", "AV", "FeH"]


def read(fname):
    with open(fname) as f:
        cols = [l for l in f if l.startswith("#")][-1][1:].split()
    d = np.genfromtxt(fname, names=cols, dtype=None, encoding=None)
    return d


full = read(osp.join(HERE, "..", "phangs_cluster_slug_summary.txt"))
fid = read(osp.join(HERE, "skip25_fiducial_summary.txt"))
pos = {i: k for k, i in enumerate(full["index"])}
sel = np.array([pos[i] for i in fid["index"]])
ref = full[sel]
numcols = [c for c in fid.dtype.names if c != "galaxy"]
diff = max(np.nanmax(np.abs(fid[c] - ref[c])) for c in numcols)
print(f"Fiducial skip-25 ({len(fid)} clusters) vs full run: max |difference| = {diff:.3g}\n")

print("Shift of each variant relative to fiducial: median of the 50th percentile shift "
      "[16th, 84th percentile of shift across clusters]; and median change in 16-84 width")
for name in ["alphaM-2.0", "alphaM-1.0", "alphaT0.0", "alphaT-1.0"]:
    v = read(osp.join(HERE, f"skip25_{name}_summary.txt"))
    assert np.all(v["index"] == fid["index"])
    print(f"\n{name}:")
    for q in QTY:
        s = v[f"{q}_50"] - fid[f"{q}_50"]
        w = (v[f"{q}_84"] - v[f"{q}_16"]) - (fid[f"{q}_84"] - fid[f"{q}_16"])
        lo, med, hi = np.percentile(s, [16, 50, 84])
        print(f"  {q:5s} median shift {med:+.3f} [{lo:+.3f}, {hi:+.3f}], "
              f"|shift|>0.1 for {np.mean(np.abs(s) > 0.1):5.1%}; "
              f"width change {np.median(w):+.3f}")
