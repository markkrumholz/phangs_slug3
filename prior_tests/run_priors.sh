#!/bin/zsh
# Rerun phangs_cluster_slug.py on every 25th cluster with varied priors
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/astropy/bin/python
run() {
    name=$1; shift
    echo "=== $name: $* ($(date))"
    $PY ../phangs_cluster_slug.py --ncluster 0 --skip 25 "$@" \
        --out-text skip25_${name}_summary.txt --out-hdf5 skip25_${name}_samples.h5 \
        > skip25_${name}_log.txt 2>&1 || { echo "FAILED: $name"; exit 1; }
    tail -1 skip25_${name}_log.txt
}
run fiducial
run alphaM-2.0 --alpha-m -2
run alphaM-1.0 --alpha-m -1
run alphaT0.0 --alpha-t 0
run alphaT-1.0 --alpha-t -1
echo "All done $(date)"
