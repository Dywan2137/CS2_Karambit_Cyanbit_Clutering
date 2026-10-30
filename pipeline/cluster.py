"""Clustering and nearest-neighbour look-alikes.

Label convention for every method: -1 = noise / unique pattern (no look-alike).
Agglomerative singletons are mapped to -1 too, so the methods are comparable.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import HDBSCAN

import config as cfg


def knn_table(z, seeds, k=cfg.KNN_K):
    """Top-k most similar seeds for every seed (cosine on L2-normalised z)."""
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argsort(-sim, axis=1)[:, :k]
    rows = [
        {"seed": seeds[i], "rank": r + 1, "neighbor": seeds[j], "similarity": float(sim[i, j])}
        for i in range(len(seeds)) for r, j in enumerate(idx[i])
    ]
    return pd.DataFrame(rows)


def run_hdbscan(u, min_cluster_size, min_samples):
    hd = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                 metric="euclidean", cluster_selection_method="leaf")
    labels = hd.fit_predict(u)
    return labels, hd.probabilities_


def agglo_linkage(z, method=cfg.AGGLO_LINKAGE):
    return linkage(z, method=method, metric="cosine")


def agglo_cut(Z, threshold):
    labels = fcluster(Z, t=threshold, criterion="distance") - 1
    return singletons_to_noise(labels)


def singletons_to_noise(labels):
    labels = np.asarray(labels).copy()
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    labels[np.isin(labels, ids[counts < 2])] = -1
    # relabel 0..n-1 in order of size (largest first)
    keep = [i for i, _ in sorted(zip(*np.unique(labels[labels >= 0], return_counts=True)),
                                 key=lambda t: -t[1])]
    remap = {old: new for new, old in enumerate(keep)}
    return np.array([remap.get(l, -1) for l in labels])


def method_grid():
    """All (method_name, kind, params) combinations from config."""
    grid = [(f"hdbscan_mcs{m}_ms{s}", "hdbscan", {"min_cluster_size": m, "min_samples": s})
            for m in cfg.HDBSCAN_MIN_CLUSTER_SIZES for s in cfg.HDBSCAN_MIN_SAMPLES]
    grid += [(f"agglo_t{t}", "agglo", {"threshold": t}) for t in cfg.AGGLO_THRESHOLDS]
    return grid


def cluster_one(kind, params, z, u, Z=None):
    """Labels (and probabilities, if any) for one method on one embedding."""
    if kind == "hdbscan":
        labels, probs = run_hdbscan(u, **params)
        return singletons_to_noise(labels), probs
    Z = agglo_linkage(z) if Z is None else Z
    return agglo_cut(Z, params["threshold"]), np.ones(len(z))
