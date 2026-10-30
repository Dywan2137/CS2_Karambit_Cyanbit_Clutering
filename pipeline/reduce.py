"""Turn raw embeddings into a space where seed-specific differences dominate.

Raw embeddings of 1000 renders of the same knife all point in nearly the same
direction. Subtracting the dataset mean removes that shared "it's a karambit"
component; L2 + PCA then keep the directions along which patterns differ.
"""

import numpy as np
from sklearn.decomposition import PCA

import config as cfg


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def mean_cosine(x):
    """Mean off-diagonal cosine similarity: ~1 means everything looks the same."""
    z = l2(x)
    s = z @ z.T
    n = len(z)
    return float((s.sum() - n) / (n * (n - 1)))


class Reducer:
    """center -> L2 -> PCA(whiten) -> L2. Fit once, reuse for new (e.g. shifted) images."""

    def __init__(self, variance=cfg.PCA_VARIANCE, max_dims=cfg.PCA_MAX_DIMS, whiten=True):
        self.variance, self.max_dims, self.whiten = variance, max_dims, whiten

    def fit(self, x):
        self.mu = x.mean(axis=0, keepdims=True)
        z = l2(x - self.mu)
        full = PCA(n_components=min(len(z), z.shape[1]), random_state=cfg.SEED).fit(z)
        k = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), self.variance) + 1)
        self.k = max(2, min(k, self.max_dims))
        self.pca = PCA(n_components=self.k, whiten=self.whiten, random_state=cfg.SEED).fit(z)
        return self

    def transform(self, x):
        return l2(self.pca.transform(l2(x - self.mu)))

    def fit_transform(self, x):
        return self.fit(x).transform(x)


def umap_reduce(z, dims=cfg.UMAP_DIMS, neighbors=cfg.UMAP_NEIGHBORS, seed=cfg.SEED):
    import umap
    return umap.UMAP(n_components=dims, n_neighbors=neighbors, min_dist=0.0,
                     metric="cosine", random_state=seed).fit_transform(z)
