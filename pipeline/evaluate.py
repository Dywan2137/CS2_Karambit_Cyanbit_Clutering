"""Evaluation without ground truth (internal, stability, shift test) and with
hand-labelled look-alike pairs."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score

import config as cfg
from pipeline.cluster import cluster_one


# ---- internal ---------------------------------------------------------------

def internal_metrics(z, labels):
    labels = np.asarray(labels)
    clustered = labels >= 0
    ids, sizes = np.unique(labels[clustered], return_counts=True)
    out = {
        "n_clusters": int(len(ids)),
        "noise_frac": float(1 - clustered.mean()),
        "largest_cluster": int(sizes.max()) if len(sizes) else 0,
        "median_size": float(np.median(sizes)) if len(sizes) else 0.0,
        "silhouette": np.nan,
    }
    if len(ids) >= 2 and clustered.sum() > len(ids):
        out["silhouette"] = float(silhouette_score(z[clustered], labels[clustered], metric="cosine"))
    return out


# ---- stability --------------------------------------------------------------

def _noise_unique(labels):
    """Give every noise point its own id so ARI doesn't treat noise as one cluster."""
    labels = np.asarray(labels).copy()
    noise = labels < 0
    labels[noise] = labels.max() + 1 + np.arange(noise.sum())
    return labels


def stability(kind, params, z, u, base_labels, runs=cfg.STABILITY_RUNS,
              frac=cfg.STABILITY_FRAC, seed=cfg.SEED):
    """Mean ARI between the full-data clustering and clusterings of subsamples."""
    rng = np.random.default_rng(seed)
    n = len(z)
    scores = []
    for _ in range(runs):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        sub, _ = cluster_one(kind, params, z[idx], u[idx])
        scores.append(adjusted_rand_score(_noise_unique(base_labels[idx]), _noise_unique(sub)))
    return float(np.mean(scores))


# ---- synthetic shift test ---------------------------------------------------

def shift_test(z_gallery, z_query, query_idx, ks=(1, 5)):
    """Recall@k: is the original seed among the k nearest gallery items of its
    shifted copy?"""
    sim = z_query @ z_gallery.T
    order = np.argsort(-sim, axis=1)
    out = {}
    for k in ks:
        hits = [qi in order[r, :k] for r, qi in enumerate(query_idx)]
        out[f"shift_recall@{k}"] = float(np.mean(hits))
    return out


# ---- labelled pairs ---------------------------------------------------------

def load_pairs(path=cfg.PAIRS_CSV):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["same"].isin([0, 1])]
    return df if len(df) else None


def pair_metrics(z, labels, seeds, pairs):
    """AUC of cosine similarity on labelled pairs + pairwise cluster precision/recall."""
    if pairs is None:
        return {}
    pos = {s: i for i, s in enumerate(seeds)}
    pairs = pairs[pairs["seed_a"].isin(pos) & pairs["seed_b"].isin(pos)]
    a = pairs["seed_a"].map(pos).to_numpy()
    b = pairs["seed_b"].map(pos).to_numpy()
    y = pairs["same"].to_numpy()
    out = {"n_pairs": int(len(y))}
    if len(np.unique(y)) == 2:
        out["pair_auc"] = float(roc_auc_score(y, (z[a] * z[b]).sum(1)))
    if labels is not None:
        labels = np.asarray(labels)
        pred = (labels[a] == labels[b]) & (labels[a] >= 0)
        tp = int((pred & (y == 1)).sum())
        out["pair_precision"] = tp / pred.sum() if pred.sum() else np.nan
        out["pair_recall"] = tp / (y == 1).sum() if (y == 1).sum() else np.nan
    return out
