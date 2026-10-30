"""Pipeline CLI.

    python run.py scrape                     # download all 1000 renders
    python run.py prep                       # blade mask + crops
    python run.py model-inputs [--models ...]  # save crops at each model's input size
    python run.py embed [--models dinov2 clip resnet50 colorhist] [--no-tta]
    python run.py cluster [--models ...]     # HDBSCAN / agglomerative / kNN + stability
    python run.py eval [--models ...]        # shift test, label scores, pick best, HTML report
    python run.py label [--model dinov2]     # hand-label look-alike pairs
    python run.py all                        # prep -> embed -> cluster -> eval
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd

import config as cfg


def _models(args):
    return args.models or cfg.MODELS


def cmd_scrape(args):
    import scraper
    scraper.main([])


def cmd_prep(args):
    from pipeline.data import load_dataset
    from pipeline.preprocess import preprocess
    seeds, paths = load_dataset()
    preprocess(seeds, paths)


def cmd_model_inputs(args):
    from pipeline.embed import save_model_inputs
    from pipeline.preprocess import load_processed
    seeds, paths, mask, _ = load_processed()
    save_model_inputs(_models(args), paths, seeds, mask.shape)


def cmd_embed(args):
    from pipeline.embed import build_extractor, embed_paths, embedding_path
    from pipeline.preprocess import load_processed
    from pipeline.reduce import Reducer, mean_cosine
    seeds, paths, mask, meta = load_processed()
    mean_crop = np.load(cfg.PROCESSED_DIR / "mean_crop.npy")
    cfg.EMB_DIR.mkdir(parents=True, exist_ok=True)
    shifts = cfg.TTA_SHIFTS if args.tta else [(0, 0)]
    for name in _models(args):
        ext = build_extractor(name, mask, mean_crop)
        X = embed_paths(ext, paths, meta["fill"], shifts=shifts)
        np.save(embedding_path(name, args.tta), X)
        z = Reducer().fit_transform(X)
        print(f"  {name}: {X.shape}  mean cosine raw {mean_cosine(X):.3f} -> "
              f"centred+PCA {mean_cosine(z):.3f} ({z.shape[1]} dims)")


def cmd_cluster(args):
    from pipeline.cluster import agglo_linkage, cluster_one, knn_table, method_grid
    from pipeline.embed import load_embeddings
    from pipeline.evaluate import internal_metrics, stability
    from pipeline.preprocess import load_processed
    from pipeline.reduce import Reducer, umap_reduce

    seeds, *_ = load_processed()
    rows = []
    for name in _models(args):
        out = cfg.RESULTS_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        X = load_embeddings(name, args.tta)
        reducer = Reducer().fit(X)
        z = reducer.transform(X)
        u = umap_reduce(z)
        xy = umap_reduce(z, dims=2)
        joblib.dump(reducer, out / "reducer.joblib")
        np.save(out / "z.npy", z); np.save(out / "umap.npy", u); np.save(out / "xy.npy", xy)
        knn_table(z, seeds).to_csv(out / "neighbors.csv", index=False)

        Z = agglo_linkage(z)
        for method, kind, params in method_grid():
            labels, probs = cluster_one(kind, params, z, u, Z)
            d = out / method
            d.mkdir(exist_ok=True)
            pd.DataFrame({"seed": seeds, "cluster": labels, "prob": probs}).to_csv(
                d / "labels.csv", index=False)
            m = {"model": name, "method": method, **internal_metrics(z, labels)}
            m["stability_ari"] = stability(kind, params, z, u, labels)
            rows.append(m)
            print(f"  {name:10s} {method:22s} clusters {m['n_clusters']:4d}  "
                  f"noise {m['noise_frac']:.2f}  sil {m['silhouette']:.3f}  "
                  f"ARI {m['stability_ari']:.3f}")
    _merge_csv(cfg.RESULTS_DIR / "cluster_metrics.csv", pd.DataFrame(rows), ["model", "method"])


def cmd_eval(args):
    from pipeline.embed import build_extractor, embed_images, shift_image
    from pipeline.evaluate import load_pairs, pair_metrics, shift_test
    from pipeline.preprocess import load_processed
    from pipeline.visualize import report
    from PIL import Image

    seeds, paths, mask, meta = load_processed()
    mean_crop = np.load(cfg.PROCESSED_DIR / "mean_crop.npy")
    pairs = load_pairs()
    models = _models(args)

    # ---- synthetic shift test, per model --------------------------------
    shift_rows = []
    if not args.skip_shift:
        rng = np.random.default_rng(cfg.SEED)
        q_idx = rng.choice(len(seeds), size=min(cfg.SHIFT_TEST_N, len(seeds)), replace=False)
        f = cfg.SHIFT_TEST_FRAC
        shifted = []
        for i in q_idx:
            im = np.asarray(Image.open(paths[i]).convert("RGB"))
            dx, dy = rng.uniform(-f, f, 2) * im.shape[1]
            shifted.append(shift_image(im, dx, dy, meta["fill"]))
        shifts = cfg.TTA_SHIFTS if args.tta else [(0, 0)]
        for name in models:
            out = cfg.RESULTS_DIR / name
            reducer = joblib.load(out / "reducer.joblib")
            ext = build_extractor(name, mask, mean_crop)
            Xq = np.concatenate([embed_images(ext, shifted[i:i + cfg.BATCH_SIZE], meta["fill"], shifts)
                                 for i in range(0, len(shifted), cfg.BATCH_SIZE)])
            r = shift_test(np.load(out / "z.npy"), reducer.transform(Xq), q_idx)
            shift_rows.append({"model": name, **r})
            print(f"  shift test {name:10s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))
        _merge_csv(cfg.RESULTS_DIR / "shift_metrics.csv", pd.DataFrame(shift_rows), ["model"])

    # ---- combine metrics + label-based scores ---------------------------
    cm = pd.read_csv(cfg.RESULTS_DIR / "cluster_metrics.csv")
    cm = cm[cm.model.isin(models)]
    sp = cfg.RESULTS_DIR / "shift_metrics.csv"
    if sp.exists():
        cm = cm.merge(pd.read_csv(sp), on="model", how="left")
    pm = []
    for _, r in cm.iterrows():
        z = np.load(cfg.RESULTS_DIR / r.model / "z.npy")
        labels = pd.read_csv(cfg.RESULTS_DIR / r.model / r.method / "labels.csv").cluster.to_numpy()
        pm.append(pair_metrics(z, labels, seeds, pairs))
    metrics = pd.concat([cm.reset_index(drop=True), pd.DataFrame(pm)], axis=1)
    if {"pair_precision", "pair_recall"} <= set(metrics.columns):
        p, rc = metrics.pair_precision, metrics.pair_recall
        metrics["pair_f1"] = 2 * p * rc / (p + rc)
    metrics.to_csv(cfg.RESULTS_DIR / "metrics.csv", index=False)

    best = (args.model, args.method) if args.model and args.method else pick_best(metrics, args.model)
    print(f"\n  Best: {best[0]} / {best[1]}")
    labels = pd.read_csv(cfg.RESULTS_DIR / best[0] / best[1] / "labels.csv").cluster.to_numpy()
    knn = pd.read_csv(cfg.RESULTS_DIR / best[0] / "neighbors.csv")
    xy = np.load(cfg.RESULTS_DIR / best[0] / "xy.npy")
    (cfg.RESULTS_DIR / "best.json").write_text(json.dumps({"model": best[0], "method": best[1]}))
    path = report(metrics.sort_values(["model", "method"]), best, seeds, labels, xy, knn)
    print(f"  Report: {path}")


def pick_best(metrics, model=None):
    """Model: best on human labels (pair AUC) if any, else on the shift test.
    Method: best pair F1 if labels exist, else most stable among non-degenerate runs."""
    m = metrics.copy()
    if model is None:
        key = "pair_auc" if "pair_auc" in m and m.pair_auc.notna().any() else (
            "shift_recall@1" if "shift_recall@1" in m else "stability_ari")
        model = m.groupby("model")[key].max().idxmax()
    m = m[m.model == model]
    if "pair_f1" in m and m.pair_f1.notna().any():
        return model, m.loc[m.pair_f1.idxmax(), "method"]
    ok = m[(m.n_clusters >= 5) & (m.noise_frac < 0.95) & (m.largest_cluster < 0.25 * (cfg.SEED_MAX - cfg.SEED_MIN + 1))]
    ok = ok if len(ok) else m
    return model, ok.loc[ok.stability_ari.idxmax(), "method"]


def cmd_label(args):
    from pipeline import label_tool
    from pipeline.preprocess import load_processed
    seeds, *_ = load_processed()
    model = args.model
    if model is None:
        bj = cfg.RESULTS_DIR / "best.json"
        model = json.loads(bj.read_text())["model"] if bj.exists() else "dinov2"
    knn = pd.read_csv(cfg.RESULTS_DIR / model / "neighbors.csv")
    label_tool.run(knn, seeds, n_queries=args.queries)


def cmd_all(args):
    cmd_prep(args); cmd_embed(args); cmd_cluster(args); cmd_eval(args)


def _merge_csv(path, new, keys):
    """Update rows for `keys` in an existing CSV, so running one model keeps the others."""
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old.set_index(keys).index.isin(new.set_index(keys).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["scrape", "prep", "model-inputs", "embed", "cluster", "eval", "label", "all"])
    ap.add_argument("--models", nargs="+", help=f"subset of {cfg.MODELS}")
    ap.add_argument("--no-tta", dest="tta", action="store_false", help="disable shift averaging")
    ap.add_argument("--model", help="label/eval: force this model")
    ap.add_argument("--method", help="eval: force this method (with --model)")
    ap.add_argument("--skip-shift", action="store_true", help="eval: skip the shift test")
    ap.add_argument("--queries", type=int, default=25, help="label: number of query seeds")
    args = ap.parse_args()
    globals()[f"cmd_{args.command.replace('-', '_')}"](args)


if __name__ == "__main__":
    main()
