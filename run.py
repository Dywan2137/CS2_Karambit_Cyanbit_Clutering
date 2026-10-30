"""Pipeline CLI. Every command takes --skin knife/finish[/variant]
(default karambit/gamma-doppler/phase-3); each skin has its own data/<skin>/ folder.

    python run.py process --skin butterfly/marble-fade   # everything, one go (see cmd_process)

    python run.py scrape                     # download all 1000 renders
    python run.py prep                       # blade mask + crops
    python run.py model-inputs [--models ...]  # save crops at each model's input size
    python run.py embed [--models dinov2 clip resnet50 colorhist] [--no-tta]
    python run.py cluster [--models ...]     # HDBSCAN / agglomerative / kNN + stability
    python run.py eval [--models ...]        # shift test, label scores, pick best, HTML report
    python run.py plot [--model dinov2] [--proj umap tsne pca] [--color-by groups|agglo_t0.3|none]
    python run.py similar [--model dinov2] [--self-tol]  # per-pattern lists of close look-alikes
    python run.py label [--model dinov2]     # hand-label pairs (calibrates `similar`)
    python run.py all                        # prep -> embed -> cluster -> eval
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd


def _skin_from_argv(argv):
    """config.py reads the skin when it is imported, so set it before that."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--skin")
    known, _ = pre.parse_known_args(argv)
    if known.skin:
        os.environ["SKIN"] = known.skin


_skin_from_argv(sys.argv[1:])
import config as cfg  # noqa: E402


def _models(args):
    return args.models or cfg.MODELS


def cmd_scrape(args):
    import scraper
    print(f"Downloading {cfg.SLUG} -> {cfg.RAW_DIR}")
    scraper.scrape(list(range(cfg.SEED_MIN, cfg.SEED_MAX + 1)))


def cmd_prep(args):
    from pipeline.data import load_dataset
    from pipeline.preprocess import preprocess
    seeds, paths = load_dataset()
    preprocess(seeds, paths)


def cmd_model_inputs(args):
    from pipeline.embed import save_model_inputs
    from pipeline.preprocess import load_processed
    seeds, paths, mask, _ = load_processed()
    save_model_inputs(_models(args), paths, seeds, mask.shape, cfg.MODEL_INPUTS_DIR)


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


def reduce_model(name, seeds, tta=True):
    """Fit centre+PCA on one model's embeddings and save reducer.joblib, z.npy and
    neighbors.csv to results/<model>/. Returns z."""
    from pipeline.cluster import knn_table
    from pipeline.embed import load_embeddings
    from pipeline.reduce import Reducer
    out = cfg.RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    X = load_embeddings(name, tta)
    reducer = Reducer().fit(X)
    z = reducer.transform(X)
    joblib.dump(reducer, out / "reducer.joblib")
    np.save(out / "z.npy", z)
    knn_table(z, seeds).to_csv(out / "neighbors.csv", index=False)
    return z


def cmd_cluster(args):
    from pipeline.cluster import agglo_linkage, cluster_one, method_grid
    from pipeline.evaluate import internal_metrics, stability
    from pipeline.preprocess import load_processed
    from pipeline.reduce import umap_reduce

    seeds, *_ = load_processed()
    rows = []
    for name in _models(args):
        out = cfg.RESULTS_DIR / name
        z = reduce_model(name, seeds, args.tta)
        u = umap_reduce(z)
        xy = umap_reduce(z, dims=2)
        np.save(out / "umap.npy", u); np.save(out / "xy.npy", xy)

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


def _model(args):
    if args.model:
        return args.model
    bj = cfg.RESULTS_DIR / "best.json"
    return json.loads(bj.read_text())["model"] if bj.exists() else "dinov2"


def cmd_similar(args):
    """Per-pattern lists whose length is decided by each seed's similarity curve."""
    from pipeline import similar
    from pipeline.evaluate import load_pairs
    from pipeline.preprocess import load_processed
    seeds, paths, mask, meta = load_processed()
    model = _model(args)
    out = cfg.RESULTS_DIR / model
    z = np.load(out / "z.npy")

    self_sims = None
    if args.self_tol or (out / "self_sim.npy").exists():
        self_sims = similar.self_tolerance(model, seeds, paths, mask, meta["fill"], out)
        print(f"  self tolerance: seeds vs their shifted copies, median {np.median(self_sims):.2f}, "
              f"worst-case p5 {np.percentile(self_sims.min(1), 5):.2f}")
    near = similar.near_levels(self_sims, len(seeds))

    floor, source = cfg.SIM_FLOOR, "config default"
    fitted, report = similar.calibrate_floor(z, seeds, load_pairs())
    if fitted is not None:
        print("  calibration from labels:\n" + report)
    else:
        print(f"  calibration skipped: {report}; using SIM_FLOOR={floor}")
    if fitted is not None:
        floor, source = round(fitted, 3), "fitted from labels"

    df = similar.similar_lists(z, seeds, near, floor=floor)
    summ = similar.summary(df, seeds, near)
    gid = similar.make_groups(z, seeds, df)
    names, lost = similar.carry_names(gid)
    groups = similar.group_table(z, seeds, gid, names)
    summ.insert(1, "group", summ.seed.map(gid))
    summ.insert(2, "group_name", summ.group.map(lambda g: names.get(g, "")))
    groups.to_csv(out / "groups.csv", index=False, encoding="utf-8")
    similar.save_names(groups, lost)
    params = {"model": model, "floor": floor, "floor_source": source,
              "near": "per-seed self tolerance" if self_sims is not None else cfg.SIM_NEAR,
              "min_gap": cfg.SIM_MIN_GAP, "max_candidates": cfg.SIM_MAX_CANDIDATES}
    similar.save(df, summ, params, out)

    n = summ.n_matches
    print(f"  floor {floor:.2f} ({source}), min drop {cfg.SIM_MIN_GAP}")
    print(f"  matches per seed: none {np.mean(n == 0):.0%} | 1 {np.mean(n == 1):.0%} | "
          f"2-3 {np.mean((n >= 2) & (n <= 3)):.0%} | 4-6 {np.mean((n >= 4) & (n <= 6)):.0%} | "
          f"7+ {np.mean(n >= 7):.0%} | with a near-identical twin {np.mean(summ.n_near_identical > 0):.0%}")
    sz = groups["size"]
    print(f"  groups (avg similarity >= {cfg.GROUP_MIN_AVG}): {len(groups)} groups covering "
          f"{sz.sum()} seeds, sizes 2: {(sz == 2).sum()} | 3-5: {sz.between(3, 5).sum()} | "
          f"6+: {(sz >= 6).sum()} (largest {sz.max()}); {(summ.group < 0).sum()} seeds on their own")
    if names or lost:
        print(f"  names: {len(names)} carried over" + (f", {len(lost)} no longer match a group "
              "(kept with group -1)" if lost else ""))
    print(f"  -> {out / 'groups.csv'}  (one row per group)")
    print(f"  -> {cfg.GROUP_NAMES_CSV}  (type group names here; they survive reruns)")
    print(f"  -> {out / 'similar_summary.csv'}  (one row per seed)")
    print(f"  -> {out / 'similar.csv'}  (one row per match)")


def cmd_plot(args):
    """2D maps (PNG + interactive HTML) of one model's embeddings."""
    from pipeline.preprocess import load_processed
    from pipeline.projection import plot_html, plot_png, project_2d
    seeds, *_ = load_processed()
    bj = cfg.RESULTS_DIR / "best.json"
    best = json.loads(bj.read_text()) if bj.exists() else {}
    model = _model(args)
    out = cfg.RESULTS_DIR / model
    sim_csv = out / "similar.csv"
    lists = pd.read_csv(sim_csv) if sim_csv.exists() else None
    z = np.load(out / "z.npy")
    groups_csv = out / "groups.csv"
    default = "groups" if groups_csv.exists() else (best.get("method") if best.get("model") == model else None)
    color_by = args.color_by or default
    labels, names = None, None
    if color_by == "groups":
        g = pd.read_csv(groups_csv, dtype={"name": str}).fillna({"name": ""})
        gid = {int(s): int(r.group) for r in g.itertuples() for s in str(r.members).split()}
        labels = np.array([gid.get(int(s), -1) for s in seeds])
        names = dict(zip(g.group, g.name))
    elif color_by and color_by != "none":
        labels = pd.read_csv(out / color_by / "labels.csv").cluster.to_numpy()
    for proj in args.proj:
        xy = project_2d(z, proj)
        np.save(out / f"xy_{proj}.npy", xy)
        title = f"{model} | {proj.upper()} | colour: {color_by or 'none'}"
        plot_png(xy, labels, out / f"map_{proj}.png", title)
        plot_html(xy, seeds, labels, z, out / f"map_{proj}.html", title, lists=lists,
                  group_names=names)
        print(f"  {proj}: {out / f'map_{proj}.html'}")


def cmd_label(args):
    from pipeline import label_tool
    from pipeline.preprocess import load_processed
    seeds, *_ = load_processed()
    z = np.load(cfg.RESULTS_DIR / _model(args) / "z.npy")
    label_tool.run(z, seeds, per_bin=args.per_bin)


def cmd_all(args):
    """Research run: every model and clustering method, evaluated and compared."""
    cmd_prep(args); cmd_embed(args); cmd_cluster(args); cmd_eval(args)


def cmd_process(args):
    """The everyday path for one skin, skipping steps that are already done:
    download -> mask + crops -> DINOv2 embeddings -> centre+PCA -> per-seed shift
    tolerance -> look-alike lists + groups -> map. --force redoes all but the download."""
    import time
    from pipeline.data import list_seeds
    from pipeline.embed import embedding_path
    from pipeline.preprocess import load_processed

    model = args.model or "dinov2"
    out = cfg.RESULTS_DIR / model
    expected = cfg.SEED_MAX - cfg.SEED_MIN + 1
    print(f"== {cfg.SKIN}  ->  {cfg.SKIN_DIR}")
    if cfg.OVERRIDES:
        print(f"   settings.json overrides: {cfg.OVERRIDES}")
    t0 = time.time()

    def step(title, done, fn):
        skip = done and not args.force
        print(f"\n[{title}]" + ("  already done, skipping" if skip else ""))
        if not skip:
            fn()

    have = len(list_seeds())
    step(f"1/6 download ({have}/{expected} images on disk)", have >= expected,
         lambda: cmd_scrape(args))
    n_raw = len(list_seeds())
    if not n_raw:
        raise SystemExit("No images downloaded; check that the skin name matches the CDN.")

    meta_ok = (cfg.PROCESSED_DIR / "meta.json").exists() and len(load_processed()[0]) == n_raw
    step("2/6 blade mask + crops", meta_ok, lambda: cmd_prep(args))

    emb = embedding_path(model, args.tta)
    step(f"3/6 {model} embeddings (GPU)", emb.exists() and meta_ok,
         lambda: cmd_embed(argparse.Namespace(**{**vars(args), "models": [model]})))

    z_path = out / "z.npy"
    z_ok = z_path.exists() and z_path.stat().st_mtime >= emb.stat().st_mtime
    seeds = load_processed()[0]
    step("4/6 centre + PCA", z_ok, lambda: reduce_model(model, seeds, args.tta))
    if not z_ok or args.force:
        (out / "self_sim.npy").unlink(missing_ok=True)       # stale once z changes

    print("\n[5/6 look-alike lists + groups]")
    cmd_similar(argparse.Namespace(**{**vars(args), "model": model, "self_tol": True}))

    print("\n[6/6 map]")
    cmd_plot(argparse.Namespace(**{**vars(args), "model": model, "color_by": "groups",
                                   "proj": args.proj if args.proj_set else ["umap"]}))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Open {out / 'map_umap.html'}")
    print(f"Name groups in {cfg.GROUP_NAMES_CSV}, then run `similar` and `plot --proj umap` "
          "(or `process` again) to carry the names into the map.")


def _merge_csv(path, new, keys):
    """Update rows for `keys` in an existing CSV, so running one model keeps the others."""
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old.set_index(keys).index.isin(new.set_index(keys).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["process", "scrape", "prep", "model-inputs", "embed", "cluster",
                                        "eval", "similar", "plot", "label", "all"])
    ap.add_argument("--skin", default=cfg.DEFAULT_SKIN,
                    help="knife/finish[/variant] as in the CDN URL, e.g. butterfly/marble-fade")
    ap.add_argument("--force", action="store_true", help="process: redo every step except the download")
    ap.add_argument("--models", nargs="+", help=f"subset of {cfg.MODELS}")
    ap.add_argument("--no-tta", dest="tta", action="store_false", help="disable shift averaging")
    ap.add_argument("--model", help="label/eval: force this model")
    ap.add_argument("--method", help="eval: force this method (with --model)")
    ap.add_argument("--skip-shift", action="store_true", help="eval: skip the shift test")
    ap.add_argument("--proj", nargs="+", default=None,
                    help="plot: 2D projections to make (umap, tsne, pca)")
    ap.add_argument("--color-by", help="plot: 'groups' (default once run.py similar has run), a clustering run like agglo_t0.3, or 'none'")
    ap.add_argument("--per-bin", type=int, default=15, help="label: pairs per similarity band")
    ap.add_argument("--self-tol", action="store_true",
                    help="similar: measure each seed's own shift tolerance (GPU, ~5 min, cached)")
    args = ap.parse_args()
    args.proj_set = args.proj is not None
    args.proj = args.proj or ["umap", "tsne", "pca"]
    globals()[f"cmd_{args.command.replace('-', '_')}"](args)


if __name__ == "__main__":
    main()
