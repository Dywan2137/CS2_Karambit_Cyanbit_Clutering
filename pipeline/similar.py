"""Per-pattern lists of closely resembling seeds, with a list length that
depends on the pattern instead of a fixed k.

For each seed, walking down its ranking of most similar seeds:
  1. floor     - candidates must have similarity >= floor
  2. near      - candidates >= the seed's near-identical level are always kept
  3. drop      - the list ends after the largest drop between consecutive
                 candidates, if that drop is bigger than shift noise (min_gap);
                 with no clear drop, every candidate above the floor is kept
  4. two-way   - if A lists B, B also lists A

The near-identical level can be measured per seed: similarity between a seed
and slightly shifted copies of itself (`self_tolerance`). The floor can be
fitted from hand-labelled pairs (`calibrate_floor`).
"""

import json

import numpy as np
import pandas as pd

import config as cfg


# ---- per-seed tolerance -----------------------------------------------------

def self_tolerance(model, seeds, paths, mask, fill, out_dir, shifts=cfg.SELF_TOL_SHIFTS):
    """(N, len(shifts)) similarity of each seed to shifted copies of itself.
    Cached to out_dir/self_sim.npy."""
    path = out_dir / "self_sim.npy"
    if path.exists():
        sims = np.load(path)
        if sims.shape == (len(seeds), len(shifts)):
            return sims
    import joblib
    from PIL import Image
    from tqdm import tqdm
    from pipeline.embed import build_extractor, embed_images, shift_image

    reducer = joblib.load(out_dir / "reducer.joblib")
    z = np.load(out_dir / "z.npy")
    mean_crop = np.load(cfg.PROCESSED_DIR / "mean_crop.npy")
    ext = build_extractor(model, mask, mean_crop)
    sims = np.zeros((len(seeds), len(shifts)), dtype=np.float32)
    bs = cfg.BATCH_SIZE
    for start in tqdm(range(0, len(seeds), bs), desc="self tolerance"):
        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in paths[start:start + bs]]
        for k, (fx, fy) in enumerate(shifts):
            moved = [shift_image(im, fx * im.shape[1], fy * im.shape[0], fill) for im in imgs]
            zq = reducer.transform(embed_images(ext, moved, fill))
            sims[start:start + len(imgs), k] = (zq * z[start:start + len(imgs)]).sum(1)
    np.save(path, sims)
    return sims


def near_levels(self_sims=None, n=None, default=cfg.SIM_NEAR, clip=cfg.SELF_TOL_RANGE):
    """Per-seed near-identical level: the typical (median) similarity a seed has
    to its own shifted copies, clipped; or the global default when not measured.
    (The worst copy is too lenient: it labels half of all seeds as having a twin.)"""
    if self_sims is None:
        return np.full(n, default, dtype=np.float32)
    return np.clip(np.median(self_sims, axis=1), *clip)


# ---- calibration from labels ------------------------------------------------

def calibrate_floor(z, seeds, pairs, prob=cfg.CALIB_FLOOR_PROB, min_pairs=cfg.CALIB_MIN_PAIRS):
    """Fit P(alike | similarity) on labelled pairs and return (floor, report).
    floor = similarity where P(alike) = prob. None if there aren't enough labels."""
    if pairs is None or len(pairs) < min_pairs or pairs["same"].nunique() < 2:
        n = 0 if pairs is None else len(pairs)
        return None, f"only {n} labelled pairs (need {min_pairs} with both answers)"
    from sklearn.linear_model import LogisticRegression
    pos = {s: i for i, s in enumerate(seeds)}
    pairs = pairs[pairs.seed_a.isin(pos) & pairs.seed_b.isin(pos)]
    a = pairs.seed_a.map(pos).to_numpy()
    b = pairs.seed_b.map(pos).to_numpy()
    sim = (z[a] * z[b]).sum(1)
    y = pairs.same.to_numpy()
    lr = LogisticRegression(C=100).fit(sim[:, None], y)
    w, c = lr.coef_[0, 0], lr.intercept_[0]
    if w <= 0:
        return None, "labels don't increase with similarity; check them"
    floor = float((np.log(prob / (1 - prob)) - c) / w)

    lines = ["  similarity band   pairs   judged alike"]
    bins = cfg.LABEL_BINS
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (sim >= lo) & (sim < hi)
        if m.any():
            lines.append(f"  {lo:.1f} - {min(hi, 1):.1f}        {m.sum():4d}   {y[m].mean():5.0%}")
    lines.append(f"  fitted: P(alike) = {prob:.0%} at similarity {floor:.2f}")
    return floor, "\n".join(lines)


# ---- the lists --------------------------------------------------------------

def cut_list(sims, near, floor, min_gap):
    """How many of the (descending) candidate similarities to keep.

    The drop from the last item above the floor to the first one below it also
    counts as a possible cut, so a big fall that crosses the floor is seen."""
    n_above = int((sims >= floor).sum())
    if n_above <= 1 or len(sims) < 2:
        return n_above
    n_near = int((sims >= near).sum())
    gaps = sims[:-1] - sims[1:]                 # gaps[j] = drop after item j
    j_idx = np.arange(len(gaps))
    # cut after item j: only above the floor, never inside the near-identical run
    allowed = (j_idx < n_above) & (j_idx >= max(n_near - 1, 0))
    if not allowed.any():
        return n_above
    gaps = np.where(allowed, gaps, -np.inf)
    j = int(np.argmax(gaps))
    return j + 1 if gaps[j] >= min_gap else n_above


def similar_lists(z, seeds, near, floor=cfg.SIM_FLOOR, min_gap=cfg.SIM_MIN_GAP,
                  max_candidates=cfg.SIM_MAX_CANDIDATES,
                  loose_floor=cfg.SIM_LOOSE_FLOOR, loose_fill=cfg.SIM_LOOSE_FILL):
    """Long table: seed, neighbor, similarity, tier, how (rule / two-way / loose).

    Seeds with fewer than `loose_fill` matches get topped up with their next
    nearest seeds down to `loose_floor`, marked tier "loose"."""
    S = z @ z.T
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)[:, :max_candidates]
    keep = {}
    for i in range(len(seeds)):
        cand = order[i]
        n = cut_list(S[i, cand], near[i], floor, min_gap)
        keep[i] = {int(j): "rule" for j in cand[:n]}
    for i in list(keep):                        # two-way: A lists B -> B lists A
        for j in list(keep[i]):
            keep[j].setdefault(i, "two-way")
    loose_floor = min(loose_floor, floor)
    for i in keep:                              # loose top-up, one-way
        for j in order[i]:
            if len(keep[i]) >= loose_fill or S[i, j] < loose_floor:
                break
            keep[i].setdefault(int(j), "loose")

    rows = []
    for i, nbrs in keep.items():
        for j, how in sorted(nbrs.items(), key=lambda t: -S[i, t[0]]):
            sim = float(S[i, j])
            tier = ("loose" if how == "loose" else
                    "near-identical" if sim >= near[i] else "similar")
            rows.append({"seed": seeds[i], "neighbor": seeds[j], "similarity": round(sim, 4),
                         "tier": tier, "how": how})
    df = pd.DataFrame(rows, columns=["seed", "neighbor", "similarity", "tier", "how"])
    df["rank"] = df.groupby("seed").cumcount() + 1
    return df


def summary(df, seeds, near):
    """One row per seed: counts and a readable list."""
    g = {s: d for s, d in df.groupby("seed")}
    rows = []
    for i, s in enumerate(seeds):
        d = g.get(s)
        if d is None:
            rows.append({"seed": s, "n_matches": 0, "n_near_identical": 0,
                         "near_level": round(float(near[i]), 3), "matches": ""})
            continue
        rows.append({
            "seed": s, "n_matches": len(d), "n_near_identical": int((d.tier == "near-identical").sum()),
            "near_level": round(float(near[i]), 3),
            "matches": ", ".join(f"{n} ({v:.2f}{_MARK.get(t, '')})"
                                 for n, v, t in zip(d.neighbor, d.similarity, d.tier)),
        })
    return pd.DataFrame(rows)


_MARK = {"near-identical": "*", "loose": "~"}


# ---- groups -----------------------------------------------------------------

def make_groups(z, seeds, df, min_avg=cfg.GROUP_MIN_AVG):
    """Split the seeds into non-overlapping groups (each seed in exactly one).

    Average-linkage merging, with two rules:
      - two groups may only merge if some seed of one is on a list of the other
        (all tiers, loose included), so groups follow the look-alike lists
      - the average similarity over all pairs in the merged group must stay
        >= min_avg, which stops long chains A~B~C~...~Z from forming
    A seed ends up in the group it fits best on average, so it can land in a
    different group than its single closest match.
    Returns {seed: group_id} with -1 for seeds left on their own."""
    pos = {s: i for i, s in enumerate(seeds)}
    n = len(seeds)
    S = (z @ z.T).astype(np.float64)
    np.fill_diagonal(S, 0)
    linked = np.zeros((n, n), bool)
    a, b = df.seed.map(pos).to_numpy(), df.neighbor.map(pos).to_numpy()
    linked[a, b] = linked[b, a] = True

    tot, size = S.copy(), np.ones(n)            # tot[A,B] = sum of similarities between groups
    members = {i: [i] for i in range(n)}
    while True:
        avg = np.where(linked, tot / np.outer(size, size), -np.inf)
        i, j = divmod(int(np.argmax(avg)), n)
        if avg[i, j] < min_avg:
            break
        # within-group average after merging i and j
        inner = lambda k: S[np.ix_(members[k], members[k])].sum() / 2
        m = size[i] + size[j]
        if (inner(i) + inner(j) + tot[i, j]) / (m * (m - 1) / 2) < min_avg:
            linked[i, j] = linked[j, i] = False  # would make the group too loose
            continue
        tot[i] += tot[j]; tot[:, i] += tot[:, j]
        linked[i] |= linked[j]; linked[:, i] |= linked[:, j]
        tot[j] = tot[:, j] = 0
        linked[j] = linked[:, j] = linked[i, i] = False
        size[i] = m
        members[i] += members.pop(j)

    groups = sorted((g for g in members.values() if len(g) > 1),
                    key=lambda g: (-len(g), min(seeds[k] for k in g)))
    gid = {s: -1 for s in seeds}
    for g_id, g in enumerate(groups):
        for k in g:
            gid[seeds[k]] = g_id
    return gid


def group_table(z, seeds, gid, names=None):
    """One row per group: id, name, size, cohesion (average pair similarity), members."""
    pos = {s: i for i, s in enumerate(seeds)}
    rows = []
    for g in sorted(set(gid.values()) - {-1}):
        mem = sorted(s for s, v in gid.items() if v == g)
        idx = [pos[s] for s in mem]
        sub = z[idx] @ z[idx].T
        k = len(idx)
        rows.append({"group": g, "name": (names or {}).get(g, ""), "size": k,
                     "cohesion": round(float((sub.sum() - k) / (k * (k - 1))), 3),
                     "members": " ".join(map(str, mem))})
    return pd.DataFrame(rows, columns=["group", "name", "size", "cohesion", "members"])


def carry_names(gid, names_csv=cfg.GROUP_NAMES_CSV, min_overlap=0.5):
    """Names you typed into group_names.csv, matched to the new groups by membership
    (Jaccard >= min_overlap), so names survive reruns that shift a seed or two.
    Returns ({group_id: name}, [names that no longer match any group])."""
    if not names_csv.exists():
        return {}, []
    old = pd.read_csv(names_csv, dtype={"name": str}).fillna({"name": ""})
    old = old[old.name.str.strip() != ""]
    new = {}
    for s, g in gid.items():
        if g >= 0:
            new.setdefault(g, set()).add(s)
    names, lost = {}, []
    for name, members in zip(old.name, old.members):
        mem = set(int(x) for x in str(members).split())
        best, score = None, 0.0
        for g, m in new.items():
            j = len(mem & m) / len(mem | m)
            if j > score and g not in names:
                best, score = g, j
        if best is not None and score >= min_overlap:
            names[best] = name
        else:
            lost.append((name, " ".join(map(str, sorted(mem)))))
    return names, lost


def save_names(groups, lost, names_csv=cfg.GROUP_NAMES_CSV):
    """Rewrite group_names.csv with the current groups (names carried over).
    Names whose group fell apart are kept at the bottom with group -1."""
    names_csv.parent.mkdir(parents=True, exist_ok=True)
    out = groups[["group", "name", "size", "members"]].copy()
    if lost:
        out = pd.concat([out, pd.DataFrame([{"group": -1, "name": n, "size": len(m.split()),
                                             "members": m} for n, m in lost])])
    out.to_csv(names_csv, index=False, encoding="utf-8")


def save(df, summ, params, out_dir):
    df.to_csv(out_dir / "similar.csv", index=False)
    summ.to_csv(out_dir / "similar_summary.csv", index=False)
    (out_dir / "similar_params.json").write_text(json.dumps(params, indent=2))
