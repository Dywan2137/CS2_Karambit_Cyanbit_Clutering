"""Hand-label look-alike pairs to calibrate the similarity floor and score models.

Pairs are sampled evenly across similarity bands (config.LABEL_BINS), so you
see clear twins, clear non-matches and the borderline cases in between. That
lets `run.py similar` estimate, at each similarity level, how often you'd call
two patterns alike.

Keys in the terminal:  y = same look   n = different   s = skip   q = quit
Answers are appended to data/<skin>/labels/pairs.csv, so you can stop and resume.
"""

import csv

import matplotlib
import numpy as np
import pandas as pd

import config as cfg


def candidate_pairs(z, seeds, per_bin=15, done=frozenset(), bins=cfg.LABEL_BINS, seed=None):
    """Up to `per_bin` random unlabelled pairs from each similarity band, shuffled."""
    rng = np.random.default_rng(seed)
    S = z @ z.T
    iu, ju = np.triu_indices(len(seeds), 1)
    sim = S[iu, ju]
    pairs = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = np.flatnonzero((sim >= lo) & (sim < hi))
        rng.shuffle(idx)
        taken = 0
        for k in idx:
            a, b = int(seeds[iu[k]]), int(seeds[ju[k]])
            if (min(a, b), max(a, b)) in done:
                continue
            pairs.append((a, b, f"band{lo:.1f}"))
            taken += 1
            if taken == per_bin:
                break
    return [pairs[i] for i in rng.permutation(len(pairs))]


def run(z, seeds, per_bin=15):
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from PIL import Image

    cfg.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if cfg.PAIRS_CSV.exists():
        old = pd.read_csv(cfg.PAIRS_CSV)
        done = {tuple(sorted(p)) for p in zip(old.seed_a, old.seed_b)}
    new_file = not cfg.PAIRS_CSV.exists()

    todo = candidate_pairs(z, seeds, per_bin, done)
    print(f"{len(todo)} pairs to label. y = same, n = different, s = skip, q = quit")

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    with open(cfg.PAIRS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["seed_a", "seed_b", "same", "source"])
        for i, (a, b, src) in enumerate(todo, 1):
            for ax, s in zip(axes, (a, b)):
                ax.clear()
                ax.imshow(Image.open(cfg.PROCESSED_DIR / f"seed_{s:04d}.png"))
                ax.set_title(f"seed {s}")
                ax.axis("off")
            fig.suptitle(f"[{i}/{len(todo)}] Same look?")
            fig.canvas.draw()
            plt.pause(0.05)
            ans = ""
            while ans not in ("y", "n", "s", "q"):
                ans = input(f"  {a} vs {b}  [y/n/s/q]: ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue
            w.writerow([a, b, int(ans == "y"), src])
            f.flush()
    plt.close(fig)
    print(f"Saved to {cfg.PAIRS_CSV}")
