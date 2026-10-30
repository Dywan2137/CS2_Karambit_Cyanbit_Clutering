"""Hand-label look-alike pairs to score the models.

Shows a query seed next to a candidate seed. Candidates are drawn from the
current model's nearest neighbours (likely matches) plus some random seeds
(likely non-matches), so both answers occur.

Keys in the terminal:  y = same look   n = different   s = skip   q = quit
Answers are appended to data/labels/pairs.csv, so you can stop and resume.
"""

import csv

import matplotlib
import numpy as np
import pandas as pd

import config as cfg


def candidate_pairs(knn_df, seeds, n_queries=25, ranks=(1, 2, 3, 5, 10), n_random=1, seed=None):
    rng = np.random.default_rng(seed)
    queries = rng.choice(seeds, size=min(n_queries, len(seeds)), replace=False)
    pairs = []
    for q in queries:
        nb = knn_df[knn_df.seed == q].set_index("rank").neighbor
        pairs += [(int(q), int(nb[r]), f"rank{r}") for r in ranks if r in nb.index]
        others = [s for s in seeds if s != q]
        pairs += [(int(q), int(s), "random") for s in rng.choice(others, n_random, replace=False)]
    order = rng.permutation(len(pairs))
    return [pairs[i] for i in order]


def run(knn_df, seeds, n_queries=25):
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from PIL import Image

    cfg.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if cfg.PAIRS_CSV.exists():
        old = pd.read_csv(cfg.PAIRS_CSV)
        done = {tuple(sorted(p)) for p in zip(old.seed_a, old.seed_b)}
    new_file = not cfg.PAIRS_CSV.exists()

    todo = [p for p in candidate_pairs(knn_df, seeds, n_queries) if tuple(sorted(p[:2])) not in done]
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
