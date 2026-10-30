"""Contact sheets, a 2D map and an HTML report for inspecting clusters by eye."""

import html
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import config as cfg

THUMB = 160


def _thumb(seed, size=THUMB):
    """Thumbnail `size` px wide, keeping the processed image's aspect ratio."""
    p = cfg.PROCESSED_DIR / f"seed_{seed:04d}.png"
    im = Image.open(p).convert("RGB")
    im = im.resize((size, max(1, round(im.height * size / im.width))), Image.LANCZOS)
    ImageDraw.Draw(im).text((4, 4), str(seed), fill=(255, 255, 255))
    return im


def contact_sheet(seeds, path, cols=8, size=THUMB, max_n=48):
    seeds = list(seeds)[:max_n]
    cols = min(cols, len(seeds))
    rows = (len(seeds) + cols - 1) // cols
    thumbs = [_thumb(s, size) for s in seeds]
    th = thumbs[0].height if thumbs else size
    sheet = Image.new("RGB", (cols * size, rows * th), (20, 20, 20))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * size, (i // cols) * th))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def neighbor_sheet(query, neighbors, path, size=THUMB):
    """Query on the left, its nearest seeds to the right."""
    contact_sheet([query] + list(neighbors), path, cols=len(neighbors) + 1, size=size)


def scatter_2d(xy, labels, path, title=""):
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    noise = labels < 0
    ax.scatter(xy[noise, 0], xy[noise, 1], s=6, c="#bbbbbb", label="unique")
    ax.scatter(xy[~noise, 0], xy[~noise, 1], s=10, c=labels[~noise] % 20, cmap="tab20")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report(metrics_df, best, seeds, labels, xy, knn_df, out_path=cfg.RESULTS_DIR / "report.html",
           n_neighbor_examples=12):
    """One HTML page: model comparison, 2D map, and a contact sheet per cluster."""
    model, method = best
    run_dir = cfg.RESULTS_DIR / model / method
    sheets = run_dir / "sheets"
    labels = np.asarray(labels)

    scatter_2d(xy, labels, run_dir / "map.png", f"{model} / {method}")
    cluster_ids = [c for c in np.unique(labels) if c >= 0]
    for c in cluster_ids:
        contact_sheet([s for s, l in zip(seeds, labels) if l == c], sheets / f"cluster_{c:03d}.png")

    rng = np.random.default_rng(cfg.SEED)
    examples = rng.choice(seeds, size=min(n_neighbor_examples, len(seeds)), replace=False)
    for q in examples:
        nb = knn_df[knn_df.seed == q].sort_values("rank").neighbor.tolist()[:7]
        neighbor_sheet(q, nb, sheets / f"nn_{q:04d}.png")

    rel = lambda p: os.path.relpath(p, out_path.parent).replace(os.sep, "/")
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Pattern clusters</title>",
        "<style>body{font-family:sans-serif;margin:24px;background:#111;color:#eee}"
        "table{border-collapse:collapse}td,th{border:1px solid #444;padding:4px 8px;font-size:13px}"
        "img{max-width:100%}h3{margin-top:28px}</style>",
        f"<h1>{cfg.SLUG}: pattern clusters</h1>",
        f"<p>Shown: <b>{html.escape(model)} / {html.escape(method)}</b> — "
        f"{len(cluster_ids)} groups, {(labels < 0).sum()} unique seeds.</p>",
        "<h2>Model / method comparison</h2>",
        metrics_df.to_html(index=False, float_format=lambda v: f"{v:.3f}"),
        "<h2>Map (UMAP 2D)</h2>", f"<img src='{rel(run_dir / 'map.png')}'>",
        "<h2>Nearest look-alikes (query on the left)</h2>",
    ]
    parts += [f"<img src='{rel(sheets / f'nn_{q:04d}.png')}'><br>" for q in examples]
    parts.append("<h2>Clusters</h2>")
    for c in cluster_ids:
        members = [s for s, l in zip(seeds, labels) if l == c]
        parts.append(f"<h3>Group {c} — {len(members)} seeds: {', '.join(map(str, members[:40]))}</h3>")
        parts.append(f"<img src='{rel(sheets / f'cluster_{c:03d}.png')}'>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
