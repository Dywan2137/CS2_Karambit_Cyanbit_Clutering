"""Remove the parts of the render that are the same for every seed.

The knife model, handle and background never change between seeds. Only the
pattern on the blade does. We find those pixels from the data itself: the
per-pixel standard deviation across all renders is ~0 for static pixels and
high where the pattern lives. That gives a blade mask, which we use to crop
every render and blank out everything else.
"""

import json

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import config as cfg
from pipeline.data import load_image


def pixel_stats(paths):
    """Per-pixel mean and std over all images (streaming Welford, float64)."""
    mean = m2 = None
    for n, p in enumerate(tqdm(paths, desc="pixel stats"), 1):
        x = load_image(p).astype(np.float64)
        if mean is None:
            mean = np.zeros_like(x)
            m2 = np.zeros_like(x)
        d = x - mean
        mean += d / n
        m2 += d * (x - mean)
    std = np.sqrt(m2 / max(len(paths) - 1, 1))
    return mean.astype(np.float32), std.astype(np.float32)


def blade_mask(std, threshold=cfg.MASK_STD_THRESHOLD):
    """Boolean HxW mask of pixels that vary between seeds."""
    s = std.mean(axis=2)
    s8 = np.clip(s / max(s.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    if threshold is None:
        _, m = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        m = ((s > threshold) * 255).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    # drop specks: keep components larger than 1% of the biggest one
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.flatnonzero(areas >= 0.01 * areas.max())
        m = np.isin(lab, keep).astype(np.uint8) * 255
    return m > 0


def part_boxes(mask, margin=cfg.MASK_MARGIN):
    """Bounding box (x0, y0, x1, y1) of each patterned part, largest first.

    On the karambit these are the blade and the ring. They sit at opposite
    ends of the knife, so they are cut out separately instead of one big box.
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n < 2:
        raise SystemExit("Blade mask is empty: check the std heatmap / threshold.")
    h, w = mask.shape
    boxes = []
    for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
        x, y, bw, bh = stats[i, :4]
        boxes.append((max(x - margin, 0), max(y - margin, 0),
                      min(x + bw + margin, w), min(y + bh + margin, h)))
    return [tuple(int(v) for v in b) for b in boxes]


def make_layout(boxes):
    """Largest part on top, the other parts side by side below it, centred.
    Returns (canvas_w, canvas_h, [(box, (dst_x, dst_y)), ...])."""
    top, rest = boxes[0], boxes[1:]
    tw, th = top[2] - top[0], top[3] - top[1]
    rw = sum(b[2] - b[0] for b in rest)
    rh = max((b[3] - b[1] for b in rest), default=0)
    W, H = max(tw, rw), th + rh
    placed = [(top, ((W - tw) // 2, 0))]
    x = (W - rw) // 2
    for b in rest:
        placed.append((b, (x, th)))
        x += b[2] - b[0]
    return W, H, placed


def compose(img, layout, fill):
    """Cut each part out of `img` and paste it onto the layout canvas."""
    W, H, placed = layout
    out = np.empty((H, W) + img.shape[2:], dtype=img.dtype)
    out[...] = fill
    for (x0, y0, x1, y1), (dx, dy) in placed:
        out[dy:dy + y1 - y0, dx:dx + x1 - x0] = img[y0:y1, x0:x1]
    return out


def resize(img, wh, nearest=False):
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(img, wh, interpolation=interp)


def apply_mask(img, mask, fill):
    out = img.copy()
    out[~mask] = fill
    return out


def save_diagnostics(mean, std, mask, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mean.clip(0, 255).astype(np.uint8)).save(out_dir / "mean.png")
    s = std.mean(axis=2)
    heat = cv2.applyColorMap((s / max(s.max(), 1e-6) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    Image.fromarray(heat[..., ::-1]).save(out_dir / "std_heatmap.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / "blade_mask.png")
    overlay = mean.clip(0, 255).astype(np.uint8).copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    Image.fromarray(overlay).save(out_dir / "mask_overlay.png")


def preprocess(seeds, paths, out_dir=cfg.PROCESSED_DIR, width=cfg.PROC_WIDTH, reuse_stats=True):
    """Write masked blade+ring crops to out_dir and return metadata.
    Pixel stats are cached, so tweaking the layout doesn't re-read every image twice."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mp, sp = out_dir / "_pixel_mean.npy", out_dir / "_pixel_std.npy"
    cached = reuse_stats and mp.exists() and sp.exists()
    if cached and np.load(sp, mmap_mode="r").shape[:2] != load_image(paths[0]).shape[:2]:
        cached = False
    if cached:
        mean, std = np.load(mp), np.load(sp)
    else:
        mean, std = pixel_stats(paths)
        np.save(mp, mean); np.save(sp, std)
    mask = blade_mask(std)
    save_diagnostics(mean, std, mask, out_dir / "_diagnostics")

    fill = mean[mask].mean(axis=0).round().astype(np.uint8)   # average blade colour
    boxes = part_boxes(mask)
    layout = make_layout(boxes)
    W, H, _ = layout
    wh = (width, int(round(H * width / W)))
    mask_c = resize(compose(mask.astype(np.uint8), layout, 0), wh, nearest=True).astype(bool)
    mean_c = resize(compose(mean, layout, fill.astype(np.float32)), wh)

    Image.fromarray((mask_c * 255).astype(np.uint8)).save(out_dir / "mask.png")
    np.save(out_dir / "mean_crop.npy", mean_c)

    for seed, p in tqdm(list(zip(seeds, paths)), desc="crop+mask"):
        img = compose(apply_mask(load_image(p), mask, fill), layout, fill)
        Image.fromarray(resize(img, wh)).save(out_dir / f"seed_{seed:04d}.png")

    meta = {
        "boxes": boxes, "size": wh, "fill": fill.tolist(), "scale": width / W,
        "mask_fraction": float(mask.mean()), "n": len(seeds), "seeds": seeds,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))
    print(f"  Mask covers {mask.mean():.1%} of the render in {len(boxes)} parts {boxes}")
    print(f"  Processed size {wh[0]}x{wh[1]} (scale {width / W:.2f} of the original)")
    print(f"  Check {out_dir / '_diagnostics' / 'mask_overlay.png'}")
    return meta


def load_processed(out_dir=cfg.PROCESSED_DIR):
    """(seeds, paths, mask, meta) for the processed crops."""
    meta = json.loads((out_dir / "meta.json").read_text())
    seeds = meta["seeds"]
    paths = [out_dir / f"seed_{s:04d}.png" for s in seeds]
    mask = np.asarray(Image.open(out_dir / "mask.png")) > 127
    return seeds, paths, mask, meta
