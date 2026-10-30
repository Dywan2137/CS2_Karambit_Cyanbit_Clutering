"""Load the raw renders and check that the dataset is complete and consistent."""

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import config as cfg

_SEED_RE = re.compile(r"seed_(\d+)\.png$")


def list_seeds(folder=cfg.RAW_DIR):
    """Sorted list of (seed, path) for every render in the folder."""
    items = []
    for p in Path(folder).glob("seed_*.png"):
        m = _SEED_RE.search(p.name)
        if m:
            items.append((int(m.group(1)), p))
    return sorted(items)


def load_image(path):
    """RGB uint8 array. Transparent backgrounds are flattened onto black."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
    return np.asarray(img.convert("RGB"))


def load_dataset(folder=cfg.RAW_DIR, expected=range(cfg.SEED_MIN, cfg.SEED_MAX + 1)):
    """Return (seeds, paths) after sanity checks. Prints a short report."""
    items = list_seeds(folder)
    if not items:
        raise SystemExit(f"No renders found in {folder}. Run `python run.py scrape` first.")
    seeds = [s for s, _ in items]
    paths = [p for _, p in items]

    missing = sorted(set(expected) - set(seeds))
    if missing:
        print(f"  WARNING: {len(missing)} seeds missing (first: {missing[:10]})")

    sizes = defaultdict(list)
    hashes = defaultdict(list)
    for s, p in items:
        with Image.open(p) as im:
            sizes[im.size].append(s)
        hashes[hashlib.sha256(p.read_bytes()).hexdigest()].append(s)

    if len(sizes) > 1:
        common = max(sizes, key=lambda k: len(sizes[k]))
        odd = {k: v[:5] for k, v in sizes.items() if k != common}
        raise SystemExit(f"Images have different sizes. Most common {common}, others: {odd}")

    dupes = [v for v in hashes.values() if len(v) > 1]
    if dupes:
        print(f"  NOTE: {len(dupes)} groups of byte-identical renders, e.g. {dupes[:3]}")

    print(f"  Loaded {len(seeds)} renders, size {next(iter(sizes))}")
    return seeds, paths
