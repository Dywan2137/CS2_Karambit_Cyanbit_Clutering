"""Download all 1000 Karambit Gamma Doppler Phase 3 pattern renders.

Each pattern render is served from a predictable CDN URL:

    <CDN_BASE>/<knife>-<finish>[-<variant>]/<seed>.png
    e.g. karambit-gamma-doppler-phase-3/42.png

Images land in data/<skin>/raw/seed_XXXX.png and a
manifest.csv (seed, path, status, sha256) is written next to them.
Already-downloaded files are skipped, so an interrupted run is safe to repeat.

    python run.py scrape --skin butterfly/marble-fade        # usual way
    python scraper.py --max-seed 50    # quick test (skin from the SKIN env var)
"""

import argparse
import csv
import hashlib
import time
from pathlib import Path

import requests

import config as cfg

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://csgoskins.gg/",
}


def seed_path(seed, folder=cfg.RAW_DIR):
    return Path(folder) / f"seed_{seed:04d}.png"


def fetch(url, retries=cfg.SCRAPE_RETRIES):
    """Return image bytes, or None if the URL doesn't give a valid image."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=15, headers=HEADERS)
            ct = r.headers.get("content-type", "")
            # a real image is 200, an image content-type, and not a tiny error blob
            if r.status_code == 200 and "image" in ct and len(r.content) > 1000:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


def scrape(seeds, folder=cfg.RAW_DIR, delay=cfg.SCRAPE_DELAY):
    if not cfg.CDN_BASE:
        raise SystemExit("CDN_BASE is not set. Set it in config.py or the CDN_BASE env var.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    rows, ok, skipped, failed = [], 0, 0, 0
    for i, seed in enumerate(seeds, 1):
        dest = seed_path(seed, folder)
        if dest.exists():
            status = "skipped"
            skipped += 1
        else:
            data = fetch(f"{cfg.CDN_BASE}/{cfg.SLUG}/{seed}.png")
            if data is None:
                status = "failed"
                failed += 1
            else:
                dest.write_bytes(data)
                status = "ok"
                ok += 1
            time.sleep(delay)
        sha = hashlib.sha256(dest.read_bytes()).hexdigest() if dest.exists() else ""
        rows.append({"seed": seed, "path": dest.name, "status": status, "sha256": sha})
        if failed >= 10 and ok == 0 and skipped == 0:
            raise SystemExit(f"  The first {failed} seeds all failed. Check the skin name: "
                             f"{cfg.CDN_BASE}/{cfg.SLUG}/<seed>.png")
        if i % 50 == 0 or i == len(seeds):
            print(f"  [{i:4d}/{len(seeds)}]  ok {ok}  skip {skipped}  fail {failed}")

    with open(folder / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "path", "status", "sha256"])
        w.writeheader()
        w.writerows(rows)

    if failed:
        print(f"  WARNING: {failed} seeds failed: rerun to retry them.")
    return ok, skipped, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download Karambit Gamma Doppler P3 renders.")
    ap.add_argument("--min-seed", type=int, default=cfg.SEED_MIN)
    ap.add_argument("--max-seed", type=int, default=cfg.SEED_MAX)
    args = ap.parse_args(argv)
    seeds = list(range(args.min_seed, args.max_seed + 1))
    print(f"Downloading {len(seeds)} seeds of {cfg.SLUG} -> {cfg.RAW_DIR}")
    scrape(seeds)


if __name__ == "__main__":
    main()
