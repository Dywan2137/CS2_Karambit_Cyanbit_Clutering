"""Central settings for the Karambit Gamma Doppler Phase 3 clustering pipeline."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- skin -------------------------------------------------------------------
KNIFE = "karambit"
FINISH = "gamma-doppler"
VARIANT = "phase-3"
SLUG = f"{KNIFE}-{FINISH}-{VARIANT}"           # CDN path segment
SEED_MIN, SEED_MAX = 1, 1000                    # all pattern seeds

# ---- scraping ---------------------------------------------------------------
# Base URL of the csgoskins.gg image CDN, e.g. "https://cdn.example.com/patterns".
# Images are fetched from <CDN_BASE>/<SLUG>/<seed>.png
CDN_BASE = os.environ.get("CDN_BASE", "https://cdn.csgoskins.gg/public/images/patterns/v1")
SCRAPE_DELAY = 0.5        # seconds between requests
SCRAPE_RETRIES = 2
SEED = 42                 # random seed for anything stochastic

# ---- paths ------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw" / KNIFE / FINISH / VARIANT
MANIFEST = RAW_DIR / "manifest.csv"
PROCESSED_DIR = DATA_DIR / "processed"
EMB_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = DATA_DIR / "results"
LABELS_DIR = DATA_DIR / "labels"
PAIRS_CSV = LABELS_DIR / "pairs.csv"

# ---- preprocessing ----------------------------------------------------------
MASK_STD_THRESHOLD = None   # None = Otsu threshold on the per-pixel std map
MASK_MARGIN = 8             # px of padding around each patterned part
PROC_WIDTH = 896            # width of the processed image (blade on top, ring below)

# ---- embeddings -------------------------------------------------------------
MODELS = ["dinov2", "clip", "resnet50", "colorhist"]
DINO_NAME = "facebook/dinov2-base"
DINO_WIDTH = 672           # DINOv2 input width; height follows the aspect ratio (multiples of 14)
CLIP_ARCH, CLIP_WEIGHTS = "ViT-B-32", "laion2b_s34b_b79k"
BATCH_SIZE = 16
TTA_SHIFTS = [(0, 0), (0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03)]  # fraction of size

# ---- reduction / clustering -------------------------------------------------
PCA_VARIANCE = 0.95
PCA_MAX_DIMS = 64
UMAP_DIMS = 10
UMAP_NEIGHBORS = 15
HDBSCAN_MIN_CLUSTER_SIZES = [2, 3, 5]
HDBSCAN_MIN_SAMPLES = [1, 2, 3]
AGGLO_LINKAGE = "average"
AGGLO_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]   # cosine distance cut levels
KNN_K = 10

# ---- evaluation -------------------------------------------------------------
SHIFT_TEST_N = 200          # seeds used in the synthetic shift test
SHIFT_TEST_FRAC = 0.04      # max shift as fraction of image size
STABILITY_RUNS = 10
STABILITY_FRAC = 0.8
