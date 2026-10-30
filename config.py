"""Central settings for the pattern clustering pipeline.

The skin is picked with `run.py --skin knife/finish[/variant]` (or the SKIN env
var). Every skin gets its own folder data/<knife>/<finish>/<variant>/ with raw
images, processed crops, embeddings, results and labels. A settings.json in that
folder overrides any setting below for that skin only, e.g. {"SIM_FLOOR": 0.65}.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- skin -------------------------------------------------------------------
DEFAULT_SKIN = "karambit/gamma-doppler/phase-3"


def normalize_skin(skin):
    """'Karambit / Gamma Doppler / Phase 3' -> 'karambit/gamma-doppler/phase-3'."""
    parts = [p.strip().lower().replace(" ", "-") for p in skin.replace("\\", "/").split("/")]
    parts = [p for p in parts if p]
    if not 2 <= len(parts) <= 3:
        raise SystemExit(f"--skin must look like knife/finish or knife/finish/variant, got {skin!r}")
    return "/".join(parts)


SKIN = normalize_skin(os.environ.get("SKIN") or DEFAULT_SKIN)
KNIFE, FINISH, *_rest = SKIN.split("/")
VARIANT = _rest[0] if _rest else ""
SLUG = SKIN.replace("/", "-")                    # CDN path segment
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
SKIN_DIR = DATA_DIR / SKIN
RAW_DIR = SKIN_DIR / "raw"
MANIFEST = RAW_DIR / "manifest.csv"
PROCESSED_DIR = SKIN_DIR / "processed"
MODEL_INPUTS_DIR = SKIN_DIR / "model_inputs"
EMB_DIR = SKIN_DIR / "embeddings"
RESULTS_DIR = SKIN_DIR / "results"
LABELS_DIR = SKIN_DIR / "labels"
PAIRS_CSV = LABELS_DIR / "pairs.csv"
SKIN_SETTINGS = SKIN_DIR / "settings.json"

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

# ---- per-pattern similar lists (run.py similar) -----------------------------
# Defaults; `run.py similar` replaces SIM_FLOOR with the value fitted from your
# labels (<skin>/labels/pairs.csv) when there are enough of them.
SIM_FLOOR = 0.60            # below this a pattern is never "close"
SIM_NEAR = 0.80             # near-identical: as close as the same pattern shifted 2-4%
SIM_MIN_GAP = 0.05          # a drop must exceed this (shift noise) to end a list
SIM_MAX_CANDIDATES = 15     # only look this far down each seed's ranking
SELF_TOL_SHIFTS = [(0.02, 0.02), (-0.02, 0.02), (0.03, -0.03), (-0.04, -0.01)]
SELF_TOL_RANGE = (0.75, 0.92)   # per-seed near-identical level is clipped to this
CALIB_FLOOR_PROB = 0.5     # floor = similarity where a pair is judged "alike" 50% of the time
CALIB_MIN_PAIRS = 40
LABEL_BINS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]   # labeling samples evenly from these bands
SIM_LOOSE_FLOOR = 0.50      # loose tier: weaker matches shown for seeds with short lists
SIM_LOOSE_FILL = 3          # ...topping each list up to this many entries

# Groups: every seed in at most one group, built from the lists above.
GROUP_MIN_AVG = 0.50        # average similarity over all pairs inside a group
GROUP_NAMES_CSV = LABELS_DIR / "group_names.csv"   # type names in the `name` column


# ---- per-skin overrides -----------------------------------------------------
def _apply_overrides(path=SKIN_SETTINGS):
    if not path.exists():
        return {}
    overrides = json.loads(path.read_text(encoding="utf-8"))
    g = globals()
    unknown = [k for k in overrides if k not in g or not k.isupper()]
    if unknown:
        raise SystemExit(f"{path}: unknown settings {unknown}")
    for k, v in overrides.items():
        g[k] = tuple(v) if isinstance(g[k], tuple) else v
    return overrides


OVERRIDES = _apply_overrides()
