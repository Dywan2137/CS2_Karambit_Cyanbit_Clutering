# CS2_Karambit_Cyanbit_Clutering
Groups CS2 skin patterns (seeds 1-1000) that look very similar, so players have cheaper alternatives to patterns that are already highly priced. Float and corner are not taken into account. Built and tuned on the Karambit Gamma Doppler Phase 3, but it runs on any skin the image CDN has pattern renders for.

## Quick start: one skin, one command
```
python run.py process --skin karambit/gamma-doppler/phase-3
python run.py process --skin butterfly-knife/marble-fade
python run.py process --skin ak-47/case-hardened
```
`--skin` is `knife/finish[/variant]`, written as in the CDN URL `<CDN_BASE>/<knife>-<finish>[-<variant>]/<seed>.png` (so the Butterfly is `butterfly-knife`). A wrong name stops after 10 failed downloads.

`process` downloads the renders (about 1.8 GB and 10-15 min per skin), finds the pattern mask, embeds with DINOv2, builds the look-alike lists and groups, and draws the map. Steps that are already done are skipped, so it is safe to rerun. Use `--force` to redo everything except the download. On a new skin, check `processed/_diagnostics/mask_overlay.png` once: the mask should cover only the patterned parts.

Each skin has its own folder, so skins never overwrite each other:
```
data/<knife>/<finish>/<variant>/
  raw/  processed/  embeddings/  results/  labels/  settings.json (optional)
```
`settings.json` overrides any setting from `config.py` for that skin only, e.g. `{"SIM_FLOOR": 0.65, "GROUP_MIN_AVG": 0.55}`. The defaults were tuned on the Gamma Doppler, so a skin with a very different pattern (Case Hardened, Fade) is best calibrated by labelling about 100 pairs (`python run.py label --skin ...`) and running `process` again.

## How it works
1. **Scrape**: download all 1000 pattern renders (`scraper.py`).
2. **Prep**: the knife model, handle and background are the same in every render, so the pipeline computes the per-pixel variation across all seeds. Pixels that change are the pattern (the blade and the ring). Both parts are cut out, stacked (blade on top, ring below) and everything else is blanked out.
3. **Embed**: features from DINOv2 (main), CLIP, ResNet50 and a colour-histogram baseline. Each embedding is averaged over a few slightly shifted copies, so shifted patterns land close together.
4. **Reduce**: subtract the mean embedding (removes the shared "it's a karambit" part), then L2-normalize and PCA. UMAP is used for HDBSCAN.
5. **Cluster**: HDBSCAN and agglomerative clustering. Seeds with no look-alike are labelled `-1` (unique). The top 10 look-alikes per seed are also saved.
6. **Evaluate**: silhouette, stability (ARI on subsamples), a synthetic shift test, and scores on pairs you label by hand. The best model and method are picked and written to an HTML report.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPU build
pip install -r requirements.txt
```
The image CDN address is set in `config.py` (`CDN_BASE`) and can be overridden with the `CDN_BASE` environment variable.

## Usage (step by step)
Every command takes `--skin` (default `karambit/gamma-doppler/phase-3`). `process` is the everyday path. The commands below are the single steps, plus the research steps (`cluster`, `eval`, `all`) that compare all four models and clustering methods. Those were used to pick DINOv2, and are worth running again on a very different skin to check that DINOv2 still wins.
```
python run.py scrape            # download renders to data/<skin>/raw/
python run.py prep              # check data/<skin>/processed/_diagnostics/mask_overlay.png
python run.py model-inputs      # optional: save crops at each model's input size
python run.py embed             # all models (or --models dinov2)
python run.py cluster
python run.py eval              # writes data/<skin>/results/report.html
python run.py similar --self-tol   # per-pattern lists of close look-alikes
python run.py plot              # 2D maps (interactive HTML + PNG)
python run.py label             # hand-label ~100 pairs, then rerun `similar` to fit the floor
```

### Per-pattern look-alike lists (`run.py similar`)
Each seed gets its own list of close look-alikes. How long the list is depends on the pattern, not on a fixed k:
1. **Floor:** a match must have similarity of at least `SIM_FLOOR` (0.60 by default, or fitted from your labels).
2. **Near-identical:** seeds at least as similar as the pattern is to slightly shifted copies of itself are always kept. With `--self-tol` this level is measured for each seed.
3. **Biggest drop:** the list ends at the largest drop in similarity, if that drop is bigger than shift noise (`SIM_MIN_GAP`).
4. **Two-way:** if A lists B, B also lists A.
5. **Loose tier:** seeds with fewer than `SIM_LOOSE_FILL` (3) matches get their next nearest seeds down to `SIM_LOOSE_FLOOR` (0.50) added, marked `~`.

**Groups.** The lists are then turned into groups, with each seed in at most one group. Groups are merged step by step (average linkage), and two groups can only merge if one has a seed on the other's lists and the average similarity across all pairs in the merged group stays at or above `GROUP_MIN_AVG` (0.50). That limit stops long chains from forming. A seed ends up in the group it fits best on average, which may not be the group of its single closest match. Seeds that fit no group get `-1`.

**Naming groups.** Type names into the `name` column of `data/<skin>/labels/group_names.csv`. On each rerun, a name moves to whichever new group shares at least half its members with the named one, so names survive small changes. Names whose group fell apart stay at the bottom with group `-1`.

`python run.py label` shows pairs from every similarity level. Answer y/n for about 100 pairs, then rerun `python run.py similar`: it fits the floor so that a pair at the floor is judged alike 50% of the time.
`notebooks/explore.ipynb` lets you browse groups and look-alikes for any seed.

Outputs (all under `data/<skin>/`, which git ignores):
- `results/metrics.csv`: every model × method with its scores
- `results/<model>/groups.csv`: one row per group: name, size, cohesion (average similarity across all pairs), members
- `results/<model>/similar_summary.csv`: one row per seed with its group and look-alike list (`*` = near-identical, `~` = loose)
- `results/<model>/similar.csv`: one row per (seed, look-alike) with similarity and tier
- `results/<model>/map_umap.html`: interactive 2D map coloured by group; search a seed to see its group and list
- `results/<model>/neighbors.csv`: top 10 look-alikes per seed with similarity
- `results/<model>/<method>/labels.csv`: group per seed (`-1` = unique)
- `results/report.html`: visual report for the best run
