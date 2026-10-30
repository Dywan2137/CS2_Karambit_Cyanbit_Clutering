# CS2_Karambit_Cyanbit_Clutering
Clustering for the CS2 Karambit Gamma Doppler Phase 3 knife. It groups patterns that look very similar, so players have alternatives to patterns that are already highly priced. Float and corner are not taken into account.

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

## Usage
```
python run.py scrape            # download renders to data/raw/...
python run.py prep              # check data/processed/_diagnostics/mask_overlay.png
python run.py model-inputs      # optional: save crops at each model's input size
python run.py embed             # all models (or --models dinov2)
python run.py cluster
python run.py eval              # writes data/results/report.html
python run.py label             # hand-label look-alike pairs, then rerun eval
```
`notebooks/explore.ipynb` lets you browse groups and look-alikes for any seed.

Outputs (all under `data/`, which git ignores):
- `results/metrics.csv`: every model × method with its scores
- `results/<model>/neighbors.csv`: top 10 look-alikes per seed with similarity
- `results/<model>/<method>/labels.csv`: group per seed (`-1` = unique)
- `results/report.html`: visual report for the best run
