# Project journal

Major changes, decisions and findings, newest first. Each entry says what changed, why, and what it led to.

---

## 2026-10-31: Next step under discussion: group map with up to 6 families
**Idea:** a 2D map with one pin per group (placed at the average of its members' embeddings) and the groups combined into at most 6 families. Directions on the map should mean something: neighbouring groups sit above, below or to the side depending on how the pattern shifts.

**Findings that shape it:**
- The seeds vary along only about **3.7 directions** (a nearest-neighbour estimate on the DINOv2 embeddings before whitening). This fits the game's pattern system: one fixed texture sheet, and the seed sets how far it shifts left/right and up/down and how much it rotates.
- A flat 2D layout (PCA) of the group centres keeps **32%** of the differences (PC1 21%, PC2 11%, PC3 8%).
- Splitting into 2–6 families works, but the boundaries are soft because the patterns form a smooth continuum. 6 families gives sizes 70/57/35/31/28/16.
- PCA on the whitened `z.npy` is meaningless (every direction comes out around 2%). Use the embeddings from before whitening for layouts.

**Plan:**
1. Group map from what we already have: PCA layout so directions are consistent, axis ends labelled with example patterns, families shown as coloured regions.
2. The pattern template: get each seed's texture shift and rotation, either from the game's formula or by rebuilding the texture sheet from the renders. The map then shows where on the texture each knife is cut from. The formula still needs a source and a check against our data.

---

## 2026-10-30: One command for any skin (`89a67b1`)
**Why:** the method has nothing specific to the Karambit Gamma Doppler. The mask is worked out from the images themselves, and the rest only needs a set of renders. But the skin was fixed in `config.py`, and every skin shared the same data folders.

**Changes:**
- `--skin knife/finish[/variant]` on every command. The name is read before `config.py` is loaded (`SKIN` environment variable).
- Each skin gets its own folder: `data/<knife>/<finish>/<variant>/{raw,processed,embeddings,results,labels}`. The existing Karambit data moved to `data/karambit/gamma-doppler/phase-3/`.
- An optional `settings.json` in a skin's folder overrides any setting from `config.py` for that skin.
- `run.py process`: download → mask → DINOv2 → centre + PCA → lists + groups → map, skipping steps already done; `--force` redoes everything except the download.
- The scraper stops if the first 10 seeds all fail, which means a wrong skin name.

**Checked:**
- The Karambit gave identical results after the move.
- 40 seeds of Karambit Marble Fade ran end to end, and the mask was found correctly.
- The CDN uses the same URL format for Fades, Dopplers, Case Hardeneds and the AK-47. The Butterfly is `butterfly-knife`.

## 2026-10-30: Loose matches and groups (`89a67b1`)
**Why:** an unusual seed like 960 had only one match (227, exactly at the 0.60 floor), even though 514 and 314 were only 0.01 below. The user also wanted one shared group per set of look-alikes, not overlapping per-seed lists.

**Changes:**
- **Loose tier:** a list shorter than 3 is topped up with the next nearest seeds, down to 0.50, marked `~`.
- **Groups:** each seed belongs to at most one group. Groups are merged by average linkage, with two rules:
  - Two groups can only merge if a seed of one appears on a list of a seed in the other.
  - The average similarity between all members must stay at or above `GROUP_MIN_AVG` (0.50).
  - Simply joining everything linked by any list chained seeds into groups of up to 56.
- **Group names:** typed into `labels/group_names.csv`. A name moves to whichever new group shares at least half its members, so it survives reruns.
- **Map:** coloured by group. Clicking a seed shows its group, its list and its loose matches.

**Result:** 237 groups covering 970 seeds (sizes 2: 53 groups, 3–5: 139, 6+: 45, largest 12), and 30 seeds in no group. Seed 960 is grouped with 104 and 487, because its average similarity to them is 0.54, against 0.50 for 227's group.

## 2026-10-30: Per-seed look-alike lists (`89a67b1`)
**Why:** the similarities form a continuum with no natural gap, so there is no single cutoff and no fixed k that suits every seed.
- Shifted copy of the same seed: median 0.84–0.89.
- A seed's closest other seed: median 0.75.
- Random pairs: median 0.

The k-means elbow is smooth, and silhouette levels off around 0.25.

**Changes:** `pipeline/similar.py` builds one list per seed with four rules:
1. **Floor:** similarity at least 0.60. Once there are about 40 labelled pairs, the floor is fitted from them instead.
2. **Near-identical:** seeds at or above the seed's own near-identical level are always kept. That level is the median similarity between the seed and 4 slightly shifted copies of itself, clipped to 0.75–0.92.
3. **Biggest drop:** the list is cut at the largest drop in similarity, if it is at least 0.05.
4. **Two-way:** if A lists B, B also lists A.

`label_tool.py` now samples pairs evenly from each similarity band (0.3–1.0). `pipeline/projection.py` draws 2D maps (UMAP, t-SNE, PCA) as an interactive HTML page.

**Decision:** the near-identical level uses the median shifted copy, not the worst one. The worst one marked 49% of seeds as having a near-identical twin; the median gives 14%.

## 2026-10-30: First pipeline (`cb7ed57`)
**Why:** the knife model, handle and background are identical in every render, so whole-image embeddings all look alike (mean cosine about 0.97).

**Changes:**
- **Scraper:** all 1000 seeds downloaded (1.8 GB, no duplicates).
- **Mask from the data:** the per-pixel variation across seeds finds the two patterned parts (blade and ring). They are stacked, at 896×1057.
- **Four models:** DINOv2, CLIP, ResNet50 and a colour histogram. Each embedding is averaged over shifted copies of the image.
- **Reduction:** subtract the mean, then L2 + PCA with whitening. Mean cosine drops from about 0.97 to about 0.
- **Clustering and evaluation:** HDBSCAN and agglomerative clustering; a stability check, a synthetic shift test and an HTML report.

**Result:** in the shift test, the shifted copy is the nearest seed (recall@1) for DINOv2 0.99, ResNet 0.93, CLIP 0.77 and colour histogram 0.34. **DINOv2 became the main model.**
