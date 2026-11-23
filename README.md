# Rock-exposure mapping and subglacial-drilling targeting in Antarctica

In the Lambert-Amery system, where is bedrock from which new cosmogenic-exposure constraints could be acquired located?  Are any such locations shallowly buried but reachable by drilling?

This script runs in two stages, each reported as it comes out.  It uses the SCAR rock-outcrop dataset, Landsat imagery, and Bedmap3.

---

## Results

Stage 1 did not outperform its baseline.  Stage 2 produced a usable shortlist.
I did not anticipate this.

### Stage 2 — drilling-site candidates (full results: RESULTS_stage2.md)

206 site candidates, of which 29 are both frozen-bedded and radar-measured.  No prior samples are reported within 10 km, the bed depth is sounded, and the thermal regime is agreed upon by independent methods.

Subglacial-drilling-site candidates in the Lambert–Amery system with their basal thermal states over bed
topography: figures/stage2_site_map.png

A model trained on bed shape (bed slope and local relief at 50 and 10 km) predicts where rock emerges with ROC-AUC 0.9225 under
spatially blocked cross-validation without information about the overlying ice.
Inverting that identifies near misses (still-covered bedrock that looks like outcrop terrain).

Unanticipated result:

| basal thermal state | candidate sites | ice sheet |
| frozen | **66%** | 18% |
| uncertain | 33% | 36% |
| thawed | **1%** | 46% |

1 of 206 sites sits over a thawed bed. The thermal classification comes from 9 ISMIP6 models and ≤120 m thickness filter was set
on drilling logistics with no thermal input.
Site reachability selects almost
exactly for the condition that makes its cosmogenic signal survive.

Best single candidate: 73.15°S, 63.97°E, 23 m of ice over a peak
standing 1500 m above its surroundings, 69 radar soundings, bed known 
±12 m, 48 km from nearest existing sample.

Full list: results/stage2_sites_final.csv

### Stage 1 — Landsat rock exposure (full results: RESULTS_stage1.md)

Precision–recall at true prevalence: figures/stage1_precision_recall.png

NDSI threshold beats the learned model at true prevalence:
F1 0.742 vs. 0.703, 500 false positives against 5033.

ROC-AUC = 0.9975. 0.68% prevalence, so we correct precision to the true prior
rather than from the stratified training table.

Most likely label circularity: SCAR outcrop polygons are
themselves threshold output, so, rather than compete with independent ground truth, a threshold baseline partly reproduces the
label-generating function rather. The learned model also has less information to work with, as USGS does
not generate the thermal band for this region.

---

## Three null results

The methodology section of this project is mostly null results.

Stage 2, first attempt: ROC-AUC 0.9957. Leakage. `ice_thickness` carried
5.6× the importance of any other feature because Bedmap3 derives thickness partly from the same rock mask the labels come from, and outcrop is where ice
thickness is zero. Every ice-derived feature is contaminated the
same way.

Stage 2, second attempt: `dist_to_water_km` dominated. Permutation
importance put it 10× above everything else.  The model had learned where
the PCMs are but not what a buried peak looks like.
Ablation showed this was redundancy (relief + slope alone
scores AUC 0.933), so we dropped it.

Stage-1 feature ranges exploded. NDSI came back [−12198, +17653] bounded by construction to [−1, 1]. Collection-2 surface
reflectance is unreliable over snow.  An ND with a negative denominator diverges, and 1-3% of visible-band pixels returned negative and 76-84% above
1.0, because LaSRC's aerosol retrieval assumes dark
targets and over-corrects. We now clip reflectance to [0, 1] before computing any index.

---

## Layout

```
src/aeml/            importable package
    features.py      texture, Landsat scaling, indices, QA masking   [no GDAL]
    sampling.py      blocked CV, stratified sampling, spatial blocks [no GDAL]
    landsat.py       MTL parsing, Collection 2 scene reading         [rasterio]
    labels.py        baseline, outcrop rasterisation, edge buffer    [rasterio]

scripts/             pipeline, in run order
    00_inspect_inputs.py      check every input before committing compute
    10_organise_landsat.py    flat band files, one directory per scene
    11_extract_stage1.py      polygons + scenes, per-pixel training table
    12_train_stage1.py        compare against baseline, train, evaluate
    20_extract_stage2.py      outcrop + Bedmap3 + velocity, 500 m stack
    21_add_thermal.py         attach basal thermal state
    22_rank_sites.py          cluster cells into sites, train, invert
    utils_parquet_to_npz.py   format conversion

config/lambert_amery.yml      all tunable parameters and study region
tests/test_features.py        27 tests, no GDAL/data required
docs/DATA.md                  what to download and from where
results/                      committed outputs, site-list product
figures/
```

`features.py` and `sampling.py` avoid raster dependencies, so
spatial splitting, scaling constants, class balancing, the QA snow-bit
trap, and index computation are testable
without downloading scenes.

```bash
conda env create -f environment.yml && conda activate aeml
python tests/test_features.py          # 27/27, no data needed
```

Data is not redistributed.  `data/` is gitignored. See
docs/DATA.md.

---

## Methodological notes

Spatially blocked splits: Adjacent pixels are not independent.
Blocks are stratified by whether they contain any positives (most do not), and splits are by whole block (25 km at stage 2, 10 km at stage 1 where
features are rougher).

Precision at true prevalence: Training uses stratified sampling.  We inflate measured precision by the ratio of the odds and 
correct it back via the likelihood ratio. ROC and recall are
prevalence-invariant.  Precision is not prevalence-invariant. Accuracy is never reported.  At
0.68% prevalence, predicting "no rock" everywhere scores 99.3%.

Feature selection by ablation, not importance: See above.

Physical checks before ranking: A stage-2 candidate must have
positive bed relief (a peak) and bed uncertainty smaller than
the overlying-ice thickness (an earlier ranking put a cell 112 m below its
surroundings).

Sites are ranked radar-surveyed first. With 91% of the grounded bed
never sounded, a confident score over interpolated terrain might only reflect Bedmap3's interpolation.

---

## Data

None redistributed here.  All separately cited and openly licensed and separately cited.

- **Rock outcrop** — Burton-Johnson, A., Black, M., Fretwell, P. T., &
  Kaluza-Gilbert, J. (2016). *The Cryosphere*, 10, 1665–1677.
  doi:10.5285/f7947381-6fd7-466f-8894-25d3262cbcf5
- **Bedmap3** — Pritchard, H. D., et al. (2025). *Scientific Data*.
  doi:10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2
- **Ice velocity** — MEaSUREs Phase-Based Antarctica Ice Velocity Map V1,
  NSIDC-0754
- **Basal thermal state** — Seiner, O., et al. *A synthesis of the basal
  thermal state of the Antarctic ice sheet*, *Journal of Glaciology*.
  doi:10.5281/zenodo.15556691
- **Landsat 8/9 Collection 2 Level-2** — USGS EarthExplorer
- **ICE-D:ANTARCTICA** — cosmogenic-nuclide sample database

---

## Limitations

- Stage 1 rests on two Landsat scenes, one supplying 91% of rows (not
  enough for generalization).
- No thermal band. USGS does not generate the Level-2 surface-temperature
  product for this region. Burton-Johnson used thermal.  This model does not use thermal.
- Surface reflectance is the wrong product over ice. Clipping discards variation. Level-1 TOA is the planned fix and correct
  input.
- 91% of the grounded bed is unsounded. The two-tier ranking does not solve this.
- Basal thermal state is 8 km data on 500 m candidates (regional
  setting, not site-specific).
- Outcrop presence is a proxy. The model learns where rock is exposed
  today, which reflects climatic and erosional history and bed shape.
  A buried peak that resembles an exposed one is unverified.
- No field validation. Nothing here has been tested against a drilling
  campaign.

## License

MIT.
