# Data acquisition

Everything here is free and openly licensed, but none of it is redistributed
in this repository. `data/` is gitignored. Download into the layout below.

```
data/
├── landsat/                      # untarred Landsat L2 scene directories
│   └── LC08_L2SP_127110_20200115_20200824_02_T1/
├── add/                          # SCAR Antarctic Digital Database
│   └── Rock_outcrop_medium_res_polygon.{shp,dbf,shx,prj}
├── bedmap3/
├── rema/
├── velocity/
├── iced/
└── processed/                    # pipeline outputs; also gitignored
```

Everything in the Lambert-Amery region is natively in **EPSG:3031**
(WGS 84 / Antarctic Polar Stereographic) — Landsat scenes south of ~63°S,
Bedmap3, REMA and the ADD layers all share it. Check rather than assume, but
in practice this means no reprojection between stages.

---

## Stage 1

### 1. Rock outcrop polygons — the labels

Burton-Johnson, A., Black, M., Fretwell, P. T., & Kaluza-Gilbert, J. (2016).
*An automated methodology for differentiating rock from snow, clouds and sea
in Antarctica from Landsat 8 imagery: a new rock outcrop map and area
estimation for the entire Antarctic continent.* The Cryosphere, 10, 1665–1677.

Distributed through the **SCAR Antarctic Digital Database** (add.scar.org) as
"Rock outcrop" — take the medium-resolution polygon layer. Read the paper
before using the polygons; stage 1 is a comparison against this work and the
write-up needs to represent their method accurately.

### 2. Landsat 8/9 Collection 2 Level-2 scenes

From USGS EarthExplorer or landsatlook.usgs.gov. Filter on:

- **Product:** Collection 2 Level-2, **L2SP** — not L2SR. L2SR has no thermal
  band, and surface temperature carries much of the rock/snow signal. The
  loader raises rather than silently proceeding if ST_B10 is missing.
- **Season:** December–February. Outside midsummer, high solar zenith angles
  mean Level-2 surface reflectance may not be generated at all.
- **Cloud cover:** under ~20% scene-wide, though the QA mask handles the rest.
- **Tier 1** only, for geometric quality.

Start with **2–4 scenes over the Prince Charles Mountains** rather than the
whole region. That is enough to validate the pipeline end to end, and the PCM
has abundant outcrop so the rock class is well populated. Expand afterwards.

Each untarred scene is roughly 1 GB. Bands used: SR_B1–B7, ST_B10, QA_PIXEL.

---

## Stage 2

### 3. Bedmap3 — primary bed product

Pritchard, H. D., et al. (2025). *Bedmap3 updated ice bed, surface and
thickness gridded datasets for Antarctica.* Scientific Data.

From the BAS data centre (ramadda.data.bas.ac.uk). 500 m grid, NetCDF. Take
bed elevation, ice thickness, surface elevation, **and the uncertainty
grids** — the uncertainty is a feature in its own right, not an afterthought.

### 4. BedMachine Antarctica v3 — cross-check

NSIDC (NSIDC-0756). Mass-conservation thickness, often better near margins
where Bedmap is interpolation-dominated. Not the primary product here, but
worth having: disagreement between the two is itself informative about where
a candidate target is trustworthy.

### 5. REMA — surface topography

Polar Geospatial Center. **Take the 100 m or 200 m mosaic.** The 2 m product
is terabytes and nothing in stage 2 needs it. Used for surface slope and as a
cross-check on Bedmap3 surface elevation.

### 6. Ice surface velocity

MEaSUREs ITS_LIVE or the MEaSUREs Antarctic Ice Velocity Map, both via NSIDC.
Drives the cold-based filter: fast ice erases the cosmogenic signal in
subglacial bedrock, so fast-flowing areas are not useful drilling targets
regardless of how thin the ice is.

### 7. ICE-D:ANTARCTICA

You already have a querying pipeline for this. Needed for the
distance-to-existing-constraint feature and for validating stage 2 against
sites where subglacial or near-margin bedrock has already been sampled.

---

## Rough sizes

| Dataset | Approx. size |
|---|---|
| Landsat L2SP scene | ~1 GB each |
| ADD rock outcrop | < 100 MB |
| Bedmap3 (500 m, with uncertainty) | a few hundred MB |
| BedMachine v3 | ~1–2 GB |
| REMA 100 m mosaic, regional subset | a few hundred MB |
| Velocity mosaic | a few hundred MB |

A few scenes plus the stage-2 rasters is comfortably under 20 GB.

---

## Citation

Every dataset above requires citation. Collect the DOIs as you download and
put them in the repository README — a reviewer who finds an uncited dataset
will reasonably wonder what else was skipped.
