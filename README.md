# AI4MultiGIS — Invasive Crayfish Risk Projection

> **A Multimodal Federated Framework for Intelligent Geospatial Management:
> Projecting the Risk of Invasive Crayfish in European River Systems**
>
> *Fatima Chahal, Kahina Alitouche, Akram Hakiri, Richard Chbeir*
> University of Pau & Pays de l'Adour (UPPA) — LIUPPA Lab
> SN Computer Science, Springer — 2025

[![Project Website](https://img.shields.io/badge/Project-ai4multigis.eu-blue)](https://www.ai4multigis.eu/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![QGIS](https://img.shields.io/badge/QGIS-3.28+-green)](https://qgis.org/)

---

## Overview

This repository contains the full implementation supporting the journal paper.
The framework validates three claims on the
[World of Crayfish (WoC)](https://www.worldofcrayfish.org) biodiversity database,
focusing on the prediction and spatial projection of *Faxonius limosus*
(spiny-cheek crayfish) invasion risk across European river networks,
with a targeted habitat suitability projection onto Romania.

**Three research questions are addressed:**

- **RQ1** — Does multimodal fusion of spatial, temporal, and genetic data
  improve invasive species prediction over single-modality baselines?
- **RQ2** — Can federated learning preserve contributor privacy while
  maintaining competitive predictive accuracy?
- **RQ3** — Can a QGIS plugin make the pipeline accessible to ecologists
  without machine learning expertise?

---

## Repository Structure

```
ai4multigis/
├── pilot2_exp.py                  # RQ1 + RQ2: ablation study and FL simulation
├── romania_risk_projection.py     # Habitat suitability projection onto Romania
├── inspect_romania_data.py        # Utility: inspect terrain data files
│
├── qgis_plugin/
│   └── ai4multigis_plugin/
│       ├── __init__.py            # QGIS plugin entry point
│       ├── metadata.txt           # Plugin metadata
│       └── ai4multigis_plugin.py  # Plugin logic and UI
│
├── data/
│   └── database-WoC1.2.xlsx      # World of Crayfish occurrence database
│
├── results/                       # Generated outputs (figures, tables, model)
│   ├── model_B4_full.pkl          # Trained Random Forest model (B4)
│   ├── encoder_COI_prefix.pkl     # COI prefix label encoder
│   ├── table1_ablation.csv        # Table 1: ablation study results
│   ├── table2_federated.csv       # Table 2: FL comparison
│   ├── table3_romania_hotspots.csv# Table 3: top-risk Romanian rivers
│   ├── fig1_ablation.png          # Figure 1: ablation bar chart
│   ├── fig2_feature_importance.png# Figure 2: feature importance
│   ├── fig3_federated.png         # Figure 3: FL comparison
│   ├── fig4_temporal_spread.png   # Figure 4: temporal spread of F. limosus
│   ├── romania_risk_map.png       # Figure 5: Romania invasion risk map
│   ├── romania_risk_map_hires.png # Figure 5 (300 DPI for submission)
│   └── romania_river_risk.geojson # Risk layer — open directly in QGIS
│
└── README.md
```

---

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl joblib
```

For the Romania projection script, you additionally need:

```bash
pip install geopandas shapely
```

**Tested on:** Python 3.12 · Ubuntu 24.04 (WSL) · QGIS 3.40

---

## Usage

### 1 — Run the Experiments (RQ1 and RQ2)

Place `database-WoC1.2.xlsx` in the same folder as the script, then:

```bash
python pilot2_exp.py
```

This produces in `results/`:
- `table1_ablation.csv` and `fig1_ablation.png` — ablation study (RQ1)
- `fig2_feature_importance.png` — feature importance for the B4 model
- `table2_federated.csv` and `fig3_federated.png` — FL comparison (RQ2)
- `fig4_temporal_spread.png` — temporal spread analysis
- `model_B4_full.pkl` and `encoder_COI_prefix.pkl` — trained model artefacts

### 2 — Generate the Romania Risk Map

You need the three Romania terrain files (not included due to size):
- `rivers-romania.geojson` — river network from OpenStreetMap
- `contours-romania.geojson` — 10m elevation contours from SRTM
- `elevation-30m-romania.tif` — 30m SRTM digital elevation model

Then run:

```bash
python romania_risk_projection.py
```

This produces `results/romania_risk_map.png` and `results/romania_river_risk.geojson`.

### 3 — Inspect Terrain Data

```bash
python inspect_romania_data.py
```

### 4 — Install the QGIS Plugin

```bash
# Copy plugin folder to QGIS plugins directory
cp -r qgis_plugin/ai4multigis_plugin/ \
  ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/
```

Then in QGIS: **Plugins → Manage and Install Plugins → Installed →
enable "AI4MultiGIS Invasion Risk"**.

The plugin loads `results/romania_river_risk.geojson`, styles it by
risk level, and provides filtering and export in four steps.

---

## Data Sources

| Dataset | Source | Licence |
|---|---|---|
| World of Crayfish (WoC) | [worldofcrayfish.org](https://www.worldofcrayfish.org) | Open data |
| Romania river network | [OpenStreetMap](https://www.openstreetmap.org) via QuickOSM | ODbL |
| Elevation contours / DEM | [OpenTopography SRTM GL1](https://doi.org/10.5069/G9445JDF) | Public domain |

> **Note on terrain files:** The Romania GeoJSON and GeoTIFF files are not
> included in this repository due to their size (55 MB, 3 GB, and 58 MB
> respectively). Download instructions are provided in the scripts.

---

## Results Summary

| Claim | Configuration | AUC-ROC | F1 |
|---|---|---|---|
| RQ1 — Spatial only (baseline) | B1 | 0.829 | 0.815 |
| RQ1 — Full multimodal (best) | B4 | **0.853** | 0.768 |
| RQ2 — Centralised (upper bound) | — | 0.853 | 0.768 |
| RQ2 — Federated (FedAvg) | — | 0.811 | 0.760 |
| RQ2 — Local-only | — | 0.795 | 0.869 |

The federated model achieves near-centralised accuracy (gap: 0.008 F1)
while preserving the privacy of 32 confidential occurrence records.

---

## Project

This work is part of the **AI4MultiGIS** project
([ai4multigis.eu](https://www.ai4multigis.eu/)),
funded by the EU Horizon Europe programme through the CHIST-ERA 2023 call
(project ID: CHIST-ERA-23-MultiGIS-01, grant agreement No. EP/Z003490/1).

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{chahal2025multigis,
  author    = {Chahal, Fatima and Alitouche, Kahina and
               Hakiri, Akram and Chbeir, Richard},
  title     = {A Multimodal Federated Framework for Intelligent Geospatial
               Management: Projecting the Risk of Invasive Crayfish in
               European River Systems},
  journal   = {SN Computer Science},
  publisher = {Springer},
  year      = {2025},
}
```

---

## Authors

| Name | ORCID |
|---|---|
| Fatima Chahal | [0009-0001-6100-2945](https://orcid.org/0009-0001-6100-2945) |
| Kahina Alitouche | [0009-0002-3509-1144](https://orcid.org/0009-0002-3509-1144) |
| Akram Hakiri | [0000-0001-7151-5499](https://orcid.org/0000-0001-7151-5499) |
| Richard Chbeir | [0000-0003-4112-1426](https://orcid.org/0000-0003-4112-1426) |

---

## Licence

This repository is released under the [MIT Licence](LICENSE).
The WoC dataset is subject to its own open data terms — see the
[World of Crayfish](https://www.worldofcrayfish.org) website.# ai4multigis-crayfish-invasion
Multimodal federated framework for invasive crayfish risk projection — journal extension
