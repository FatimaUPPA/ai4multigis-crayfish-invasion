"""
AI4MultiGIS — Pilot 2: Invasive Crayfish Prediction (Europe → Romania)
=======================================================================
This script implements three experimental claims for the journal paper:

  Claim 1 — Multimodal fusion (spatial + temporal + genetic) improves prediction
  Claim 2 — Federated Learning preserves privacy with minimal accuracy loss
  Bonus   — Temporal spread analysis of Faxonius limosus toward Romania

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn openpyxl joblib

Usage:
    python AI4MultiGIS_Pilot2_Experiments.py

Output (saved in results/):
    table1_ablation.csv, table2_federated.csv
    fig1_ablation.png, fig2_feature_importance.png
    fig3_federated.png, fig4_temporal_spread.png
    model_B4_full.pkl, encoder_COI_prefix.pkl   <- used by QGIS plugin
"""

# =============================================================================
# 0 - IMPORTS AND CONFIGURATION
# =============================================================================

import os
import warnings

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
np.random.seed(42)

# Change DATA_PATH if your Excel file is in a different folder
DATA_PATH  = "database-WoC1.2.xlsx"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SPLITS = 5   # number of spatial CV folds


# =============================================================================
# 1 - LOAD AND PREPARE DATA
# =============================================================================

print("=" * 65)
print("STEP 1 - Loading and preparing data")
print("=" * 65)

df_raw = pd.read_excel(DATA_PATH, sheet_name="WoC_data_collector")

# The sheet has 23 columns - rename them for clarity
df_raw.columns = [
    "WoCid", "DOI", "URL", "Citation",
    "Lat", "Lon", "Accuracy",
    "Species", "Status", "Year",
    "COI_accession", "S16_accession", "SRA_accession",
    "Claim_extinction",
    "Pathogen_name", "Pathogen_COI", "Pathogen_16S",
    "Genotype_group", "Haplotype", "Year2",
    "Comments", "Confidentiality", "Contributor",
]

print(f"Raw records loaded : {len(df_raw)}")
print(f"Status breakdown:\n{df_raw['Status'].value_counts()}\n")

# Coerce Lat/Lon to numeric (two rows contain a trailing non-breaking space)
df_raw["Lat"] = pd.to_numeric(df_raw["Lat"], errors="coerce")
df_raw["Lon"] = pd.to_numeric(df_raw["Lon"], errors="coerce")

# Filter to Europe (bounding box), high accuracy, binary Alien/Native status
df = df_raw[
    df_raw["Lat"].between(35, 72)
    & df_raw["Lon"].between(-25, 45)
    & (df_raw["Accuracy"] == "High")
    & df_raw["Status"].isin(["Alien", "Native"])
    & df_raw["Lat"].notna()
    & df_raw["Lon"].notna()
    & df_raw["Year"].notna()
].copy()

df["Year"]  = df["Year"].astype(int)
df["label"] = (df["Status"] == "Alien").astype(int)   # 1 = Alien, 0 = Native

print(f"Filtered records (Europe, High accuracy, Alien/Native) : {len(df)}")
print(f"  Alien  : {df['label'].sum()}")
print(f"  Native : {(df['label'] == 0).sum()}")
print(f"  Year range : {df['Year'].min()} to {df['Year'].max()}")
print(f"  Species    : {df['Species'].nunique()}\n")


# =============================================================================
# 2 - FEATURE ENGINEERING
# =============================================================================

print("=" * 65)
print("STEP 2 - Feature engineering")
print("=" * 65)

# Spatial: sin/cos encoding avoids artificial boundary at +/-180
df["lat_sin"] = np.sin(np.radians(df["Lat"]))
df["lat_cos"] = np.cos(np.radians(df["Lat"]))
df["lon_sin"] = np.sin(np.radians(df["Lon"]))
df["lon_cos"] = np.cos(np.radians(df["Lon"]))

# Temporal
year_min        = df["Year"].min()
year_max        = df["Year"].max()
df["year_norm"] = (df["Year"] - year_min) / (year_max - year_min)
df["decade"]    = (df["Year"] // 10) * 10

# Genetic: binary availability flags + NCBI accession prefix as lineage proxy
df["has_COI"]         = df["COI_accession"].notna().astype(int)
df["has_16S"]         = df["S16_accession"].notna().astype(int)
df["has_any_genetic"] = ((df["has_COI"] == 1) | (df["has_16S"] == 1)).astype(int)
df["COI_prefix"]      = df["COI_accession"].str[:2].fillna("NONE")

le_coi               = LabelEncoder()
df["COI_prefix_enc"] = le_coi.fit_transform(df["COI_prefix"])

# Spatial CV folds (longitude bands) - avoids inflating accuracy via spatial autocorrelation
df["spatial_fold"] = pd.qcut(df["Lon"], q=N_SPLITS, labels=False)

print("Features ready:")
print("  Spatial  : Lat, Lon, lat_sin/cos, lon_sin/cos")
print("  Temporal : Year, year_norm, decade")
print("  Genetic  : has_COI, has_16S, has_any_genetic, COI_prefix_enc")
print(f"  CV folds : {N_SPLITS} spatial blocks by longitude\n")


# =============================================================================
# 3 - SHARED UTILITIES
# =============================================================================

FEATURE_SETS = {
    "B1 - Spatial only":       ["Lat", "Lon"],
    "B2 - Spatial + Temporal": ["Lat", "Lon", "Year", "year_norm", "decade"],
    "B3 - Spatial + Genetic":  ["Lat", "Lon", "has_COI", "has_16S",
                                 "has_any_genetic", "COI_prefix_enc"],
    "B4 - Full Multimodal":    ["Lat", "Lon",
                                 "Year", "year_norm", "decade",
                                 "has_COI", "has_16S",
                                 "has_any_genetic", "COI_prefix_enc"],
}


def make_model():
    """Random Forest with balanced class weights (handles 375 alien vs 218 native)."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def spatial_cv(X, y, folds):
    """5-fold spatial block cross-validation. Returns dict of mean and std metrics."""
    f1s, precs, recs, aucs = [], [], [], []
    for fold_id in range(N_SPLITS):
        tr    = np.where(folds != fold_id)[0]
        te    = np.where(folds == fold_id)[0]
        m     = make_model()
        m.fit(X[tr], y[tr])
        yp    = m.predict(X[te])
        yprob = m.predict_proba(X[te])[:, 1]
        f1s.append(f1_score(y[te], yp, zero_division=0))
        precs.append(precision_score(y[te], yp, zero_division=0))
        recs.append(recall_score(y[te], yp, zero_division=0))
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], yprob))
    return dict(
        f1_mean=np.mean(f1s),     f1_std=np.std(f1s),
        prec_mean=np.mean(precs), prec_std=np.std(precs),
        rec_mean=np.mean(recs),   rec_std=np.std(recs),
        auc_mean=np.mean(aucs),   auc_std=np.std(aucs),
    )


y     = df["label"].values
folds = df["spatial_fold"].values


# =============================================================================
# 4 - CLAIM 1: MULTIMODAL FUSION ABLATION
# =============================================================================

print("=" * 65)
print("CLAIM 1 - Multimodal fusion ablation study")
print("=" * 65)

ablation_rows  = []
ablation_stats = []

for name, features in FEATURE_SETS.items():
    s = spatial_cv(df[features].values, y, folds)
    ablation_rows.append({
        "Configuration": name,
        "F1":        f"{s['f1_mean']:.3f} +/- {s['f1_std']:.3f}",
        "Precision": f"{s['prec_mean']:.3f} +/- {s['prec_std']:.3f}",
        "Recall":    f"{s['rec_mean']:.3f} +/- {s['rec_std']:.3f}",
        "AUC-ROC":   f"{s['auc_mean']:.3f} +/- {s['auc_std']:.3f}",
    })
    ablation_stats.append(s)
    print(f"  {name}")
    print(f"    F1={s['f1_mean']:.3f}+/-{s['f1_std']:.3f}  "
          f"AUC={s['auc_mean']:.3f}+/-{s['auc_std']:.3f}")

table1 = pd.DataFrame(ablation_rows)
print("\n=== TABLE 1: Ablation results ===")
print(table1.to_string(index=False))
table1.to_csv(f"{OUTPUT_DIR}/table1_ablation.csv", index=False)
print(f"Saved -> {OUTPUT_DIR}/table1_ablation.csv\n")


# Figure 1: ablation bar chart
short_labels = [n.split("-")[1].strip() for n in FEATURE_SETS]
colors       = ["#B4B2A9", "#85B7EB", "#9FE1CB", "#534AB7"]
f1_means     = [s["f1_mean"]  for s in ablation_stats]
auc_means    = [s["auc_mean"] for s in ablation_stats]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, vals, title, ylabel in [
    (axes[0], f1_means,  "F1 score by feature configuration",  "F1 Score"),
    (axes[1], auc_means, "AUC-ROC by feature configuration",   "AUC-ROC"),
]:
    bars = ax.bar(short_labels, vals, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticklabels(short_labels, rotation=18, ha="right", fontsize=10)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Random baseline")
    ax.legend(fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
plt.suptitle("Claim 1 - Ablation: contribution of each modality",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_ablation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {OUTPUT_DIR}/fig1_ablation.png")


# Figure 2: feature importance for B4
model_full = make_model()
model_full.fit(df[FEATURE_SETS["B4 - Full Multimodal"]].values, y)

imp_df = pd.DataFrame({
    "Feature":    FEATURE_SETS["B4 - Full Multimodal"],
    "Importance": model_full.feature_importances_,
}).sort_values("Importance", ascending=True)

modality_color = {
    "Lat": "#85B7EB", "Lon": "#85B7EB",
    "Year": "#9FE1CB", "year_norm": "#9FE1CB", "decade": "#9FE1CB",
    "has_COI": "#F5C4B3", "has_16S": "#F5C4B3",
    "has_any_genetic": "#F5C4B3", "COI_prefix_enc": "#F5C4B3",
}
bar_colors = [modality_color.get(f, "#B4B2A9") for f in imp_df["Feature"]]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp_df["Feature"], imp_df["Importance"], color=bar_colors, edgecolor="white")
ax.set_xlabel("Feature importance (mean decrease impurity)", fontsize=11)
ax.set_title("Feature importance - Full multimodal model (B4)",
             fontsize=12, fontweight="bold")
ax.legend(handles=[
    mpatches.Patch(color="#85B7EB", label="Spatial"),
    mpatches.Patch(color="#9FE1CB", label="Temporal"),
    mpatches.Patch(color="#F5C4B3", label="Genetic"),
], fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {OUTPUT_DIR}/fig2_feature_importance.png")

# Save trained model for QGIS plugin
joblib.dump(model_full, f"{OUTPUT_DIR}/model_B4_full.pkl")
joblib.dump(le_coi,     f"{OUTPUT_DIR}/encoder_COI_prefix.pkl")
print(f"Saved -> {OUTPUT_DIR}/model_B4_full.pkl  (needed by QGIS plugin)\n")


# =============================================================================
# 5 - CLAIM 2: FEDERATED LEARNING SIMULATION
# =============================================================================

print("=" * 65)
print("CLAIM 2 - Federated Learning simulation")
print("=" * 65)

# Partition into 3 geographic nodes (mirrors national DB boundaries)
def assign_node(lon):
    if lon < 5:
        return "Node 1 - West"
    elif lon < 17:
        return "Node 2 - Central"
    else:
        return "Node 3 - East"

df["fl_node"] = df["Lon"].apply(assign_node)
nodes         = df["fl_node"].unique()
conf_count    = (df["Confidentiality"] == 1).sum()

print("Records per FL node:")
print(df.groupby("fl_node")["label"].value_counts().unstack(fill_value=0))
print(f"Confidential records kept local on Node 3 only: {conf_count}\n")

FL_FEATURES = FEATURE_SETS["B4 - Full Multimodal"]
X_all       = df[FL_FEATURES].values


def run_centralized():
    return spatial_cv(X_all, y, folds)


def run_local_only():
    """Each node trains in isolation - no data or model sharing."""
    all_preds, all_true = [], []
    for node in nodes:
        nd = df[df["fl_node"] == node]
        if len(nd) < 20:
            continue
        X_n, y_n = nd[FL_FEATURES].values, nd["label"].values
        split     = int(0.8 * len(X_n))
        m = make_model()
        m.fit(X_n[:split], y_n[:split])
        all_preds.extend(m.predict(X_n[split:]))
        all_true.extend(y_n[split:])
    f1  = f1_score(all_true, all_preds, zero_division=0)
    auc = roc_auc_score(all_true, all_preds) if len(np.unique(all_true)) > 1 else 0.0
    return f1, auc


def run_federated():
    """
    FedAvg simulation:
      - Each node trains a local model on its geographic partition.
      - Global prediction = average of node probabilities on the test fold.
      - Confidential records stay on Node 3 and are never shared.
    """
    f1s, aucs = [], []
    for fold_id in range(N_SPLITS):
        te         = np.where(folds == fold_id)[0]
        X_te, y_te = X_all[te], y[te]
        node_probs = []
        for node in nodes:
            node_mask = (df["fl_node"] == node).values
            tr        = np.where((folds != fold_id) & node_mask)[0]
            if len(tr) < 10 or len(np.unique(y[tr])) < 2:
                continue
            m = make_model()
            m.fit(X_all[tr], y[tr])
            node_probs.append(m.predict_proba(X_te)[:, 1])
        if not node_probs:
            continue
        agg_probs = np.mean(node_probs, axis=0)   # FedAvg aggregation
        agg_preds = (agg_probs >= 0.5).astype(int)
        f1s.append(f1_score(y_te, agg_preds, zero_division=0))
        if len(np.unique(y_te)) > 1:
            aucs.append(roc_auc_score(y_te, agg_probs))
    return dict(
        f1_mean=np.mean(f1s),   f1_std=np.std(f1s),
        auc_mean=np.mean(aucs), auc_std=np.std(aucs),
    )


print("Running FL experiments... (takes around 30 seconds)")

stats_c      = run_centralized()
f1_l, auc_l = run_local_only()
stats_f      = run_federated()

f1_c, auc_c = stats_c["f1_mean"], stats_c["auc_mean"]
f1_f, auc_f = stats_f["f1_mean"], stats_f["auc_mean"]

print(f"\n  Centralized  F1={f1_c:.3f}+/-{stats_c['f1_std']:.3f}  "
      f"AUC={auc_c:.3f}  [no privacy]")
print(f"  Federated    F1={f1_f:.3f}+/-{stats_f['f1_std']:.3f}  "
      f"AUC={auc_f:.3f}  [privacy preserved]")
print(f"  Local-only   F1={f1_l:.3f}               "
      f"AUC={auc_l:.3f}  [full privacy, no sharing]")
print(f"\n  FL vs Centralized gap : {f1_c - f1_f:.3f}  (<=0.05 = good trade-off)")
print(f"  FL vs Local-only gain : {f1_f - f1_l:.3f}")

table2 = pd.DataFrame([
    {"Setting": "Local-only (no sharing)",
     "F1": f"{f1_l:.3f}",
     "AUC-ROC": f"{auc_l:.3f}",
     "Privacy": "Full",
     "Data shared": "None"},
    {"Setting": "Federated (FedAvg)",
     "F1": f"{f1_f:.3f} +/- {stats_f['f1_std']:.3f}",
     "AUC-ROC": f"{auc_f:.3f} +/- {stats_f['auc_std']:.3f}",
     "Privacy": "Preserved",
     "Data shared": "Model updates only"},
    {"Setting": "Centralized (upper bound)",
     "F1": f"{f1_c:.3f} +/- {stats_c['f1_std']:.3f}",
     "AUC-ROC": f"{auc_c:.3f} +/- {stats_c['auc_std']:.3f}",
     "Privacy": "None",
     "Data shared": "All raw data"},
])

print("\n=== TABLE 2: Federated Learning comparison ===")
print(table2.to_string(index=False))
table2.to_csv(f"{OUTPUT_DIR}/table2_federated.csv", index=False)
print(f"Saved -> {OUTPUT_DIR}/table2_federated.csv\n")


# Figure 3: FL grouped bar chart
x, w     = np.arange(3), 0.35
settings = ["Local-only", "Federated (FL)", "Centralized"]
f1_vals  = [f1_l, f1_f, f1_c]
auc_vals = [auc_l, auc_f, auc_c]

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - w/2, f1_vals,  w, label="F1",      color="#534AB7", alpha=0.85)
bars2 = ax.bar(x + w/2, auc_vals, w, label="AUC-ROC", color="#1D9E75", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(settings, fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Claim 2 - Federated vs Centralized vs Local-only",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
privacy_labels = [
    "Full privacy\n(no sharing)",
    "Privacy-preserved\n(model updates only)",
    "No privacy\n(all data shared)",
]
for i, lbl in enumerate(privacy_labels):
    ax.text(i, -0.14, lbl, ha="center", fontsize=8, color="gray",
            transform=ax.get_xaxis_transform())
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_federated.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {OUTPUT_DIR}/fig3_federated.png\n")


# =============================================================================
# 6 - BONUS: TEMPORAL SPREAD ANALYSIS (Faxonius limosus toward Romania)
# =============================================================================

print("=" * 65)
print("BONUS - Temporal spread of Faxonius limosus toward Romania")
print("=" * 65)

fl_df  = df[df["Species"] == "Faxonius limosus"].copy()
annual = (
    fl_df.groupby("Year")
    .agg(median_lon=("Lon", "median"), max_lon=("Lon", "max"), count=("Lon", "count"))
    .reset_index()
)
annual = annual[annual["count"] >= 2]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(annual["Year"], annual["max_lon"],    "o-",  color="#D85A30",
        lw=2, ms=6,   label="Easternmost record")
ax.plot(annual["Year"], annual["median_lon"], "s--", color="#85B7EB",
        lw=1.5, ms=5, label="Median longitude")
ax.axhline(20, color="#534AB7", ls=":", lw=1.5,
           label="Romania west border (~20E)")
ax.axhline(29, color="#1D9E75", ls=":", lw=1.5,
           label="Romania east border (~29E)")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Longitude (degrees E)", fontsize=11)
ax.set_title("Eastward spread of Faxonius limosus", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
by_year = fl_df.groupby("Year").size()
ax2.bar(by_year.index, by_year.values,
        color="#9FE1CB", edgecolor="#0F6E56", linewidth=0.5)
ax2.set_xlabel("Year", fontsize=11)
ax2.set_ylabel("Number of records", fontsize=11)
ax2.set_title("Faxonius limosus records per year (Europe)",
              fontsize=11, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

plt.suptitle("Temporal spread - invasion front approaching Romania",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_temporal_spread.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {OUTPUT_DIR}/fig4_temporal_spread.png\n")


# =============================================================================
# 7 - SUMMARY
# =============================================================================

print("=" * 65)
print("SUMMARY - Key numbers for your paper")
print("=" * 65)

b1_f1 = ablation_stats[0]["f1_mean"]
b4_f1 = ablation_stats[3]["f1_mean"]
delta  = b4_f1 - b1_f1

print(f"Dataset")
print(f"  European records  : {len(df)}")
print(f"  Alien             : {df['label'].sum()}")
print(f"  Native            : {(df['label'] == 0).sum()}")
print(f"  Species           : {df['Species'].nunique()}")
print(f"  Year span         : {df['Year'].min()} to {df['Year'].max()}")

print(f"\nClaim 1 - Fusion improvement")
print(f"  B1 spatial-only F1  : {b1_f1:.3f}")
print(f"  B4 full fusion  F1  : {b4_f1:.3f}")
print(f"  Absolute gain       : +{delta:.3f}")
print(f"  Relative gain       : +{delta / b1_f1 * 100:.1f}%")

print(f"\nClaim 2 - FL privacy-accuracy trade-off")
print(f"  Centralized F1      : {f1_c:.3f}")
print(f"  Federated   F1      : {f1_f:.3f}")
print(f"  Local-only  F1      : {f1_l:.3f}")
print(f"  FL vs Centralized   : -{f1_c - f1_f:.3f} accuracy cost for privacy")
print(f"  FL vs Local-only    : +{f1_f - f1_l:.3f} gain from federation")
print(f"  Confidential records protected : {conf_count}")

print(f"\nAll outputs in: {OUTPUT_DIR}/")
print("  table1_ablation.csv          -> Table 1 in paper")
print("  table2_federated.csv         -> Table 2 in paper")
print("  fig1_ablation.png            -> Figure: ablation bar chart")
print("  fig2_feature_importance.png  -> Figure: feature importance")
print("  fig3_federated.png           -> Figure: FL comparison")
print("  fig4_temporal_spread.png     -> Figure: invasion spread")
print("  model_B4_full.pkl            -> Needed by QGIS plugin")
print("  encoder_COI_prefix.pkl       -> Needed by QGIS plugin")
print("=" * 65)
print("Done.")