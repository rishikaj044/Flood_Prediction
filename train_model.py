"""
Flood Prediction System — Model Training Script
================================================
Trains a Random Forest Classifier on the India flood risk dataset,
evaluates it, saves the model, and exports visualisation charts.

Run from the project root:
    python model/train_model.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
import pickle

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "dataset", "flood_data.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "saved_model", "model.pkl")
IMAGES_DIR  = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
C_BLUE   = "#1A73E8"
C_RED    = "#E53935"
C_AMBER  = "#FB8C00"
C_GREEN  = "#43A047"
C_DARK   = "#1C1C2E"
C_LIGHT  = "#F5F7FA"

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  FLOOD PREDICTION MODEL — TRAINING PIPELINE")
print("=" * 60)

print("\n[1/6] Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"      Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f"      Missing values: {df.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/6] Cleaning data …")
before = len(df)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
print(f"      Removed {before - len(df)} duplicate / null rows")
print(f"      Clean rows: {len(df):,}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/6] Selecting features …")

FEATURES = [
    "Rainfall (mm)",
    "Humidity (%)",
    "River Discharge (m³/s)",
    "Water Level (m)",
    "Elevation (m)",
    "Historical Floods",
]
TARGET = "Flood Occurred"

X = df[FEATURES]
y = df[TARGET]

print(f"      Features : {FEATURES}")
print(f"      Target   : {TARGET}")
print(f"      Class balance — No Flood: {(y==0).sum():,}  |  Flood: {(y==1).sum():,}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/6] Splitting dataset (80 / 20) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — TRAIN RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/6] Training Random Forest Classifier …")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("      Training complete.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════
print("\n[6/6] Evaluating model …")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc     = accuracy_score(y_test, y_pred)
print(f"\n      ✅ Accuracy : {acc * 100:.2f}%")
print("\n      Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))

# ══════════════════════════════════════════════════════════════════════════
# SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════
print(f"\n💾  Saving model → {MODEL_PATH}")
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": model, "features": FEATURES}, f)
print("    Model saved successfully.")

# ══════════════════════════════════════════════════════════════════════════
# GENERATE CHARTS
# ══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": C_LIGHT,
    "axes.facecolor":   C_LIGHT,
    "axes.edgecolor":   "#CCCCCC",
    "axes.labelcolor":  C_DARK,
    "xtick.color":      C_DARK,
    "ytick.color":      C_DARK,
    "text.color":       C_DARK,
    "font.family":      "sans-serif",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

# ── Chart 1: Flood Occurrence ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df[TARGET].value_counts()
bars = ax.bar(["No Flood", "Flood"], counts.values,
              color=[C_GREEN, C_RED], width=0.5, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.set_title("Flood Occurrence Distribution", fontweight="bold", pad=12)
ax.set_ylabel("Number of Records")
ax.set_ylim(0, counts.max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "flood_occurrence.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Chart saved: flood_occurrence.png")

# ── Chart 2: Rainfall vs Water Level ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sample = df.sample(n=min(2000, len(df)), random_state=42)
colors = [C_RED if v == 1 else C_BLUE for v in sample[TARGET]]
ax.scatter(sample["Rainfall (mm)"], sample["Water Level (m)"],
           c=colors, alpha=0.35, s=18, edgecolors="none")
ax.set_title("Rainfall vs Water Level  (coloured by flood outcome)", fontweight="bold", pad=12)
ax.set_xlabel("Rainfall (mm)")
ax.set_ylabel("Water Level (m)")
ax.spines[["top", "right"]].set_visible(False)
legend_handles = [
    mpatches.Patch(color=C_RED,  label="Flood"),
    mpatches.Patch(color=C_BLUE, label="No Flood"),
]
ax.legend(handles=legend_handles, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "rainfall_vs_waterlevel.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Chart saved: rainfall_vs_waterlevel.png")

# ── Chart 3: Feature Importance ─────────────────────────────────────────
importances = model.feature_importances_
feat_series = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 4))
colors_bar = [C_RED if v == feat_series.max() else C_BLUE for v in feat_series.values]
bars = ax.barh(feat_series.index, feat_series.values,
               color=colors_bar, edgecolor="white", height=0.6)
for bar, val in zip(bars, feat_series.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_title("Feature Importance (Random Forest)", fontweight="bold", pad=12)
ax.set_xlabel("Importance Score")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Chart saved: feature_importance.png")

# ── Chart 4: Confusion Matrix ────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No Flood", "Flood"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Chart saved: confusion_matrix.png")

# ── Chart 5: River Discharge by Flood Outcome ────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
flood_vals    = df[df[TARGET] == 1]["River Discharge (m³/s)"]
no_flood_vals = df[df[TARGET] == 0]["River Discharge (m³/s)"]
ax.hist(no_flood_vals, bins=40, alpha=0.6, color=C_BLUE,  label="No Flood", edgecolor="none")
ax.hist(flood_vals,    bins=40, alpha=0.6, color=C_RED,   label="Flood",    edgecolor="none")
ax.set_title("River Discharge Distribution by Flood Outcome", fontweight="bold", pad=12)
ax.set_xlabel("River Discharge (m³/s)")
ax.set_ylabel("Frequency")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "river_discharge_dist.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Chart saved: river_discharge_dist.png")

print("\n" + "=" * 60)
print("  TRAINING PIPELINE COMPLETE")
print(f"  Model accuracy : {acc * 100:.2f}%")
print(f"  Saved model    : {MODEL_PATH}")
print(f"  Charts saved   : {IMAGES_DIR}/")
print("=" * 60 + "\n")
