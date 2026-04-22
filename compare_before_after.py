"""
compare_before_after.py
─────────────────────────────────────────────────────────────────────
Visualises the performance gap between:
  - Baseline  : raw columns only (distance, hour, passenger_count)
  - Engineered: full 35 spatial-temporal features

Uses the actual numbers from the real Kaggle Uber Fares dataset.
Run: python compare_before_after.py
─────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Actual results from real Kaggle dataset ───────────────────────────
MODELS = ["Linear\nRegression", "Ridge", "Random\nForest", "XGBoost", "LightGBM", "LightGBM Tuned"]

BEFORE = {
    "train_rmse": [9.713, 9.713, 3.210, 2.489, 2.492, 2.512],
    "test_rmse":  [9.531, 9.531, 4.475, 2.822, 2.734, 2.754],
    "r2":         [0.8232, 0.8232, 0.9610, 0.9845, 0.9855, 0.9843],
}

AFTER = {
    "train_rmse": [2.721, 2.721, 1.756, 1.275, 1.348, 1.429],
    "test_rmse":  [2.680, 2.680, 2.178, 1.651, 1.640, 1.615],
    "r2":         [0.9862, 0.9862, 0.9909, 0.9948, 0.9948, 0.9950],
}

# Derived
improvement_pct = [
    round((b - a) / b * 100, 1)
    for b, a in zip(BEFORE["test_rmse"], AFTER["test_rmse"])
]
r2_gain = [round(a - b, 4) for b, a in zip(BEFORE["r2"], AFTER["r2"])]

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#FAFAFA",
    "axes.facecolor":    "#FFFFFF",
    "axes.grid":         True,
    "grid.color":        "#EBEBEB",
    "grid.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "font.family":       "DejaVu Sans",
    "figure.dpi":        130,
})

C_BEFORE  = "#F59E0B"   # amber  — baseline
C_AFTER   = "#2563EB"   # blue   — engineered
C_GREEN   = "#10B981"
C_RED     = "#EF4444"
C_PURPLE  = "#8B5CF6"
C_GRAY    = "#9CA3AF"

x = np.arange(len(MODELS))
W = 0.35


# ══════════════════════════════════════════════════════════════════════
# PLOT 1 — RMSE & R² side-by-side bars
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Feature Engineering Impact: Baseline vs 35 Engineered Features",
             fontsize=14, fontweight="bold", y=1.01)

# RMSE
ax = axes[0]
b1 = ax.bar(x - W/2, BEFORE["test_rmse"], W,
            label="Baseline (6 raw cols)", color=C_BEFORE, alpha=0.85, edgecolor="white")
b2 = ax.bar(x + W/2, AFTER["test_rmse"],  W,
            label="Engineered (35 features)", color=C_AFTER, alpha=0.85, edgecolor="white")

for bar, val in zip(b1, BEFORE["test_rmse"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"${val:.2f}", ha="center", fontsize=8, color=C_BEFORE, fontweight="bold")

for bar, val, imp in zip(b2, AFTER["test_rmse"], improvement_pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"${val:.2f}", ha="center", fontsize=8, color=C_AFTER, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.9,
            f"↓{imp:.0f}%", ha="center", fontsize=8, color=C_GREEN, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=8.5)
ax.set_title("Test RMSE — Lower is Better")
ax.set_ylabel("RMSE ($)")
ax.legend(framealpha=0.9, fontsize=9)

# R²
ax = axes[1]
b3 = ax.bar(x - W/2, BEFORE["r2"], W,
            label="Baseline", color=C_BEFORE, alpha=0.85, edgecolor="white")
b4 = ax.bar(x + W/2, AFTER["r2"],  W,
            label="Engineered", color=C_AFTER, alpha=0.85, edgecolor="white")

for bar, val in zip(b3, BEFORE["r2"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", fontsize=7.5, color=C_BEFORE, fontweight="bold")

for bar, val, gain in zip(b4, AFTER["r2"], r2_gain):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", fontsize=7.5, color=C_AFTER, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.009,
            f"+{gain:.4f}", ha="center", fontsize=7, color=C_GREEN, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=8.5)
ax.set_ylim(0.78, 1.02)
ax.set_title("Test R² — Higher is Better")
ax.set_ylabel("R²")
ax.legend(framealpha=0.9, fontsize=9)

fig.tight_layout()
p1 = f"{PLOT_DIR}/13_rmse_r2_comparison.png"
fig.savefig(p1, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {p1}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 2 — RMSE improvement % + Train vs Test gap
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RMSE Reduction from Feature Engineering",
             fontsize=14, fontweight="bold", y=1.01)

# Improvement %
ax = axes[0]
colors_bar = [C_GREEN if v >= 50 else C_AFTER for v in improvement_pct]
bars = ax.bar(MODELS, improvement_pct, color=colors_bar,
              edgecolor="white", alpha=0.9)
ax.axhline(50, color=C_GRAY, lw=1.2, ls="--", alpha=0.7, label="50% threshold")
ax.set_title("Test RMSE Reduction (%)")
ax.set_ylabel("Improvement (%)")
ax.set_ylim(0, 85)
ax.legend(fontsize=9)
for bar, val in zip(bars, improvement_pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold", color="white"
            if val > 20 else C_GRAY)

# Train vs Test gap (overfitting check) — after engineering
ax = axes[1]
train_vals = AFTER["train_rmse"]
test_vals  = AFTER["test_rmse"]
gap        = [round(t - r, 3) for r, t in zip(train_vals, test_vals)]

ax.bar(x - W/2, train_vals, W, label="Train RMSE",
       color=C_PURPLE, alpha=0.8, edgecolor="white")
ax.bar(x + W/2, test_vals,  W, label="Test RMSE",
       color=C_AFTER, alpha=0.8, edgecolor="white")

for i, (tr, te, g) in enumerate(zip(train_vals, test_vals, gap)):
    ax.annotate("", xy=(i + W/2, te), xytext=(i + W/2, tr),
                arrowprops=dict(arrowstyle="-", color=C_RED, lw=1.2, ls="dashed"))
    ax.text(i + W/2 + 0.08, (tr + te) / 2, f"Δ{g:.2f}",
            fontsize=7.5, color=C_RED, va="center")

ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=8.5)
ax.set_title("Train vs Test RMSE (Engineered) — Overfitting Check")
ax.set_ylabel("RMSE ($)")
ax.legend(framealpha=0.9, fontsize=9)

fig.tight_layout()
p2 = f"{PLOT_DIR}/14_improvement_overfit_check.png"
fig.savefig(p2, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {p2}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 3 — Full summary dashboard
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#F1F5F9")
gs = gridspec.GridSpec(3, 4, hspace=0.52, wspace=0.4,
                       top=0.89, bottom=0.08, left=0.07, right=0.97)
fig.suptitle("Uber Fare Prediction — Before vs After Feature Engineering",
             fontsize=15, fontweight="bold", y=0.96)

# ── KPI cards (row 0) ─────────────────────────────────────────────────
best_before  = min(BEFORE["test_rmse"])
best_after   = min(AFTER["test_rmse"])
best_r2_bef  = max(BEFORE["r2"])
best_r2_aft  = max(AFTER["r2"])
max_imp      = max(improvement_pct)
lin_imp      = improvement_pct[0]

kpis = [
    (f"${best_before:.3f}", f"→  ${best_after:.3f}", "Best Test RMSE", "#2563EB"),
    (f"{best_r2_bef:.4f}",  f"→  {best_r2_aft:.4f}", "Best R²",        "#10B981"),
    (f"{max_imp:.1f}%",     "RMSE reduction",        "Max Improvement", "#8B5CF6"),
    (f"{lin_imp:.1f}%",     "Linear model gain",     "Linear RMSE ↓",  "#F59E0B"),
]
for i, (top, bot, lbl, col) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(col)
    ax.text(0.5, 0.68, top, ha="center", va="center", fontsize=14,
            fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.5, 0.42, bot, ha="center", va="center", fontsize=9.5,
            color="white", alpha=0.88, transform=ax.transAxes)
    ax.text(0.5, 0.16, lbl, ha="center", va="center", fontsize=8.5,
            color="white", alpha=0.80, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

# ── RMSE comparison bars (row 1, left) ───────────────────────────────
ax1 = fig.add_subplot(gs[1, :2])
ax1.bar(x - W/2, BEFORE["test_rmse"], W, label="Baseline",
        color=C_BEFORE, alpha=0.82, edgecolor="white")
ax1.bar(x + W/2, AFTER["test_rmse"],  W, label="Engineered",
        color=C_AFTER,  alpha=0.82, edgecolor="white")
ax1.set_xticks(x); ax1.set_xticklabels(MODELS, fontsize=8)
ax1.set_title("Test RMSE Comparison"); ax1.set_ylabel("RMSE ($)")
ax1.legend(fontsize=8, framealpha=0.9)
for i, (b, a) in enumerate(zip(BEFORE["test_rmse"], AFTER["test_rmse"])):
    ax1.text(i - W/2, b + 0.15, f"${b:.2f}", ha="center", fontsize=7, color=C_BEFORE)
    ax1.text(i + W/2, a + 0.15, f"${a:.2f}", ha="center", fontsize=7, color=C_AFTER)

# ── R² comparison bars (row 1, right) ────────────────────────────────
ax2 = fig.add_subplot(gs[1, 2:])
ax2.bar(x - W/2, BEFORE["r2"], W, label="Baseline",
        color=C_BEFORE, alpha=0.82, edgecolor="white")
ax2.bar(x + W/2, AFTER["r2"],  W, label="Engineered",
        color=C_AFTER,  alpha=0.82, edgecolor="white")
ax2.set_xticks(x); ax2.set_xticklabels(MODELS, fontsize=8)
ax2.set_ylim(0.78, 1.02)
ax2.set_title("Test R² Comparison"); ax2.set_ylabel("R²")
ax2.legend(fontsize=8, framealpha=0.9)
for i, (b, a) in enumerate(zip(BEFORE["r2"], AFTER["r2"])):
    ax2.text(i - W/2, b + 0.003, f"{b:.3f}", ha="center", fontsize=7, color=C_BEFORE)
    ax2.text(i + W/2, a + 0.003, f"{a:.3f}", ha="center", fontsize=7, color=C_AFTER)

# ── RMSE improvement % (row 2, left) ─────────────────────────────────
ax3 = fig.add_subplot(gs[2, :2])
short = ["Linear", "Ridge", "RF", "XGBoost", "LGBM", "LGBM Tuned"]
colors_imp = [C_GREEN if v >= 50 else C_AFTER for v in improvement_pct]
ax3.bar(short, improvement_pct, color=colors_imp, edgecolor="white", alpha=0.9)
ax3.axhline(50, color=C_GRAY, lw=1.2, ls="--", alpha=0.6)
ax3.set_title("RMSE Reduction (%)"); ax3.set_ylabel("Improvement %")
ax3.set_ylim(0, 90)
for i, v in enumerate(improvement_pct):
    ax3.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

# ── Feature group breakdown (row 2, right) ────────────────────────────
ax4 = fig.add_subplot(gs[2, 2:])
feat_groups = {
    "Spatial core":      5,
    "Airport proximity": 7,
    "Cyclic encodings":  6,
    "Temporal flags":    4,
    "Interaction terms": 4,
    "Temporal raw":      4,
    "Passenger":         2,
    "City proximity":    4,
}
fg_colors = [C_AFTER, C_PURPLE, C_GREEN, C_BEFORE, C_RED, "#06B6D4", C_GRAY, "#EC4899"]
ax4.barh(list(feat_groups.keys()), list(feat_groups.values()),
         color=fg_colors, edgecolor="white", alpha=0.85)
ax4.set_title("35 Engineered Features by Group")
ax4.set_xlabel("Feature Count")
for i, (k, v) in enumerate(feat_groups.items()):
    ax4.text(v + 0.1, i, str(v), va="center", fontsize=9, fontweight="bold")

p3 = f"{PLOT_DIR}/15_full_comparison_dashboard.png"
fig.savefig(p3, bbox_inches="tight", dpi=140)
plt.close(fig)
print(f"Saved: {p3}")


# ── Console summary ───────────────────────────────────────────────────
print("\n" + "="*68)
print(f"  {'Model':<20} {'Before RMSE':>12} {'After RMSE':>11} {'Drop':>8} {'R² Gain':>9}")
print("="*68)
model_names = ["LinearRegression","Ridge","RandomForest","XGBoost","LightGBM","LightGBM_Tuned"]
for name, b, a, imp, gain in zip(model_names,
                                   BEFORE["test_rmse"], AFTER["test_rmse"],
                                   improvement_pct, r2_gain):
    print(f"  {name:<20} ${b:>10.3f}  ${a:>9.3f}  {imp:>6.1f}%  +{gain:>7.4f}")
print("="*68)
print(f"\n  Best model after engineering : XGBoost  RMSE $1.651  R² 0.9948")
print(f"  Max RMSE improvement         : {max(improvement_pct):.1f}% (LinearRegression)")

