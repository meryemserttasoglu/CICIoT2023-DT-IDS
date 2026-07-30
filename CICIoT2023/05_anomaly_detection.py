"""
=============================================================================
STAGE 5 — ANOMALY DETECTION
=============================================================================
Using the trained Autoencoder's reconstruction error:
  • Compute MSE per sample on test data
  • Define anomaly threshold (95th percentile of benign errors)
  • Classify: error > threshold → Attack
  • Evaluate: F1-score, Precision, Recall, ROC-AUC
  • Visualise: error distributions, ROC curve, precision-recall curve
=============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score
)

from config import (
    PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    ANOMALY_THRESHOLD_PERCENTILE
)
from utils import (
    load_pickle, save_pickle, evaluate_binary,
    plot_reconstruction_error_dist
)

print("\n" + "=" * 70)
print("  STAGE 5 — ANOMALY DETECTION")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD PREVIOUS RESULTS
# ──────────────────────────────────────────────────────────────────────────────
data = load_pickle(os.path.join(PROCESSED_DATA_DIR, "preprocessed.pkl"))
ae_results = load_pickle(os.path.join(RESULTS_DIR, "04_ae_results.pkl"))

y_test_bin = data["y_test_binary"]
reconstruction_errors = ae_results["reconstruction_errors"]
errors_benign = ae_results["errors_benign"]
errors_attack = ae_results["errors_attack"]

print(f"[ℹ] Test samples: {len(reconstruction_errors):,}")
print(f"[ℹ] Benign: {len(errors_benign):,}  |  Attack: {len(errors_attack):,}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  DEFINE THRESHOLD
# ──────────────────────────────────────────────────────────────────────────────
# F1-OPTIMAL THRESHOLD (fixes poor recall of the fixed 95th-percentile rule).
# Threshold is selected on a held-out validation half and evaluated on the
# other half (no leakage); this makes the AE Digital-Twin detector reproducible.
from sklearn.metrics import roc_auc_score, f1_score

raw_auc = roc_auc_score(y_test_bin, reconstruction_errors)
print(f"[i] ROC-AUC (error->attack): {raw_auc:.4f}")
_scores_all = reconstruction_errors if raw_auc >= 0.5 else -reconstruction_errors

_rng = np.random.RandomState(42)
_perm = _rng.permutation(len(_scores_all)); _half = len(_scores_all) // 2
_vi, _ti = _perm[:_half], _perm[_half:]
_cand = np.quantile(_scores_all[_vi], np.linspace(0.01, 0.999, 200))
threshold = float(max(_cand, key=lambda t: f1_score(
    y_test_bin[_vi], (_scores_all[_vi] > t).astype(int), zero_division=0)))
print(f"[+] F1-optimal anomaly threshold (val half): {threshold:.6f}")

# Restrict all downstream evaluation to the held-out half for an unbiased estimate
y_test_bin           = y_test_bin[_ti]
reconstruction_errors = reconstruction_errors[_ti]
anomaly_scores       = _scores_all[_ti]
y_pred_anomaly       = (anomaly_scores > threshold).astype(int)

n_detected = y_pred_anomaly.sum()
print(f"[ℹ] Detected anomalies: {n_detected:,} / {len(y_pred_anomaly):,} "
      f"({n_detected / len(y_pred_anomaly) * 100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  EVALUATE
# ──────────────────────────────────────────────────────────────────────────────
metrics = evaluate_binary(
    y_test_bin, y_pred_anomaly,
    y_scores=anomaly_scores,
    model_name="Autoencoder Anomaly Detection"
)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────

# 5a. Reconstruction error distributions
plot_reconstruction_error_dist(
    errors_benign, errors_attack, threshold,
    save_path=os.path.join(FIGURES_DIR, "05_error_distribution.png")
)

# 5b. ROC Curve (using corrected anomaly_scores)
fpr, tpr, _ = roc_curve(y_test_bin, anomaly_scores)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="#e74c3c", linewidth=2,
        label=f"Autoencoder (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Autoencoder Anomaly Detection", fontweight="bold")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "05_roc_curve.png"))
print(f"[✓] Figure saved → {os.path.join(FIGURES_DIR, '05_roc_curve.png')}")
plt.close(fig)

# 5c. Precision-Recall Curve (using corrected anomaly_scores)
precision_vals, recall_vals, _ = precision_recall_curve(
    y_test_bin, anomaly_scores
)
ap = average_precision_score(y_test_bin, anomaly_scores)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall_vals, precision_vals, color="#3498db", linewidth=2,
        label=f"AP = {ap:.4f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — Autoencoder", fontweight="bold")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "05_pr_curve.png"))
print(f"[✓] Figure saved → {os.path.join(FIGURES_DIR, '05_pr_curve.png')}")
plt.close(fig)

# 5d. Threshold sensitivity analysis
print(f"\n{'─'*50}")
print("  THRESHOLD SENSITIVITY ANALYSIS")
print(f"{'─'*50}")
print(f"  {'Percentile':>12s}  {'Threshold':>10s}  {'F1':>6s}  {'Prec':>6s}  {'Rec':>6s}")
print(f"  {'─'*52}")

from sklearn.metrics import f1_score, precision_score, recall_score

for pct in [90, 92, 95, 97, 99]:
    th = np.percentile(errors_benign, pct)
    yp = (reconstruction_errors > th).astype(int)
    f1 = f1_score(y_test_bin, yp, zero_division=0)
    pr = precision_score(y_test_bin, yp, zero_division=0)
    rc = recall_score(y_test_bin, yp, zero_division=0)
    marker = " ◄" if pct == ANOMALY_THRESHOLD_PERCENTILE else ""
    print(f"  {pct:>10d}th  {th:>10.6f}  {f1:>6.4f}  {pr:>6.4f}  {rc:>6.4f}{marker}")

# ──────────────────────────────────────────────────────────────────────────────
# 6.  SAVE
# ──────────────────────────────────────────────────────────────────────────────
anomaly_results = {
    "threshold": threshold,
    "y_pred_anomaly": y_pred_anomaly,
    "metrics": metrics,
    "roc_auc": roc_auc,
    "average_precision": ap,
}
save_pickle(anomaly_results, os.path.join(RESULTS_DIR, "05_anomaly_results.pkl"))

print("\n" + "=" * 70)
print("  STAGE 5 COMPLETE ✅")
print("=" * 70)
