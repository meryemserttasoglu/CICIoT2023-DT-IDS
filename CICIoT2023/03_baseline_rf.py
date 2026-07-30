"""
=============================================================================
STAGE 3 — BASELINE MODEL (Random Forest)
=============================================================================
Train a Random Forest classifier as the traditional-ML baseline:
  • Multi-class (attack category) classification
  • Binary (benign vs attack) classification
  • Full evaluation: Accuracy, Precision, Recall, F1
  • Confusion matrix visualisation
  • Feature importance ranking
  • Model persistence with joblib
=============================================================================
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_N_JOBS, RANDOM_STATE
)
from utils import (
    load_pickle, save_pickle, evaluate_classification,
    plot_confusion_matrix
)

print("\n" + "=" * 70)
print("  STAGE 3 — BASELINE MODEL (Random Forest)")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD PREPROCESSED DATA
# ──────────────────────────────────────────────────────────────────────────────
data = load_pickle(os.path.join(PROCESSED_DATA_DIR, "preprocessed.pkl"))

X_train       = data["X_train"]
X_test        = data["X_test"]
y_train       = data["y_train_category"]     # 8-class category
y_test        = data["y_test_category"]
y_train_bin   = data["y_train_binary"]
y_test_bin    = data["y_test_binary"]
le_category   = data["le_category"]
feature_cols  = data["feature_cols"]

category_names = le_category.classes_
print(f"[ℹ] Training samples : {X_train.shape[0]:,}")
print(f"[ℹ] Test samples     : {X_test.shape[0]:,}")
print(f"[ℹ] Categories       : {list(category_names)}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  TRAIN — MULTI-CLASS (category)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[⏳] Training Random Forest (multi-class) ...")
t0 = time.time()

rf_multi = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    n_jobs=RF_N_JOBS,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    verbose=0,
)
rf_multi.fit(X_train, y_train)
train_time = time.time() - t0
print(f"[✓] Training completed in {train_time:.1f}s")

# Predict
y_pred_multi = rf_multi.predict(X_test)
metrics_multi = evaluate_classification(
    y_test, y_pred_multi,
    model_name="Random Forest (Multi-Class)",
    labels=np.arange(len(category_names)),
    target_names=list(category_names),
    average="weighted"
)
metrics_multi["train_time_s"] = train_time

# Compute multi-class ROC-AUC (M2 fix)
try:
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    y_pred_probs_rf = rf_multi.predict_proba(X_test)
    y_test_bin_mc = label_binarize(y_test, classes=np.arange(len(category_names)))
    roc_auc_rf = roc_auc_score(
        y_test_bin_mc, y_pred_probs_rf, multi_class="ovr", average="weighted"
    )
    metrics_multi["roc_auc"] = roc_auc_rf
    print(f"[✓] Multi-class ROC-AUC (weighted OVR): {roc_auc_rf:.4f}")
except Exception as e:
    print(f"[⚠] Could not compute multi-class ROC-AUC: {e}")
    metrics_multi["roc_auc"] = None

# Confusion matrix (M4 fix: use text labels)
plot_confusion_matrix(
    y_test, y_pred_multi,
    labels=np.arange(len(category_names)),
    target_names=list(category_names),
    title="Random Forest — Category Confusion Matrix",
    save_path=os.path.join(FIGURES_DIR, "03_rf_category_cm.png")
)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  TRAIN — BINARY
# ──────────────────────────────────────────────────────────────────────────────
print("\n[⏳] Training Random Forest (binary) ...")
t0 = time.time()

rf_binary = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    n_jobs=RF_N_JOBS,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    verbose=0,
)
rf_binary.fit(X_train, y_train_bin)
train_time_bin = time.time() - t0
print(f"[✓] Training completed in {train_time_bin:.1f}s")

y_pred_bin = rf_binary.predict(X_test)
metrics_bin = evaluate_classification(
    y_test_bin, y_pred_bin,
    model_name="Random Forest (Binary)",
    target_names=["Benign", "Attack"],
    average="binary"
)
metrics_bin["train_time_s"] = train_time_bin

# ──────────────────────────────────────────────────────────────────────────────
# 4.  FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print("  FEATURE IMPORTANCE (Top 15)")
print(f"{'─'*50}")

importances = rf_multi.feature_importances_
indices = np.argsort(importances)[::-1]

for rank, idx in enumerate(indices[:15], 1):
    print(f"  {rank:2d}. {feature_cols[idx]:<30s}  {importances[idx]:.4f}")

# Plot feature importance
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
top_n = 15
top_idx = indices[:top_n]
ax.barh(range(top_n), importances[top_idx][::-1], color="#3498db", edgecolor="black",
        linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_cols[i] for i in top_idx][::-1])
ax.set_xlabel("Feature Importance")
ax.set_title("Random Forest — Top 15 Feature Importances", fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "03_rf_feature_importance.png"))
print(f"[✓] Figure saved → {os.path.join(FIGURES_DIR, '03_rf_feature_importance.png')}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  SAVE MODELS & METRICS
# ──────────────────────────────────────────────────────────────────────────────
joblib.dump(rf_multi, os.path.join(MODELS_DIR, "rf_multiclass.joblib"))
joblib.dump(rf_binary, os.path.join(MODELS_DIR, "rf_binary.joblib"))
print(f"[✓] Models saved → {MODELS_DIR}")

results = {
    "rf_multi": metrics_multi,
    "rf_binary": metrics_bin,
    "feature_importance": dict(zip(
        [feature_cols[i] for i in indices],
        importances[indices]
    )),
}
save_pickle(results, os.path.join(RESULTS_DIR, "03_rf_results.pkl"))

print("\n" + "=" * 70)
print("  STAGE 3 COMPLETE ✅")
print("=" * 70)
