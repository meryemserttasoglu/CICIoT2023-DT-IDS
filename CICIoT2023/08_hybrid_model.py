"""
=============================================================================
STAGE 8 — HYBRID MODEL 🔥 (MAIN CONTRIBUTION)
=============================================================================
Late-fusion architecture combining:
  • Autoencoder anomaly scores  (reconstruction error vector)
  • ConvLSTM temporal predictions  (class probability vector)
Into a unified decision layer for superior intrusion detection.
=============================================================================
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix macOS TensorFlow threading issue
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow import keras
from tensorflow.keras import layers, callbacks, Model

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR,
    HYBRID_EPOCHS, HYBRID_BATCH_SIZE, HYBRID_LEARNING_RATE,
    HYBRID_PATIENCE, RANDOM_STATE, AE_BATCH_SIZE
)
from utils import (
    load_pickle, save_pickle, evaluate_classification,
    evaluate_binary, plot_training_history, plot_confusion_matrix
)

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print("\n" + "=" * 70)
print("  STAGE 8 — HYBRID MODEL 🔥 (MAIN CONTRIBUTION)")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD PRETRAINED MODELS & DATA
# ──────────────────────────────────────────────────────────────────────────────
data = load_pickle(os.path.join(PROCESSED_DATA_DIR, "preprocessed.pkl"))
convlstm_results = load_pickle(os.path.join(RESULTS_DIR, "06_convlstm_results.pkl"))

X_test       = data["X_test"]
y_test_cat   = data["y_test_category"]
y_test_bin   = data["y_test_binary"]
le_category  = data["le_category"]
feature_cols = data["feature_cols"]

n_classes = len(le_category.classes_)
n_features = len(feature_cols)

# Load Autoencoder
autoencoder = keras.models.load_model(os.path.join(MODELS_DIR, "autoencoder.h5"))
print("[✓] Autoencoder loaded")

# Load ConvLSTM
convlstm_model = keras.models.load_model(os.path.join(MODELS_DIR, "convlstm.h5"))
print("[✓] ConvLSTM loaded")

# Get sequence data from ConvLSTM stage
X_train_seq     = convlstm_results["X_train_seq"]
y_train_seq     = convlstm_results["y_train_seq"]
y_train_seq_bin = convlstm_results["y_train_seq_bin"]
X_test_seq      = convlstm_results["X_test_seq"]
y_test_seq      = convlstm_results["y_test_seq"]
y_test_seq_bin  = convlstm_results["y_test_seq_bin"]

print(f"[ℹ] Train sequences: {X_train_seq.shape}")
print(f"[ℹ] Test sequences : {X_test_seq.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  GENERATE FEATURES FROM BOTH MODELS
# ──────────────────────────────────────────────────────────────────────────────
print("\n[⏳] Generating Autoencoder anomaly features ...")

# For each sequence, compute AE reconstruction error on the last timestep
def get_ae_features(X_seq, autoencoder, batch_size=256):
    """
    Extract anomaly score vector from the last timestep of each sequence.
    Returns: (n_samples, n_features) — per-feature reconstruction error
    """
    # Take last timestep
    X_last = X_seq[:, -1, :]  # (n_samples, n_features)
    X_reconstructed = autoencoder.predict(X_last, batch_size=batch_size, verbose=0)
    errors = (X_last - X_reconstructed) ** 2  # per-feature MSE
    return errors


ae_train_features = get_ae_features(X_train_seq, autoencoder, AE_BATCH_SIZE)
ae_test_features = get_ae_features(X_test_seq, autoencoder, AE_BATCH_SIZE)
print(f"[✓] AE features: train {ae_train_features.shape}, test {ae_test_features.shape}")

print("[⏳] Generating ConvLSTM temporal features ...")
convlstm_train_probs = convlstm_model.predict(X_train_seq, batch_size=128, verbose=0)
convlstm_test_probs = convlstm_model.predict(X_test_seq, batch_size=128, verbose=0)
print(f"[✓] ConvLSTM probs: train {convlstm_train_probs.shape}, "
      f"test {convlstm_test_probs.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  CONCATENATE → FUSION INPUT
# ──────────────────────────────────────────────────────────────────────────────
# Also add global reconstruction error as a scalar feature
ae_train_global = ae_train_features.mean(axis=1, keepdims=True)
ae_test_global = ae_test_features.mean(axis=1, keepdims=True)

X_hybrid_train = np.concatenate([
    ae_train_features,       # (N, n_features) — per-feature AE errors
    ae_train_global,         # (N, 1) — global AE score
    convlstm_train_probs,    # (N, n_classes) — ConvLSTM class probs
], axis=1)

X_hybrid_test = np.concatenate([
    ae_test_features,
    ae_test_global,
    convlstm_test_probs,
], axis=1)

hybrid_dim = X_hybrid_train.shape[1]
print(f"[✓] Hybrid feature dimension: {hybrid_dim} "
      f"({n_features} AE + 1 global + {n_classes} ConvLSTM)")

# Labels (one-hot for multi-class)
y_train_onehot = keras.utils.to_categorical(y_train_seq, n_classes)
y_test_onehot = keras.utils.to_categorical(y_test_seq, n_classes)

# Compute class weights for imbalanced data (M1 fix)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    "balanced", classes=np.arange(n_classes), y=y_train_seq
)
class_weight_dict = dict(enumerate(class_weights))
print(f"[ℹ] Class weights computed for {n_classes} classes (balanced)")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  BUILD FUSION MODEL
# ──────────────────────────────────────────────────────────────────────────────
def build_fusion_model(input_dim, n_classes):
    """
    Fusion MLP that combines AE anomaly scores + ConvLSTM predictions.
    Sub-models are frozen; only the fusion layers are trained.
    """
    inputs = keras.Input(shape=(input_dim,), name="hybrid_input")

    x = layers.Dense(128, activation="relu", name="fusion_dense_1")(inputs)
    x = layers.BatchNormalization(name="fusion_bn_1")(x)
    x = layers.Dropout(0.3, name="fusion_drop_1")(x)

    x = layers.Dense(64, activation="relu", name="fusion_dense_2")(x)
    x = layers.BatchNormalization(name="fusion_bn_2")(x)
    x = layers.Dropout(0.3, name="fusion_drop_2")(x)

    x = layers.Dense(32, activation="relu", name="fusion_dense_3")(x)
    x = layers.Dropout(0.2, name="fusion_drop_3")(x)

    outputs = layers.Dense(n_classes, activation="softmax",
                           name="fusion_output")(x)

    model = Model(inputs, outputs, name="Hybrid_Fusion")
    return model


hybrid_model = build_fusion_model(hybrid_dim, n_classes)

hybrid_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=HYBRID_LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n[ℹ] Hybrid Fusion Architecture:")
hybrid_model.summary()

# ──────────────────────────────────────────────────────────────────────────────
# 5.  TRAIN FUSION
# ──────────────────────────────────────────────────────────────────────────────
cb = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=HYBRID_PATIENCE,
        restore_best_weights=True, verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1,
    ),
]

print("\n[⏳] Training Hybrid Fusion Model ...")
t0 = time.time()
history = hybrid_model.fit(
    X_hybrid_train, y_train_onehot,
    validation_split=0.15,
    epochs=HYBRID_EPOCHS,
    batch_size=HYBRID_BATCH_SIZE,
    callbacks=cb,
    class_weight=class_weight_dict,
    verbose=1,
)
train_time = time.time() - t0
print(f"[✓] Training completed in {train_time:.1f}s")

# Training curves
plot_training_history(
    history, metrics=("loss", "accuracy"),
    title="Hybrid Fusion — Training History",
    save_path=os.path.join(FIGURES_DIR, "08_hybrid_training.png"),
)

# ──────────────────────────────────────────────────────────────────────────────
# 6.  EVALUATE
# ──────────────────────────────────────────────────────────────────────────────
print("\n[⏳] Evaluating Hybrid Model ...")
y_pred_probs = hybrid_model.predict(X_hybrid_test, batch_size=HYBRID_BATCH_SIZE, verbose=0)
y_pred_cat = np.argmax(y_pred_probs, axis=1)

metrics_multi = evaluate_classification(
    y_test_seq, y_pred_cat,
    model_name="Hybrid Model (Category)",
    labels=np.arange(n_classes),
    target_names=list(le_category.classes_),
    average="weighted",
)
metrics_multi["train_time_s"] = train_time

# Compute multi-class ROC-AUC (M2 fix)
try:
    from sklearn.metrics import roc_auc_score
    roc_auc_hybrid = roc_auc_score(
        y_test_onehot, y_pred_probs, multi_class="ovr", average="weighted"
    )
    metrics_multi["roc_auc"] = roc_auc_hybrid
    print(f"[✓] Multi-class ROC-AUC (weighted OVR): {roc_auc_hybrid:.4f}")
except Exception as e:
    print(f"[⚠] Could not compute multi-class ROC-AUC: {e}")
    metrics_multi["roc_auc"] = None

# Binary evaluation (M5 fix: verify alignment)
assert len(y_test_seq_bin) == len(y_pred_cat), \
    f"Binary label length mismatch: {len(y_test_seq_bin)} vs {len(y_pred_cat)}"

# Determine Benign class index safely
try:
    benign_idx = le_category.transform(["Benign"])[0]
except ValueError:
    benign_idx = le_category.transform(["BenignTraffic"])[0]

y_pred_bin_hybrid = (y_pred_cat != benign_idx).astype(int)
metrics_bin = evaluate_binary(
    y_test_seq_bin, y_pred_bin_hybrid,
    model_name="Hybrid Model (Binary)",
)

# Confusion matrix (M4 fix: use text labels)
plot_confusion_matrix(
    y_test_seq, y_pred_cat,
    labels=np.arange(n_classes),
    target_names=list(le_category.classes_),
    title="Hybrid Model — Category Confusion Matrix",
    save_path=os.path.join(FIGURES_DIR, "08_hybrid_cm.png"),
)

# ──────────────────────────────────────────────────────────────────────────────
# 7.  SAVE
# ──────────────────────────────────────────────────────────────────────────────
hybrid_model.save(os.path.join(MODELS_DIR, "hybrid_model.h5"))
print(f"[✓] Hybrid model saved → {MODELS_DIR}/hybrid_model.h5")

hybrid_results = {
    "metrics_multi": metrics_multi,
    "metrics_binary": metrics_bin,
    "y_pred_probs": y_pred_probs,
    "y_pred_cat": y_pred_cat,
    "train_time_s": train_time,
    "history": history.history,
    "hybrid_dim": hybrid_dim,
}
save_pickle(hybrid_results, os.path.join(RESULTS_DIR, "08_hybrid_results.pkl"))

print("\n" + "=" * 70)
print("  STAGE 8 COMPLETE ✅")
print("=" * 70)
