"""
=============================================================================
STAGE 8B — HYBRID MODEL (AE + RANDOM FOREST FUSION)  [HONEST VARIANT]
=============================================================================
Reality-check variant: the ConvLSTM temporal path collapses to ~60% on
CICIoT2023 (no timestamp -> windowing over shuffled data is meaningless).
This variant replaces the ConvLSTM probability vector in the late-fusion
with Random Forest class probabilities, keeping the Autoencoder Digital-Twin
anomaly signal. Operates on the PLAIN (non-windowed) feature matrix.

Fusion input = [per-feature AE reconstruction error | global AE error | RF class probs]
=============================================================================
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import joblib
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
    evaluate_binary, plot_confusion_matrix
)

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print("\n" + "=" * 70)
print("  STAGE 8B — HYBRID (AE + RANDOM FOREST FUSION)")
print("=" * 70)

# 1. LOAD DATA & MODELS -------------------------------------------------------
data = load_pickle(os.path.join(PROCESSED_DATA_DIR, "preprocessed.pkl"))
X_train     = data["X_train"]
X_test      = data["X_test"]
y_train_cat = data["y_train_category"]
y_test_cat  = data["y_test_category"]
y_train_bin = data["y_train_binary"]
y_test_bin  = data["y_test_binary"]
le_category = data["le_category"]
n_classes   = len(le_category.classes_)
n_features  = X_train.shape[1]
print(f"[i] Classes={n_classes}  Features={n_features}  Train={X_train.shape[0]:,}  Test={X_test.shape[0]:,}")

autoencoder = keras.models.load_model(os.path.join(MODELS_DIR, "autoencoder.h5"))
print("[ok] Autoencoder loaded")
rf = joblib.load(os.path.join(MODELS_DIR, "rf_multiclass.joblib"))
print(f"[ok] RF loaded (classes_={getattr(rf,'classes_',None)})")

# 2. FEATURE GENERATION -------------------------------------------------------
def ae_err(X):
    recon = autoencoder.predict(X, batch_size=AE_BATCH_SIZE, verbose=0)
    return (X - recon) ** 2

print("[..] AE reconstruction errors ...")
ae_tr = ae_err(X_train); ae_te = ae_err(X_test)
print("[..] RF class probabilities ...")
rf_tr = rf.predict_proba(X_train)
rf_te = rf.predict_proba(X_test)

# Align RF proba columns to 0..n_classes-1 (RF.classes_ may be a subset/ordered)
def align(probs, classes):
    full = np.zeros((probs.shape[0], n_classes), dtype=np.float32)
    for j, c in enumerate(classes):
        full[:, int(c)] = probs[:, j]
    return full
rf_tr = align(rf_tr, rf.classes_); rf_te = align(rf_te, rf.classes_)

X_h_train = np.concatenate([ae_tr, ae_tr.mean(1, keepdims=True), rf_tr], axis=1)
X_h_test  = np.concatenate([ae_te, ae_te.mean(1, keepdims=True), rf_te], axis=1)
hybrid_dim = X_h_train.shape[1]
print(f"[ok] Hybrid dim = {hybrid_dim} ({n_features} AE + 1 global + {n_classes} RF)")

y_tr_oh = keras.utils.to_categorical(y_train_cat, n_classes)
y_te_oh = keras.utils.to_categorical(y_test_cat, n_classes)
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train_cat)
cw_dict = dict(enumerate(cw))

# 3. FUSION MLP ---------------------------------------------------------------
def build(input_dim, n_classes):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x); x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="Hybrid_AE_RF")

model = build(hybrid_dim, n_classes)
model.compile(optimizer=keras.optimizers.Adam(HYBRID_LEARNING_RATE),
              loss="categorical_crossentropy", metrics=["accuracy"])

cb = [callbacks.EarlyStopping(monitor="val_loss", patience=HYBRID_PATIENCE,
                             restore_best_weights=True, verbose=1),
      callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0)]

print("\n[..] Training fusion ...")
t0 = time.time()
model.fit(X_h_train, y_tr_oh, validation_split=0.15, epochs=HYBRID_EPOCHS,
          batch_size=HYBRID_BATCH_SIZE, callbacks=cb, class_weight=cw_dict, verbose=2)
train_time = time.time() - t0
print(f"[ok] Trained in {train_time:.1f}s")

# 4. EVALUATE -----------------------------------------------------------------
y_pred_probs = model.predict(X_h_test, batch_size=HYBRID_BATCH_SIZE, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

m_multi = evaluate_classification(
    y_test_cat, y_pred, model_name="Hybrid AE+RF (Category)",
    labels=np.arange(n_classes), target_names=list(le_category.classes_),
    average="macro")
m_multi["train_time_s"] = train_time

# weighted too
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
m_multi["f1_weighted"] = f1_score(y_test_cat, y_pred, average="weighted")
m_multi["accuracy"] = accuracy_score(y_test_cat, y_pred)

try:
    from sklearn.metrics import roc_auc_score
    m_multi["roc_auc"] = roc_auc_score(y_te_oh, y_pred_probs, multi_class="ovr", average="weighted")
except Exception as e:
    m_multi["roc_auc"] = None

try:
    benign_idx = le_category.transform(["Benign"])[0]
except ValueError:
    benign_idx = le_category.transform(["BenignTraffic"])[0]
y_pred_bin = (y_pred != benign_idx).astype(int)
m_bin = evaluate_binary(y_test_bin, y_pred_bin, model_name="Hybrid AE+RF (Binary)")

plot_confusion_matrix(
    y_test_cat, y_pred, labels=np.arange(n_classes),
    target_names=list(le_category.classes_),
    title="Hybrid AE+RF — Category Confusion Matrix",
    save_path=os.path.join(FIGURES_DIR, "08b_hybrid_rf_cm.png"))

save_pickle({"metrics_multi": m_multi, "metrics_binary": m_bin,
             "y_pred": y_pred, "y_pred_probs": y_pred_probs, "hybrid_dim": hybrid_dim},
            os.path.join(RESULTS_DIR, "08b_hybrid_rf_results.pkl"))

print("\n>>> SONUC (AE+RF Hibrit):")
print("   multi  acc=%.4f  macroF1=%.4f  weightedF1=%.4f  rocauc=%s" % (
    m_multi["accuracy"], m_multi.get("f1"), m_multi.get("f1_weighted"), str(m_multi.get("roc_auc"))[:6]))
print("   binary acc=%.4f  f1=%.4f" % (m_bin["accuracy"], m_bin["f1"]))
print("=" * 70)
