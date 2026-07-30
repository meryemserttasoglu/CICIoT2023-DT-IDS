"""
=============================================================================
STAGE 6 — ConvLSTM (TEMPORAL MODEL)
=============================================================================
Convert tabular CICIoT2023 data into time-series sequences and build a
Conv1D-LSTM hybrid model for temporal pattern detection:
  • Sliding window transform
  • Conv1D + LSTM architecture (more practical than ConvLSTM2D for tabular)
  • Multi-class classification (attack categories)
  • Full evaluation with classification report
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

# Force CPU to prevent macOS Metal mutex locks
tf.config.set_visible_devices([], 'GPU')

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow import keras
from tensorflow.keras import layers, callbacks, Model

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR,
    CONVLSTM_WINDOW_SIZE, CONVLSTM_FILTERS, CONVLSTM_EPOCHS,
    CONVLSTM_BATCH_SIZE, CONVLSTM_LEARNING_RATE, CONVLSTM_PATIENCE,
    RANDOM_STATE
)
from utils import (
    load_pickle, save_pickle, evaluate_classification,
    plot_training_history, plot_confusion_matrix
)

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print("\n" + "=" * 70)
print("  STAGE 6 — ConvLSTM (TEMPORAL MODEL)")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
data = load_pickle(os.path.join(PROCESSED_DATA_DIR, "preprocessed.pkl"))

X_train      = data["X_train"]
X_test       = data["X_test"]
y_train_cat  = data["y_train_category"]
y_test_cat   = data["y_test_category"]
y_train_bin  = data["y_train_binary"]
y_test_bin   = data["y_test_binary"]
le_category  = data["le_category"]
feature_cols = data["feature_cols"]

n_features = X_train.shape[1]
n_classes = len(le_category.classes_)
print(f"[ℹ] Features   : {n_features}")
print(f"[ℹ] Classes     : {n_classes}  ({list(le_category.classes_)})")
print(f"[ℹ] Window size : {CONVLSTM_WINDOW_SIZE}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  SLIDING WINDOW TRANSFORM
# ──────────────────────────────────────────────────────────────────────────────
# NOTE (C3 fix): The train/test split in Stage 2 is stratified-random,
# NOT time-ordered. This means sliding-window sequences are constructed
# from randomly shuffled samples rather than temporally consecutive flows.
# For production use with the real CICIoT2023 dataset, consider sorting
# by timestamp before splitting. For this synthetic evaluation, the
# Conv1D-LSTM learns local feature dependencies within each window.
def create_sequences(X, y, window_size):
    """
    Convert (N, features) → (N - window + 1, window, features)
    using a sliding window. Labels use the last timestep.
    """
    n_samples = X.shape[0] - window_size + 1
    X_seq = np.zeros((n_samples, window_size, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n_samples, dtype=y.dtype)

    for i in range(n_samples):
        X_seq[i] = X[i : i + window_size]
        y_seq[i] = y[i + window_size - 1]  # label of last timestep

    return X_seq, y_seq


print("\n[⏳] Creating sliding-window sequences ...")
t0 = time.time()

X_train_seq, y_train_seq = create_sequences(X_train, y_train_cat,
                                             CONVLSTM_WINDOW_SIZE)
X_test_seq, y_test_seq = create_sequences(X_test, y_test_cat,
                                           CONVLSTM_WINDOW_SIZE)

# Also create binary sequences (for Stage 8)
_, y_train_seq_bin = create_sequences(X_train, y_train_bin,
                                       CONVLSTM_WINDOW_SIZE)
_, y_test_seq_bin = create_sequences(X_test, y_test_bin,
                                      CONVLSTM_WINDOW_SIZE)

seq_time = time.time() - t0
print(f"[✓] Sequences created in {seq_time:.1f}s")
print(f"    Train sequences : {X_train_seq.shape}")
print(f"    Test sequences  : {X_test_seq.shape}")

# Convert labels to one-hot for categorical crossentropy
y_train_onehot = keras.utils.to_categorical(y_train_seq, n_classes)
y_test_onehot = keras.utils.to_categorical(y_test_seq, n_classes)

# Compute class weights to handle imbalanced CICIoT2023 distribution (M1 fix)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    "balanced", classes=np.arange(n_classes), y=y_train_seq
)
class_weight_dict = dict(enumerate(class_weights))
print(f"[ℹ] Class weights computed for {n_classes} classes (balanced)")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  BUILD Conv1D-LSTM MODEL
# ──────────────────────────────────────────────────────────────────────────────
def build_conv1d_lstm(window_size, n_features, n_classes, filters):
    """
    Hybrid Conv1D + LSTM:
      Conv1D extracts local patterns within each timestep window,
      LSTM captures temporal dependencies across timesteps.
    """
    inputs = keras.Input(shape=(window_size, n_features), name="seq_input")

    # Conv1D block
    x = layers.Conv1D(filters, kernel_size=3, activation="relu",
                      padding="same", name="conv1d_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Conv1D(filters // 2, kernel_size=3, activation="relu",
                      padding="same", name="conv1d_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)

    # LSTM block
    x = layers.LSTM(64, return_sequences=True, name="lstm_1")(x)
    x = layers.Dropout(0.3, name="drop_1")(x)
    x = layers.LSTM(32, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(0.3, name="drop_2")(x)

    # Classification head
    x = layers.Dense(64, activation="relu", name="fc_1")(x)
    x = layers.Dropout(0.2, name="drop_3")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="Conv1D_LSTM")
    return model


model = build_conv1d_lstm(
    window_size=CONVLSTM_WINDOW_SIZE,
    n_features=n_features,
    n_classes=n_classes,
    filters=CONVLSTM_FILTERS,
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONVLSTM_LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n[ℹ] Conv1D-LSTM Architecture:")
model.summary()

# ──────────────────────────────────────────────────────────────────────────────
# 4.  TRAIN
# ──────────────────────────────────────────────────────────────────────────────
cb = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=CONVLSTM_PATIENCE,
        restore_best_weights=True, verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1,
    ),
]

print("\n[⏳] Training Conv1D-LSTM ...")
t0 = time.time()
history = model.fit(
    X_train_seq, y_train_onehot,
    validation_split=0.15,
    epochs=CONVLSTM_EPOCHS,
    batch_size=CONVLSTM_BATCH_SIZE,
    callbacks=cb,
    class_weight=class_weight_dict,
    verbose=1,
)
train_time = time.time() - t0
print(f"[✓] Training completed in {train_time:.1f}s")

# Training curves
plot_training_history(
    history, metrics=("loss", "accuracy"),
    title="Conv1D-LSTM Training History",
    save_path=os.path.join(FIGURES_DIR, "06_convlstm_training.png"),
)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  EVALUATE
# ──────────────────────────────────────────────────────────────────────────────
print("\n[⏳] Evaluating ...")
y_pred_probs = model.predict(X_test_seq, batch_size=CONVLSTM_BATCH_SIZE, verbose=0)
y_pred_cat = np.argmax(y_pred_probs, axis=1)

metrics = evaluate_classification(
    y_test_seq, y_pred_cat,
    model_name="Conv1D-LSTM (Category)",
    labels=np.arange(n_classes),
    target_names=list(le_category.classes_),
    average="weighted",
)
metrics["train_time_s"] = train_time

# Compute multi-class ROC-AUC (M2 fix)
try:
    from sklearn.metrics import roc_auc_score
    roc_auc_multi = roc_auc_score(
        y_test_onehot, y_pred_probs, multi_class="ovr", average="weighted"
    )
    metrics["roc_auc"] = roc_auc_multi
    print(f"[✓] Multi-class ROC-AUC (weighted OVR): {roc_auc_multi:.4f}")
except Exception as e:
    print(f"[⚠] Could not compute multi-class ROC-AUC: {e}")
    metrics["roc_auc"] = None

# Confusion matrix (M4 fix: use text labels instead of numeric indices)
plot_confusion_matrix(
    y_test_seq, y_pred_cat,
    labels=np.arange(n_classes),
    target_names=list(le_category.classes_),
    title="Conv1D-LSTM — Category Confusion Matrix",
    save_path=os.path.join(FIGURES_DIR, "06_convlstm_cm.png"),
)

# ──────────────────────────────────────────────────────────────────────────────
# 6.  SAVE
# ──────────────────────────────────────────────────────────────────────────────
model.save(os.path.join(MODELS_DIR, "convlstm.h5"))
print(f"[✓] Model saved → {MODELS_DIR}/convlstm.h5")

convlstm_results = {
    "metrics": metrics,
    "y_pred_probs": y_pred_probs,
    "y_pred_cat": y_pred_cat,
    "y_test_seq": y_test_seq,
    "y_test_seq_bin": y_test_seq_bin,
    "X_test_seq": X_test_seq,        # needed for Stage 8
    "X_train_seq": X_train_seq,      # needed for Stage 8
    "y_train_seq": y_train_seq,      # needed for Stage 8
    "y_train_seq_bin": y_train_seq_bin,
    "train_time_s": train_time,
    "history": history.history,
}
save_pickle(convlstm_results, os.path.join(RESULTS_DIR, "06_convlstm_results.pkl"))

print("\n" + "=" * 70)
print("  STAGE 6 COMPLETE ✅")
print("=" * 70)
