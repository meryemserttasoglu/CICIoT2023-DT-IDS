"""
=============================================================================
CICIoT2023 — Configuration
=============================================================================
Central configuration for the Hybrid Digital-Twin IDS Pipeline.
All hyper-parameters, paths, and constants live here so that every stage
script can `from config import CFG` and stay in sync.
=============================================================================
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create directories if they don't exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_CSV = "/tmp/ciciot10"
LABEL_COLUMN = "label"
SAMPLE_FRACTION = None         # ilk 10 dosya tam (2.366.956 satir)
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Binary label mapping
BENIGN_LABEL = "BenignTraffic"

# Attack category mapping (34 attack types → 8 categories)
ATTACK_CATEGORY_MAP = {
    "BenignTraffic": "Benign",
    # DDoS
    "DDoS-RSTFINFlood": "DDoS", "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS", "DDoS-ICMP_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    # DoS
    "DoS-UDP_Flood": "DoS", "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS", "DoS-HTTP_Flood": "DoS",
    # Mirai
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    # Recon
    "Recon-PingSweep": "Recon", "Recon-OSScan": "Recon",
    "Recon-PortScan": "Recon", "Recon-HostDiscovery": "Recon",
    # Spoofing
    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    # Web
    "BrowserHijacking": "Web", "Backdoor_Malware": "Web",
    "XSS": "Web", "Uploading_Attack": "Web",
    "SqlInjection": "Web", "CommandInjection": "Web",
    # Brute Force
    "DictionaryBruteForce": "BruteForce",
    # Catch-all for any variation
    "VulnerabilityScan": "Recon",
}

# ---------------------------------------------------------------------------
# Random Forest (Stage 3)
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
RF_N_JOBS = -1

# ---------------------------------------------------------------------------
# Autoencoder (Stage 4)
# ---------------------------------------------------------------------------
AE_ENCODING_DIM = 16          # bottleneck dimension
AE_HIDDEN_LAYERS = [64, 32]   # encoder layer sizes (decoder mirrors)
AE_EPOCHS = 30          # reality-check (was 2); early stopping active
AE_BATCH_SIZE = 256
AE_LEARNING_RATE = 1e-3
AE_PATIENCE = 7               # EarlyStopping patience
AE_VALIDATION_SPLIT = 0.15

# ---------------------------------------------------------------------------
# Anomaly Detection (Stage 5)
# ---------------------------------------------------------------------------
ANOMALY_THRESHOLD_PERCENTILE = 95   # percentile on benign reconstruction errors

# ---------------------------------------------------------------------------
# ConvLSTM (Stage 6)
# ---------------------------------------------------------------------------
CONVLSTM_WINDOW_SIZE = 10
CONVLSTM_FILTERS = 64
CONVLSTM_EPOCHS = 30        # reality-check (was 2)
CONVLSTM_BATCH_SIZE = 128
CONVLSTM_LEARNING_RATE = 1e-3
CONVLSTM_PATIENCE = 5

# ---------------------------------------------------------------------------
# Hybrid Model (Stage 8)
# ---------------------------------------------------------------------------
HYBRID_EPOCHS = 30          # reality-check (was 2)
HYBRID_BATCH_SIZE = 128
HYBRID_LEARNING_RATE = 5e-4
HYBRID_PATIENCE = 5

# ---------------------------------------------------------------------------
# SHAP / RCA (Stage 7)
# ---------------------------------------------------------------------------
SHAP_SAMPLE_SIZE = 500         # samples for SHAP explanation (speed)
RCA_TOP_K = 5                  # top-k root-cause features

# ---------------------------------------------------------------------------
# Dashboard (Stage 10)
# ---------------------------------------------------------------------------
DASHBOARD_REFRESH_SECONDS = 2
