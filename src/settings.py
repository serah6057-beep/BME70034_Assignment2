"""
src/settings.py
---------------
Central configuration loader for the GKX (2020) replication project.

Responsibilities:
  - Define all file paths (data, results, logs)
  - Set hyperparameters for each model
  - Set sample parameters (date ranges, characteristic list)
  - Initialize the project-wide logger
  
All other modules import from here; changing a value here propagates
throughout the entire pipeline (extensibility requirement).
"""

import os
import logging
from pathlib import Path

# =============================================================
# ROOT PATHS
# =============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # assignment3/
DATA_DIR     = PROJECT_ROOT / "data"
SRC_DIR      = PROJECT_ROOT / "src"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
TABLES_DIR   = RESULTS_DIR / "tables"
LOGS_DIR     = RESULTS_DIR / "logs"

# Ensure output directories exist at import time
for _dir in [DATA_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# =============================================================
# DATA FILE PATHS
# NOTE: Raw CRSP/Compustat data are stored locally only and
#       listed in .gitignore — never pushed to GitHub.
# =============================================================
CRSP_FILE          = DATA_DIR / "crsp_monthly.parquet"   # CRSP monthly stock returns
CHARACTERISTICS_FILE = DATA_DIR / "characteristics.parquet"  # Xiu's stock characteristics
MACRO_FILE         = DATA_DIR / "macro_predictors.parquet"   # Welch-Goyal macro vars
RF_FILE            = DATA_DIR / "rf_monthly.parquet"         # Risk-free rate (FF)
SP500_FILE         = DATA_DIR / "sp500_returns.parquet"      # S&P 500 total return index

# =============================================================
# SAMPLE PARAMETERS
# =============================================================
# Two separate sample periods required by the assignment
SAMPLE_PERIODS = {
    "1971_2025": {"start": "1971-01-01", "end":   "2025-12-31"},
    "2006_2025": {"start": "2006-01-01", "end":   "2025-12-31"},
}

# Rolling window splits (GKX 2020 convention)
TRAIN_YEARS      = 5   # Minimum initial training window (years)
VALIDATION_YEARS = 2  # Rolling validation window (years) — for hyperparameter tuning
TEST_START_YEAR  = 2013 # Out-of-sample evaluation starts here (after 18yr train + early burn-in)

# Minimum number of stocks per month (filter months with thin coverage)
MIN_STOCKS_PER_MONTH = 5

# Minimum non-missing observations per stock (in training window) before inclusion
MIN_OBS_PER_STOCK = 12  # months

# =============================================================
# STOCK CHARACTERISTICS (94 features from Xiu's dataset)
# These are the 94 predictors used in GKX 2020.
# A curated subset is used if the full set is unavailable.
# =============================================================
# Full list — the pipeline automatically uses available columns.
CHARACTERISTIC_COLUMNS = [
    # Momentum / price signals
    "mom1m", "mom6m", "mom12m", "mom36m", "chmom",
    "indmom", "maxret", "turn", "std_turn",
    # Value signals
    "bm", "ep", "cfp", "dp", "sp",
    # Size
    "mvel1", "dolvol",
    # Profitability
    "gp", "roe", "roa", "roaq", "roavol",
    "sgr", "chcsho", "hire", "lgr",
    # Investment
    "invest", "absacc", "acc", "aeavol",
    "age", "agr", "baspread",
    "beta", "betasq", "cash",
    "cashdebt", "cashpr", "cfvol",
    "chatoia", "chato", "chinv",
    "chnanalyst", "chpmia", "chtx",
    "cinvest", "convind", "currat",
    "depr", "divi", "divo",
    "dy", "ear", "egr",
    "fgr5yr", "grltnoa", "grsaleq",
    "herf", "hire", "idiovol",
    "ill", "indmom", "invest",
    "lev", "liq_ann", "liq_qtr",
    "ms", "mve_ia", "nanalyst",
    "nincr", "operprof", "orgcap",
    "pchcapx_ia", "pchcurrat", "pchdepr",
    "pchgm_pchsale", "pchquick", "pchsale_pchinvt",
    "pchsale_pchrect", "pchsale_pchxsga", "pchsaleinv",
    "pctacc", "ps", "quick",
    "rd", "rd_mve", "rd_sale",
    "realestate", "recdts", "retvol",
    "rid", "rsst", "salecash",
    "saleinv", "salerec", "secured",
    "securedind", "sgr", "sin",
    "sp", "std_dolvol", "stdacc",
    "stdcf", "sue", "tang",
    "tb", "trans", "turn",
    "zerotrade",
]

# Remove duplicates while preserving order
CHARACTERISTIC_COLUMNS = list(dict.fromkeys(CHARACTERISTIC_COLUMNS))

# =============================================================
# MACRO PREDICTORS (Welch-Goyal 2008; used in GKX 2020)
# Exactly 8 variables as specified in GKX (2020), p. 2248
# =============================================================
MACRO_COLUMNS = [
    "dp",        # Dividend-price ratio (log)
    "ep",        # Earnings-price ratio (log)
    "bm",        # Book-to-market ratio
    "ntis",      # Net equity expansion
    "tbl",       # T-bill rate
    "tms",       # Term spread (lty - tbl)
    "dfy",       # Default yield spread (baa - aaa)
    "svar",      # Stock variance
]

# =============================================================
# MODEL HYPERPARAMETERS
# All values can be edited here without touching model code.
# =============================================================

# --- OLS (Huber loss variant = "+H") ---
OLS_PARAMS = {
    "huber_epsilon": 1.345,   # Huber loss robustness parameter
}

# --- OLS-3 (3-factor, i.e., only Size/BM/Momentum as features) ---
OLS3_FEATURES = ["mvel1", "bm", "mom12m"]   # Size, B/M, 12m momentum

# --- Principal Component Regression (PCR) ---
PCR_PARAMS = {
    "n_components_grid": [10, 30],  # Grid search candidates
}

# --- Elastic Net (ENet+H: Huber loss + elastic net penalty) ---
ENET_PARAMS = {
    "l1_ratio_grid":  [0.5],  # 1.0 = pure Lasso
    "alpha_grid":     [0.0001, 0.01],
    "huber_epsilon":  1.345,
}

# --- Random Forest (RF) ---
RF_PARAMS = {
    "n_estimators":           300,                       # paper fixes at 300
    "max_features_grid":      [3, 5, 10, 20, 30, 50],   # paper grid
    "min_samples_leaf_grid":  [5000, 10000],            # paper grid
    "n_jobs":                 -1,
    "random_state":           42,
}

# --- Neural Networks (NN2 = 2 hidden layers, NN4 = 4 hidden layers) ---
NN_PARAMS = {
    "NN2": {
        "hidden_layers": [32, 16],
        "dropout":       0.05,
        "batch_size":    20000,
        "lr":            0.001,
        "epochs":        100,
        "patience":      3,      # Early stopping patience
    },
    "NN4": {
        "hidden_layers": [32, 16, 8, 4],
        "dropout":       0.05,
        "batch_size":    10000,
        "lr":            0.001,
        "epochs":        100,
        "patience":      5,
    },
}

# Ensemble: batch normalization is applied in each NN hidden layer
NN_BATCH_NORM   = True
NN_ENSEMBLE_N   = 10    # GKX 2020 average 10 random restarts
NN_RANDOM_SEED  = 42

# =============================================================
# PORTFOLIO CONSTRUCTION
# =============================================================
PORT_DECILES    = 10     # Decile portfolios (GKX 2020 Table 1 + Figure 9)
PORT_TOP_N      = None   # None = use full "All" sample (skip Top/Bottom 1000)
LONG_DECILE     = 10     # Long top decile
SHORT_DECILE    = 1      # Short bottom decile

# Value-weight or equal-weight portfolios
WEIGHTING       = "value"   # "value" or "equal"

# =============================================================
# OUTPUT SETTINGS
# =============================================================
DPI             = 150        # Figure resolution
FIG_FORMAT      = "pdf"      # Save figures as PDF (also saves PNG for quick view)
TABLE_FORMAT    = "latex"    # Also supports "csv"

# Which algorithms to include per output (matches assignment spec)
MODELS_TABLE1  = ["OLS+H", "OLS-3+H", "PCR", "ENet+H", "RF", "NN2", "NN4"]
MODELS_FIG4    = ["PCR", "ENet+H", "RF", "NN2", "NN4"]
MODELS_FIG9    = ["OLS-3+H", "PCR", "ENet+H", "RF", "NN2", "NN4"]

# =============================================================
# LOGGING SETUP
# =============================================================
LOG_FILE = LOGS_DIR / "run.log"

def get_logger(name: str = "gkx_replication") -> logging.Logger:
    """
    Returns a module-level logger that writes to both the console
    and to results/logs/run.log.
    
    Usage:
        from src.settings import get_logger
        logger = get_logger(__name__)
        logger.info("Starting pipeline...")
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # Avoid duplicate handlers on re-import
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — DEBUG and above (full execution log)
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
