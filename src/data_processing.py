"""
src/data_processing.py
-----------------------
Cleans and merges all raw data into the model-ready feature matrix
following Gu, Kelly, and Xiu (2020) Sections 2 and 3.

Pipeline:
  1. Load CRSP returns, characteristics, and macro predictors
  2. Apply GKX sample filters (NYSE, common stocks, etc.)
  3. Cross-sectionally rank and scale characteristics to [-1, 1]
  4. Construct the full feature matrix: 
       Z_it = characteristics × (1 + macro predictors)  [interaction terms]
  5. Produce rolling train/validation/test splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Generator

from src.settings import (
    CRSP_FILE, CHARACTERISTICS_FILE, MACRO_FILE, RF_FILE,
    CHARACTERISTIC_COLUMNS, MACRO_COLUMNS,
    TRAIN_YEARS, VALIDATION_YEARS, TEST_START_YEAR,
    MIN_STOCKS_PER_MONTH, MIN_OBS_PER_STOCK,
    get_logger,
)

logger = get_logger(__name__)


# =============================================================
# 1. LOAD RAW DATA
# =============================================================

def load_crsp() -> pd.DataFrame:
    """
    Loads the cached CRSP monthly file and applies basic quality filters.

    Returns:
        DataFrame with columns [permno, date, ret, me, ...]
    """
    logger.info("Loading CRSP data...")
    df = pd.read_parquet(CRSP_FILE)

    # Ensure date is month-end
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)

    # Drop rows with missing return or non-positive market equity
    df = df.dropna(subset=["ret"])
    df = df[df["me"] > 0]

    logger.info(f"CRSP loaded: {len(df):,} stock-month observations")
    return df.sort_values(["permno", "date"]).reset_index(drop=True)


def load_characteristics() -> pd.DataFrame:
    """
    Loads the stock characteristics panel.
    Keeps only columns that exist in CHARACTERISTIC_COLUMNS.
    """
    logger.info("Loading stock characteristics...")
    df = pd.read_parquet(CHARACTERISTICS_FILE)
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)

    # Keep only the features that are both requested and available
    available = [c for c in CHARACTERISTIC_COLUMNS if c in df.columns]
    missing   = [c for c in CHARACTERISTIC_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(f"{len(missing)} characteristics not found and will be skipped: {missing[:5]}...")

    keep_cols = ["permno", "date"] + available
    df = df[keep_cols]
    # Remove duplicate columns if any exist
    df = df.loc[:, ~df.columns.duplicated()]
    logger.info(f"Characteristics loaded: {len(available)} features, {len(df):,} rows")
    return df


def load_macro() -> pd.DataFrame:
    """
    Loads the Welch-Goyal macro predictors, aligned to month-end.
    """
    logger.info("Loading macro predictors...")
    df = pd.read_parquet(MACRO_FILE)
    df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)

    available = [c for c in MACRO_COLUMNS if c in df.columns]
    df = df[available]
    logger.info(f"Macro predictors loaded: {len(available)} variables")
    return df


def load_rf() -> pd.Series:
    """
    Loads the monthly risk-free rate (1-month T-bill).

    Returns:
        Series indexed by date, named 'RF'.
    """
    df = pd.read_parquet(RF_FILE)
    df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
    return df["RF"].rename("RF")


# =============================================================
# 2. CROSS-SECTIONAL RANKING AND SCALING  (GKX 2020, Section 2.3)
# =============================================================

def rank_scale_characteristics(df: pd.DataFrame, char_cols: list[str]) -> pd.DataFrame:
    """
    GKX (2020) cross-sectional rank-scaling:
      1. Rank each characteristic within month → map to [-1, 1].
      2. Replace remaining missing values with cross-sectional median.
      3. Any still-missing values (entire-month NaN) → 0.
    """
    logger.info("Applying cross-sectional rank scaling...")

    def _scale_month(g: pd.DataFrame) -> pd.DataFrame:
        for col in char_cols:
            x = g[col]
            # If duplicate columns exist, x is a DataFrame — take the first column
            if isinstance(x, pd.DataFrame):
                x = x.iloc[:, 0]
            if x.notna().sum() < 2:
                continue
            lo, hi = x.quantile([0.01, 0.99])
            x = x.clip(lower=lo, upper=hi)
            r = x.rank(method="average", na_option="keep")
            n = r.notna().sum()
            x_scaled = 2 * (r - 1) / (n - 1) - 1
            g[col] = x_scaled
        return g

    df = df.groupby("date", group_keys=False).apply(_scale_month)

    # Step 2: fill remaining NaN with cross-sectional median (per month)
    logger.info("Filling missing values with cross-sectional median...")
    for col in char_cols:
        df[col] = df.groupby("date")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # Step 3: any still-NaN (entire month missing) → 0
    df[char_cols] = df[char_cols].fillna(0)

    return df


# =============================================================
# 3. MACRO INTERACTION FEATURES  (GKX 2020, Section 2.2)
# =============================================================

def build_interaction_features(
    char_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    char_cols: list[str],
) -> pd.DataFrame:
    """
    Constructs the full feature matrix by interacting firm characteristics
    with macro predictors as in GKX (2020):

        z_it = [c_it, c_it × q_t]

    where c_it is a (P,) vector of firm characteristics and q_t is the
    (K,) vector of macro predictors at time t.

    This expands the feature space from P to P*(K+1).

    Args:
        char_df:   Panel with [permno, date, char_cols...] — already rank-scaled.
        macro_df:  DataFrame indexed by date with macro columns.
        char_cols: Names of firm characteristic columns.

    Returns:
        DataFrame with [permno, date, <original chars>, <interaction cols>].
    """
    logger.info("Building macro interaction features...")

    macro_cols = macro_df.columns.tolist()

    # Merge macro predictors on date (lagged by 1 month to avoid look-ahead)
    macro_lagged = macro_df.shift(1)  # use info available at start of each month
    merged = char_df.join(macro_lagged, on="date", how="left")

    # Build interaction columns: c_it × q_t for each characteristic and macro var
    new_cols = {}
    for c in char_cols:
        for m in macro_cols:
            col_name = f"{c}_x_{m}"
            new_cols[col_name] = merged[c] * merged[m]

    interaction_df = pd.DataFrame(new_cols, index=merged.index)
    result = pd.concat([merged, interaction_df], axis=1)

    n_features = len(char_cols) + len(new_cols)
    logger.info(f"Feature matrix built: {n_features} total features "
                f"({len(char_cols)} chars + {len(new_cols)} interactions)")
    return result


# =============================================================
# 4. MERGE ALL DATA
# =============================================================

def build_panel(
    start: str = "1971-01-01",
    end:   str = "2025-12-31",
    use_interactions: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Master function: loads, cleans, merges, and creates the full feature
    panel including macro interactions and SIC industry dummies.
    """
    crsp_df  = load_crsp()
    char_df  = load_characteristics()
    macro_df = load_macro()
    rf       = load_rf()

    # Filter date range
    crsp_df  = crsp_df[(crsp_df["date"] >= start) & (crsp_df["date"] <= end)]
    char_df  = char_df[(char_df["date"] >= start) & (char_df["date"] <= end)]
    macro_df = macro_df[(macro_df.index >= start) & (macro_df.index <= end)]

    # Identify available characteristics
    char_cols_avail = [c for c in CHARACTERISTIC_COLUMNS if c in char_df.columns]

    # Rank-scale characteristics
    char_df = rank_scale_characteristics(char_df, char_cols_avail)

    # Optional macro interactions
    if use_interactions:
        char_df = build_interaction_features(char_df, macro_df, char_cols_avail)

    # Merge CRSP with characteristics
    panel = crsp_df[["permno", "date", "ret", "me", "siccd"]].merge(
        char_df,
        on=["permno", "date"],
        how="inner",
    )

    # Build SIC2 industry dummies (first 2 digits of SIC code, 74 categories per GKX)
    logger.info("Building SIC2 industry dummies...")
    panel["sic2"] = (panel["siccd"] // 100).astype("Int64")
    sic_dummies = pd.get_dummies(panel["sic2"], prefix="sic2", dtype=float)
    panel = pd.concat([panel.drop(columns=["sic2", "siccd"]), sic_dummies], axis=1)

    # Merge risk-free; compute excess return
    panel = panel.join(rf, on="date", how="left")
    panel["ret_excess"] = panel["ret"] - panel["RF"]

    # Drop thin months
    counts = panel.groupby("date")["permno"].transform("count")
    panel  = panel[counts >= MIN_STOCKS_PER_MONTH]

    panel = panel.sort_values(["date", "permno"]).reset_index(drop=True)

    # Final feature columns = everything except identifiers/targets
    non_feat = {"permno", "date", "ret", "ret_excess", "me", "RF",
                "shrcd", "exchcd", "siccd", "log_me", "sic2"}
    feature_cols = [c for c in panel.columns if c not in non_feat]

    logger.info(
        f"Panel built: {len(panel):,} stock-month obs | "
        f"{panel['permno'].nunique():,} unique stocks | "
        f"{panel['date'].nunique()} months | "
        f"{len(feature_cols)} features"
    )
    return panel, feature_cols


# =============================================================
# 5. ROLLING TRAIN / VALIDATION / TEST SPLIT
# =============================================================

def rolling_splits(
    panel: pd.DataFrame,
    test_start_year: int = TEST_START_YEAR,
    train_years:     int = TRAIN_YEARS,
    val_years:       int = VALIDATION_YEARS,
) -> Generator[dict, None, None]:
    """
    Generates rolling expanding-window splits (GKX 2020, Section 3):

      - Training set:   all data from sample_start to (test_start - val_years - 1)
      - Validation set: the val_years immediately before the test year
      - Test set:       a single year (12 months) — walk-forward one year at a time

    Yields:
        dict with keys:
          "year":       current test year
          "X_train":    feature matrix, training set
          "y_train":    excess returns, training set
          "w_train":    market-cap weights, training set (for weighted loss)
          "X_val":      feature matrix, validation set
          "y_val":      excess returns, validation set
          "w_val":      market-cap weights, validation set
          "X_test":     feature matrix, test set
          "y_test":     excess returns, test set (target for evaluation)
          "meta_test":  DataFrame with [permno, date, me] for portfolio construction
    """
    all_dates = panel["date"].sort_values().unique()
    all_years = sorted(pd.to_datetime(all_dates).year.unique())

    # Start yielding from test_start_year
    for test_year in [y for y in all_years if y >= test_start_year]:
        val_end_year   = test_year - 1
        val_start_year = test_year - val_years
        train_end_year = val_start_year - 1

        # Date masks
        is_train = (pd.to_datetime(panel["date"]).dt.year <= train_end_year)
        is_val   = (
            (pd.to_datetime(panel["date"]).dt.year >= val_start_year) &
            (pd.to_datetime(panel["date"]).dt.year <= val_end_year)
        )
        is_test  = (pd.to_datetime(panel["date"]).dt.year == test_year)

        train_df = panel[is_train]
        val_df   = panel[is_val]
        test_df  = panel[is_test]

        if len(train_df) < MIN_OBS_PER_STOCK or len(test_df) == 0:
            logger.debug(f"Skipping year {test_year}: insufficient data")
            continue

        # Feature column names = all columns except identifiers and targets
        non_feat = {"permno", "date", "ret", "ret_excess", "me", "RF",
                    "shrcd", "exchcd", "siccd", "log_me"}
        feat_cols = [c for c in panel.columns if c not in non_feat]

        def _arrays(df: pd.DataFrame):
            X = df[feat_cols].values.astype(np.float32)
            y = df["ret_excess"].values.astype(np.float32)
            w = df["me"].values.astype(np.float32)
            # Replace NaN in X with 0 (GKX missing-value imputation)
            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)
            w = np.nan_to_num(w, nan=1.0)
            return X, y, w

        X_train, y_train, w_train = _arrays(train_df)
        X_val,   y_val,   w_val   = _arrays(val_df)
        X_test,  y_test,  w_test  = _arrays(test_df)

        meta_test = test_df[["permno", "date", "me", "ret_excess"]].copy()

        logger.info(
            f"Split year={test_year} | "
            f"train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}"
        )

        yield {
            "year":       test_year,
            "feat_cols":  feat_cols,
            "X_train":    X_train,
            "y_train":    y_train,
            "w_train":    w_train,
            "X_val":      X_val,
            "y_val":      y_val,
            "w_val":      w_val,
            "X_test":     X_test,
            "y_test":     y_test,
            "meta_test":  meta_test,
        }
