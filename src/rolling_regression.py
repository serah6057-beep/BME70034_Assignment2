"""
src/rolling_regression.py
--------------------------
Orchestrates the rolling out-of-sample (OOS) prediction loop.

For each test year:
  1. Retrieve train/val/test split from data_processing.rolling_splits()
  2. Fit each model on the training set (with val for hyperparameter tuning)
  3. Generate out-of-sample predictions on the test set
  4. Collect predictions and realized returns for evaluation

This module also computes the aggregate OOS R² (Table 1, GKX 2020 Eq. 2)
and stores per-month predictions for portfolio construction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.settings import (
    MODELS_TABLE1, RESULTS_DIR, get_logger,
)
from src.data_processing import build_panel, rolling_splits
from src.models import get_model, oos_r2

logger = get_logger(__name__)

# Cache file for OOS predictions (avoids recomputing everything)
_PRED_CACHE_TMPL = str(RESULTS_DIR / "predictions_{period}.parquet")


# =============================================================
# MAIN ROLLING PREDICTION LOOP
# =============================================================

def run_rolling_predictions(
    start:       str,
    end:         str,
    period_label: str,
    models:      list[str] = MODELS_TABLE1,
    use_cache:   bool      = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, list]]:
    """
    Runs the full rolling OOS prediction pipeline for all specified models.

    Args:
        start:        Sample start date (e.g., "1971-01-01").
        end:          Sample end date   (e.g., "2016-12-31").
        period_label: Label for caching (e.g., "1971_2016").
        models:       List of model names to run.
        use_cache:    If True, load cached predictions if available.

    Returns:
        (all_predictions, feature_importances) where:
          all_predictions  : {model_name: DataFrame[date, permno, ret_excess, me, y_pred]}
          feature_importances: {model_name: list of pd.Series (one per test year)}
    """
    cache_file = Path(_PRED_CACHE_TMPL.format(period=period_label))

    # ---- Load from cache if available ----
    if use_cache and cache_file.exists():
        logger.info(f"Loading cached predictions from {cache_file}")
        pred_df = pd.read_parquet(cache_file)
        # Reshape back to per-model dict
        all_predictions = {
            m: pred_df[pred_df["model"] == m].drop(columns=["model"]).copy()
            for m in pred_df["model"].unique()
        }
        # Feature importances are not cached; return empty
        return all_predictions, {m: [] for m in models}

    # ---- Build the full feature panel ----
    logger.info(f"Building panel for period {start} → {end}")
    panel, feat_cols = build_panel(start=start, end=end, use_interactions=True)

    # Storage for results
    all_predictions:    dict[str, list] = {m: [] for m in models}
    feature_importances: dict[str, list] = {m: [] for m in models}

    # ---- Rolling loop ----
    splits = list(rolling_splits(panel, test_start_year=1987))
    logger.info(f"Running {len(splits)} rolling windows × {len(models)} models")

    for split in tqdm(splits, desc="Rolling windows"):
        year     = split["year"]
        meta     = split["meta_test"]
        X_train  = split["X_train"]
        y_train  = split["y_train"]
        X_val    = split["X_val"]
        y_val    = split["y_val"]
        X_test   = split["X_test"]
        y_test   = split["y_test"]

        for model_name in models:
            logger.debug(f"Year={year} | Model={model_name}")

            # Instantiate and fit
            model = get_model(model_name)
            try:
                model.fit(
                    X_train, y_train,
                    X_val=X_val, y_val=y_val,
                    feat_cols=feat_cols,
                )
            except Exception as e:
                logger.error(f"  FIT ERROR [{model_name}, {year}]: {e}")
                continue

            # Generate predictions
            try:
                y_pred = model.predict(X_test)
            except Exception as e:
                logger.error(f"  PREDICT ERROR [{model_name}, {year}]: {e}")
                continue

            # Store predictions with metadata
            pred_chunk = meta.copy()
            pred_chunk["y_pred"]  = y_pred
            pred_chunk["y_true"]  = y_test
            all_predictions[model_name].append(pred_chunk)

            # Store feature importance (for Figure 4)
            try:
                fi = model.feature_importance(feat_cols)
                fi.name = year
                feature_importances[model_name].append(fi)
            except Exception as e:
                logger.debug(f"  FI ERROR [{model_name}, {year}]: {e}")

    # ---- Concatenate results ----
    combined = {}
    for model_name, chunks in all_predictions.items():
        if chunks:
            combined[model_name] = pd.concat(chunks, ignore_index=True)
        else:
            logger.warning(f"No predictions generated for {model_name}")

    # ---- Cache predictions ----
    cache_rows = []
    for model_name, df in combined.items():
        df = df.copy()
        df["model"] = model_name
        cache_rows.append(df)

    if cache_rows:
        pd.concat(cache_rows, ignore_index=True).to_parquet(cache_file, index=False)
        logger.info(f"Predictions cached to {cache_file}")

    return combined, feature_importances


# =============================================================
# COMPUTE OOS R² PER MODEL (TABLE 1)
# =============================================================

def compute_oos_r2_table(
    all_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Computes the aggregate out-of-sample R² for each model,
    following GKX (2020) Equation (2):

        R²_oos = 1 - Σ_t Σ_i (r_it - r̂_it)² / Σ_t Σ_i r_it²

    Args:
        all_predictions: {model_name: DataFrame[ret_excess, y_pred, me]}

    Returns:
        DataFrame with index = model names, column = "OOS_R2" (as %)
    """
    rows = []
    for model_name, df in all_predictions.items():
        y_true  = df["ret_excess"].values
        y_pred  = df["y_pred"].values
        weights = df["me"].values

        r2 = oos_r2(y_true, y_pred, weights=weights)
        rows.append({"Model": model_name, "OOS_R2 (%)": r2 * 100})
        logger.info(f"  {model_name}: OOS R² = {r2 * 100:.3f}%")

    return pd.DataFrame(rows).set_index("Model")
