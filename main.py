"""
main.py
-------
Central pipeline controller for Assignment 3:
"Replication of Gu, Kelly, and Xiu (2020) – Empirical Asset Pricing via ML"

Usage:
    python main.py [--period {1971_2016,1971_2025,both}] [--skip-download]

This script runs the full end-to-end workflow:
  1. Download data (CRSP, characteristics, macro predictors, risk-free rate)
  2. Build the feature panel (cleaning, rank-scaling, macro interactions)
  3. Run rolling out-of-sample predictions for all models
  4. Construct long-short portfolios
  5. Generate all required outputs:
       - Table 1 (LaTeX + CSV + figure)
       - Figure 4 (variable importance)
       - Figure 9 (cumulative returns)

The entire workflow is automated — no manual steps required after running
this script with valid CRSP access credentials.

Architecture overview:
  data_download.py     → downloads and caches all raw data
  data_processing.py   → cleans, merges, and produces rolling splits
  models.py            → implements all ML models
  rolling_regression.py → runs the OOS prediction loop
  portfolio_construction.py → forms decile portfolios and computes stats
  outputs.py           → generates tables and figures
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.settings import (
    SAMPLE_PERIODS, MODELS_TABLE1, MODELS_FIG4, MODELS_FIG9,
    SP500_FILE, RF_FILE,
    get_logger,
)
from src.data_download import download_all
from src.rolling_regression import run_rolling_predictions
from src.outputs import generate_all_outputs

logger = get_logger("main")


# =============================================================
# ARGUMENT PARSING
# =============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GKX 2020 Replication: Empirical Asset Pricing via ML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--period",
        choices=["1971_2016", "1971_2025", "both"],
        default="both",
        help="Which sample period(s) to run.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        default=False,
        help="Skip data download step (use cached data).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Recompute predictions even if cache exists.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Run only specific models (e.g., --models OLS+H RF NN2). "
             "Default: all models.",
    )
    return parser.parse_args()


# =============================================================
# MAIN PIPELINE
# =============================================================

def run_pipeline(
    period_label: str,
    models: list[str],
    use_cache: bool = True,
) -> None:
    """
    Runs the full pipeline for a single sample period.

    Args:
        period_label: One of the keys in SAMPLE_PERIODS (e.g., "1971_2016").
        models:       List of model names to include.
        use_cache:    Whether to reuse cached OOS predictions.
    """
    period  = SAMPLE_PERIODS[period_label]
    start   = period["start"]
    end     = period["end"]

    logger.info("=" * 70)
    logger.info(f"RUNNING PIPELINE: {period_label}  [{start} → {end}]")
    logger.info(f"Models: {models}")
    logger.info("=" * 70)
    t0 = time.time()

    # -------------------------------------------------------
    # Step 3: Run rolling OOS predictions
    # -------------------------------------------------------
    logger.info("STEP 3: Rolling out-of-sample prediction loop")
    all_predictions, feature_importances = run_rolling_predictions(
        start=start,
        end=end,
        period_label=period_label,
        models=models,
        use_cache=use_cache,
    )

    # Identify feature columns from a representative prediction DataFrame
    feat_cols: list[str] = []
    for df in all_predictions.values():
        if hasattr(df, "feat_cols_"):
            feat_cols = df.feat_cols_
            break

    # -------------------------------------------------------
    # Step 4: Load S&P 500 and risk-free rate for Figure 9
    # -------------------------------------------------------
    import pandas as pd
    logger.info("STEP 4: Loading S&P 500 and risk-free rate")
    try:
        sp500_df = pd.read_parquet(SP500_FILE)
        sp500_df.index = pd.to_datetime(sp500_df.index) + pd.offsets.MonthEnd(0)
        sp500_rets = sp500_df["sp500_ret"]

        rf_df  = pd.read_parquet(RF_FILE)
        rf_df.index = pd.to_datetime(rf_df.index) + pd.offsets.MonthEnd(0)
        rf_rets = rf_df["RF"]
    except FileNotFoundError:
        logger.warning("S&P 500 / RF files not found; Figure 9 benchmark omitted.")
        sp500_rets = pd.Series(dtype=float)
        rf_rets    = pd.Series(dtype=float)

    # -------------------------------------------------------
    # Step 5: Generate all outputs
    # -------------------------------------------------------
    logger.info("STEP 5: Generating tables and figures")
    generate_all_outputs(
        all_predictions=all_predictions,
        feature_importances=feature_importances,
        feat_cols=feat_cols,
        sp500_rets=sp500_rets,
        rf_rets=rf_rets,
        period_label=period_label,
    )

    elapsed = time.time() - t0
    logger.info(f"Pipeline for {period_label} completed in {elapsed/60:.1f} minutes.")


# =============================================================
# ENTRY POINT
# =============================================================

def main() -> None:
    args   = parse_args()
    models = args.models or MODELS_TABLE1

    logger.info("=" * 70)
    logger.info("GKX (2020) REPLICATION — BME.70034: Empirical Asset Pricing")
    logger.info("=" * 70)

    # -------------------------------------------------------
    # Step 1: Download data (unless --skip-download)
    # -------------------------------------------------------
    if not args.skip_download:
        logger.info("STEP 1: Downloading all data sources")
        # Use the broadest date range to cover both sample periods
        download_all(start="1971-01-01", end="2025-12-31")
    else:
        logger.info("STEP 1: Skipping data download (--skip-download)")

    # -------------------------------------------------------
    # Step 2: Determine which sample periods to run
    # -------------------------------------------------------
    if args.period == "both":
        periods = ["1971_2016", "1971_2025"]
    else:
        periods = [args.period]

    # -------------------------------------------------------
    # Step 3-5: Run pipeline for each period
    # -------------------------------------------------------
    for period_label in periods:
        run_pipeline(
            period_label=period_label,
            models=models,
            use_cache=not args.no_cache,
        )

    logger.info("=" * 70)
    logger.info("ALL DONE. Check results/ for tables and figures.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
