"""
src/data_download.py
--------------------
Downloads and caches all raw data required for the GKX (2020) replication.

Data sources:
  1. CRSP monthly stock returns  → via WRDS Python API
  2. Stock characteristics       → Dacheng Xiu's website (cached parquet)
  3. Macro predictors            → Tidy Finance / Welch-Goyal (2008)
  4. Risk-free rate              → Ken French Data Library
  5. S&P 500 returns             → Ken French Data Library

IMPORTANT: Raw files are saved to data/ which is git-ignored.
           Only call this module once; subsequent runs load from cache.
"""

import os
from dotenv import load_dotenv
import io
import time
import zipfile
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import pandas_datareader.data as web

from src.settings import (
    DATA_DIR, CRSP_FILE, CHARACTERISTICS_FILE,
    MACRO_FILE, RF_FILE, SP500_FILE,
    get_logger,
)

logger = get_logger(__name__)

# =============================================================
# 1. CRSP MONTHLY DATA  (via WRDS)
# =============================================================

def download_crsp(start: str = "1971-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """
    Downloads NYSE common stock monthly returns from CRSP via WRDS.

    Filters applied (following GKX 2020 and standard literature):
      - shrcd in (10, 11)  : U.S. common stocks only
      - exchcd == 1         : NYSE stocks only
      - prc > 0             : positive price (drop if-then quotes)
      - me > 0              : positive market equity

    Returns:
        DataFrame with columns [permno, date, ret, me, shrcd, exchcd]
        Saved to CRSP_FILE for reuse.
    """
    # Load from cache if already downloaded
    if CRSP_FILE.exists():
        logger.info(f"Loading CRSP data from cache: {CRSP_FILE}")
        return pd.read_parquet(CRSP_FILE)

    load_dotenv()

    logger.info("Downloading CRSP monthly data from WRDS...")
    try:
        import wrds
    except ImportError:
        raise ImportError("Install 'wrds' package: pip install wrds")

    # Connect to WRDS using credentials from .env
    os.environ["WRDS_USERNAME"] = os.getenv("WRDS_USERNAME")
    os.environ["WRDS_PASSWORD"] = os.getenv("WRDS_PASSWORD")

    db = wrds.Connection()

    query = f"""
    SELECT
        a.permno,
        a.date,
        a.ret,
        a.retx,
        ABS(a.prc) AS prc,
        a.shrout,
        ABS(a.prc) * a.shrout / 1000 AS me,
        a.vol,
        b.shrcd,
        b.exchcd,
        b.siccd
    FROM crsp.msf AS a
    INNER JOIN crsp.msenames AS b
        ON a.permno = b.permno
        AND b.namedt <= a.date
        AND a.date <= b.nameendt
    WHERE a.date BETWEEN '{start}' AND '{end}'
      AND b.exchcd = 1
      AND b.shrcd IN (10, 11)
      AND ABS(a.prc) > 0
"""
    df = db.raw_sql(query, date_cols=["date"])
    db.close()

    # Compute log market equity for size sorting
    df["log_me"] = np.log(df["me"])

    # Remove duplicates (keep last record per permno-date)
    df = df.drop_duplicates(subset=["permno", "date"], keep="last")

    # Save to parquet
    df.to_parquet(CRSP_FILE, index=False)
    logger.info(f"CRSP data saved: {len(df):,} rows → {CRSP_FILE}")
    return df


# =============================================================
# 2. STOCK CHARACTERISTICS  (Xiu's dataset)
# =============================================================

def download_characteristics() -> pd.DataFrame:
    if CHARACTERISTICS_FILE.exists():
        logger.info(f"Loading characteristics from cache: {CHARACTERISTICS_FILE}")
        return pd.read_parquet(CHARACTERISTICS_FILE)

    import pyarrow as pa
    import pyarrow.parquet as pq
    import shutil

    url = "https://dachxiu.chicagobooth.edu/download/datashare.zip"
    zip_path = DATA_DIR / "datashare.zip"

    # Download only if zip not already on disk
    if not zip_path.exists():
        logger.info(f"Downloading characteristics dataset from: {url}")
        with requests.get(url, timeout=300, stream=True) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)

    # Extract CSV to disk
    logger.info("Extracting zip to disk...")
    with zipfile.ZipFile(zip_path) as z:
        fname = [f for f in z.namelist() if f.endswith(".csv")][0]
        z.extract(fname, DATA_DIR)
        extracted_path = DATA_DIR / fname

    # Read in chunks and write directly to parquet (no RAM accumulation)
    logger.info("Converting CSV to parquet in chunks...")
    writer = None
    for chunk in pd.read_csv(extracted_path, low_memory=False, chunksize=50000):
        chunk.columns = chunk.columns.str.lower().str.strip()
        if "date" in chunk.columns:
            chunk["date"] = pd.to_datetime(chunk["date"].astype(str), format="%Y%m%d") + pd.offsets.MonthEnd(0)
        elif "yyyymm" in chunk.columns:
            chunk["date"] = pd.to_datetime(chunk["yyyymm"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
            chunk.drop(columns=["yyyymm"], inplace=True)

        table = pa.Table.from_pandas(chunk)
        if writer is None:
            writer = pq.ParquetWriter(str(CHARACTERISTICS_FILE), table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

    # Clean up temporary files
    zip_path.unlink()
    extracted_path.unlink()

    logger.info(f"Characteristics saved to {CHARACTERISTICS_FILE}")
    return pd.read_parquet(CHARACTERISTICS_FILE)


# =============================================================
# 3. MACRO PREDICTORS  (Welch & Goyal 2008)
# =============================================================

def download_macro_predictors() -> pd.DataFrame:
    """
    Downloads Welch-Goyal macro predictors via the tidyfinance package.
    """
    if MACRO_FILE.exists():
        logger.info(f"Loading macro predictors from cache: {MACRO_FILE}")
        return pd.read_parquet(MACRO_FILE)

    import tidyfinance as tf
    logger.info("Downloading Welch-Goyal macro predictors via tidyfinance...")

    df = tf.download_data(
        domain="macro_predictors",
        dataset="monthly",
        start_date="1960-01-01",
        end_date="2025-12-31",
    )

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Set date index
    if "month" in df.columns:
        df["date"] = pd.to_datetime(df["month"]) + pd.offsets.MonthEnd(0)
        df = df.set_index("date").drop(columns=["month"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
        df = df.set_index("date")

    df = df.sort_index()

    # Derived columns
    if "lty" in df.columns and "tbl" in df.columns:
        df["tms"] = df["lty"] - df["tbl"]
    if "baa" in df.columns and "aaa" in df.columns:
        df["dfy"] = df["baa"] - df["aaa"]

    df.to_parquet(MACRO_FILE)
    logger.info(f"Macro predictors saved: {len(df)} months → {MACRO_FILE}")
    return df


# =============================================================
# 4. RISK-FREE RATE AND S&P 500 (Ken French Data Library)
# =============================================================

def download_rf_and_sp500() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads monthly risk-free rate (1-month T-bill) and S&P 500 returns
    from Ken French's data library.

    Returns:
        (rf_df, sp500_df) — both indexed by date (month-end).
    """
    if RF_FILE.exists() and SP500_FILE.exists():
        logger.info("Loading risk-free rate and S&P 500 from cache.")
        return pd.read_parquet(RF_FILE), pd.read_parquet(SP500_FILE)

    logger.info("Downloading Fama-French factors from Ken French Data Library...")
    ff3 = web.DataReader(
        "F-F_Research_Data_Factors",
        "famafrench",
        start="1971-01-01",
    )[0] / 100  # Convert from % to decimal

    # Risk-free rate: RF column of the FF3 factor table
    rf_df = ff3[["RF"]].copy()
    rf_df.index = rf_df.index.to_timestamp("M") + pd.offsets.MonthEnd(0)
    rf_df.index.name = "date"

    # S&P 500: download separately (Market return = Rm-Rf + Rf)
    rf_df["Mkt-RF"] = ff3["Mkt-RF"]
    rf_df["sp500_ret"] = rf_df["Mkt-RF"] + rf_df["RF"]  # Gross market return
    sp500_df = rf_df[["sp500_ret"]].copy()

    rf_df = rf_df[["RF"]]

    rf_df.to_parquet(RF_FILE)
    sp500_df.to_parquet(SP500_FILE)
    logger.info(f"RF and S&P500 saved.")
    return rf_df, sp500_df


# =============================================================
# MASTER DOWNLOAD FUNCTION
# =============================================================

def download_all(start: str = "1971-01-01", end: str = "2025-12-31") -> None:
    """
    Entry point: downloads all required datasets.
    Skips any dataset that is already cached locally.

    Args:
        start: Sample start date (YYYY-MM-DD).
        end:   Sample end date (YYYY-MM-DD).
    """
    logger.info("=" * 60)
    logger.info("Starting data download pipeline")
    logger.info(f"Sample window: {start} to {end}")
    logger.info("=" * 60)

    # 1. CRSP returns
    download_crsp(start=start, end=end)

    # 2. Stock characteristics
    download_characteristics()

    # 3. Macro predictors
    download_macro_predictors()

    # 4. Risk-free rate and S&P 500
    download_rf_and_sp500()

    logger.info("All data downloaded and cached successfully.")
