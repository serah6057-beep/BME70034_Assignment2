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
# CHARACTERISTICS EXTENSION (Xiu + OSAP 2017-2024)
# =============================================================

# Complete Xiu ↔ OSAP variable name mapping
XIU_TO_OSAP = {
    # ===== Direct matches =====
    "acc":        "Accruals",
    "agr":        "AssetGrowth",
    "bm":         "BM",
    "cash":       "Cash",
    "cfp":        "cfp",
    "chinv":      "ChInv",
    "chnanalyst": "ChNAnalyst",
    "chtx":       "ChTax",
    "convind":    "ConvDebt",
    "divi":       "DivInit",
    "divo":       "DivOmit",
    "dolvol":     "DolVol",
    "ear":        "EarningsSurprise",
    "ep":         "EP",
    "gma":        "GP",
    "grcapx":     "grcapx",
    "grltnoa":    "GrLTNOA",
    "herf":       "Herf",
    "hire":       "hire",
    "idiovol":    "IdioVol3F",
    "ill":        "Illiquidity",
    "indmom":     "IndMom",
    "invest":     "Investment",
    "lev":        "Leverage",
    "maxret":     "MaxRet",
    "mom12m":     "Mom12m",
    "mom1m":      "STreversal",
    "mom36m":     "LRreversal",
    "mom6m":      "Mom6m",
    "ms":         "MS",
    "nincr":      "NumEarnIncrease",
    "operprof":   "OperProf",
    "orgcap":     "OrgCap",
    "pctacc":     "PctAcc",
    "pricedelay": "PriceDelayRsq",
    "ps":         "PS",
    "rd":         "RD",
    "realestate": "realestate",
    "retvol":     "RealizedVol",
    "roaq":       "roaq",
    "roeq":       "RoE",
    "sgr":        "GrSaleToGrInv",
    "sin":        "sinAlgo",
    "sp":         "SP",
    "std_turn":   "std_turn",
    "tang":       "tang",
    "zerotrade":  "zerotrade1M",
    # ===== Derived (computed from OSAP) =====
    "absacc":     "Accruals",
    "baspread":   "BidAskSpread",
    "beta":       "Beta",
    "betasq":     "Beta",
    "chcsho":     "ShareIss1Y",
    "chmom":      "MomVol",
    "egr":        "ChEQ",
    "mvel1":      "Size",
    "stdacc":     "Accruals",
    "stdcf":      "VarCF",
    # ===== Industry-adjusted proxies =====
    "bm_ia":      "BM",
    "cfp_ia":     "cfp",
    "chatoia":    "ChAssetTurnover",
    "chempia":    "hire",
    "chpmia":     "OperProf",
    "mve_ia":     "Size",
    "pchcapx_ia": "grcapx",
    # ===== Closest available proxies =====
    "age":            "AgeIPO",
    "cinvest":        "InvestPPEInv",
    "dy":             "PayoutYield",
    "pchsaleinv":     "GrSaleToGrInv",
    "rd_mve":         "RD",
    "rd_sale":        "RD",
    "roic":           "RoE",
    "rsup":           "RevenueSurprise",
    "tb":             "ChTax",
}

# Xiu variables with NO OSAP counterpart — filled with 0 for 2017-2024
XIU_NO_OSAP = [
    "aeavol", "cashdebt", "cashpr", "currat", "depr",
    "pchcurrat", "pchdepr", "pchgm_pchsale", "pchquick",
    "pchsale_pchinvt", "pchsale_pchrect", "pchsale_pchxsga",
    "quick", "roavol", "salecash", "saleinv", "salerec",
    "secured", "securedind", "std_dolvol", "turn",
]

def extend_characteristics_with_osap() -> pd.DataFrame:
    """
    Extends the Xiu characteristics dataset (ends in 2016) with
    Open Source Asset Pricing data (Chen & Zimmermann) for 2017-2024.
    
    Memory-safe: downloads OSAP signals ONE AT A TIME, filters to 2017-2024,
    downcasts to float32, and caches everything to disk.
    
    Variables are renamed from OSAP convention to Xiu/GKX convention.
    Xiu variables that have no OSAP counterpart are filled with 0
    for the 2017-2024 extension.
    """
    extended_file = DATA_DIR / "characteristics_extended.parquet"
    if extended_file.exists():
        logger.info(f"Loading extended characteristics from cache: {extended_file}")
        return pd.read_parquet(extended_file)

    # 1) Load existing Xiu characteristics (≤ 2016)
    logger.info("Loading existing Xiu characteristics...")
    xiu_df = pd.read_parquet(CHARACTERISTICS_FILE)
    xiu_df["date"] = pd.to_datetime(xiu_df["date"]) + pd.offsets.MonthEnd(0)
    xiu_df = xiu_df[xiu_df["date"] <= "2016-12-31"]

    # Identify Xiu's full variable set (excluding identifier columns)
    xiu_cols = [c for c in xiu_df.columns if c not in ("permno", "date", "sic2")]

    # 2) Download / load OSAP signals for 2017-2024 (memory-safe)
    try:
        from openassetpricing import OpenAP
    except ImportError:
        raise ImportError("Install openassetpricing: pip install openassetpricing")

    osap_needed = sorted(set(v for v in XIU_TO_OSAP.values() if v is not None))

    # Cache the merged OSAP raw download so we never re-fetch
    osap_cache = DATA_DIR / "osap_raw_2017_2024.parquet"

    if osap_cache.exists():
        logger.info(f"Loading OSAP cache: {osap_cache}")
        osap_df = pd.read_parquet(osap_cache)
    else:
        logger.info(f"Downloading {len(osap_needed)} OSAP signals one-by-one...")
        oap = OpenAP()

        tmp_files = []
        for i, sig in enumerate(osap_needed, 1):
            logger.info(f"  [{i}/{len(osap_needed)}] {sig} ...")
            try:
                df_one = oap.dl_signal("pandas", [sig])
            except Exception as e:
                logger.warning(f"    Failed to download {sig}: {e}")
                continue

            # Date conversion
            if "yyyymm" in df_one.columns:
                df_one["date"] = (
                    pd.to_datetime(df_one["yyyymm"].astype(str), format="%Y%m")
                    + pd.offsets.MonthEnd(0)
                )
                df_one = df_one.drop(columns=["yyyymm"])

            # Filter to 2017-2024 immediately (cuts data ~5x)
            df_one = df_one[
                (df_one["date"] >= "2017-01-01") & (df_one["date"] <= "2024-12-31")
            ].copy()

            # Downcast to float32 to halve memory
            for col in df_one.columns:
                if col not in ("permno", "date"):
                    df_one[col] = pd.to_numeric(df_one[col], errors="coerce").astype("float32")
            df_one["permno"] = df_one["permno"].astype("int64")

            # Save to per-signal temp file
            tmp = DATA_DIR / f"_osap_tmp_{sig}.parquet"
            df_one.to_parquet(tmp, index=False)
            tmp_files.append((sig, tmp))
            del df_one

        # Merge all per-signal files on (permno, date)
        logger.info("Merging per-signal files...")
        osap_df = None
        for sig, tmp in tmp_files:
            chunk = pd.read_parquet(tmp)
            if osap_df is None:
                osap_df = chunk
            else:
                osap_df = osap_df.merge(chunk, on=["permno", "date"], how="outer")
            del chunk
            tmp.unlink()

        # Cache merged result so future runs skip the entire download
        osap_df.to_parquet(osap_cache, index=False)
        logger.info(f"OSAP raw data cached: {osap_cache}")

    logger.info(f"OSAP data: {len(osap_df):,} rows × {len(osap_df.columns)} columns")

    # 3) Build the Xiu-format DataFrame from OSAP
    logger.info("Mapping OSAP variables to Xiu format...")
    extended_2017 = osap_df[["permno", "date"]].copy()

    for xiu_var in xiu_cols:
        osap_var = XIU_TO_OSAP.get(xiu_var)
        if osap_var is None or osap_var not in osap_df.columns:
            extended_2017[xiu_var] = 0.0
        elif xiu_var == "absacc":
            extended_2017[xiu_var] = osap_df["Accruals"].abs()
        elif xiu_var == "betasq":
            extended_2017[xiu_var] = osap_df["Beta"] ** 2
        else:
            extended_2017[xiu_var] = osap_df[osap_var].values

    # Add sic2 column (NA placeholder; CRSP merge fills it later)
    if "sic2" in xiu_df.columns:
        extended_2017["sic2"] = pd.NA

    # Log coverage diagnostic
    logger.info("OSAP variable coverage (% non-zero, non-NaN):")
    for xv in ["mom12m", "mom6m", "mom1m", "indmom", "ill", "bm", "ep", "beta"]:
        if xv in extended_2017.columns:
            v = extended_2017[xv]
            nz = ((v.notna()) & (v != 0)).mean() * 100
            logger.info(f"  {xv}: {nz:.1f}%")

    # 4) Concatenate by streaming to parquet (memory-safe)
    logger.info("Concatenating Xiu (≤2016) + OSAP (2017-2024) to disk...")
    import pyarrow as pa
    import pyarrow.parquet as pq

    xiu_df = xiu_df.sort_values(["permno", "date"]).reset_index(drop=True)
    extended_2017 = extended_2017.sort_values(["permno", "date"]).reset_index(drop=True)

    # Align column order
    common_cols = [c for c in xiu_df.columns if c in extended_2017.columns]
    xiu_df = xiu_df[common_cols]
    extended_2017 = extended_2017[common_cols]

    # Align dtypes across both pieces
    xiu_df["permno"] = xiu_df["permno"].astype("int64")
    extended_2017["permno"] = extended_2017["permno"].astype("int64")
    xiu_df["date"] = pd.to_datetime(xiu_df["date"])
    extended_2017["date"] = pd.to_datetime(extended_2017["date"])

    if "sic2" in common_cols:
        xiu_df["sic2"] = pd.to_numeric(xiu_df["sic2"], errors="coerce").astype("float64")
        extended_2017["sic2"] = pd.to_numeric(extended_2017["sic2"], errors="coerce").astype("float64")

    skip = {"permno", "date", "sic2"}
    for col in common_cols:
        if col in skip:
            continue
        xiu_df[col] = pd.to_numeric(xiu_df[col], errors="coerce").astype("float32")
        extended_2017[col] = pd.to_numeric(extended_2017[col], errors="coerce").astype("float32")

    # Stream both pieces to parquet
    table1 = pa.Table.from_pandas(xiu_df, preserve_index=False)
    writer = pq.ParquetWriter(str(extended_file), table1.schema)
    writer.write_table(table1)
    del xiu_df, table1

    table2 = pa.Table.from_pandas(extended_2017, preserve_index=False)
    writer.write_table(table2)
    del extended_2017, table2
    writer.close()

    logger.info(f"Extended characteristics saved → {extended_file}")
    return pd.read_parquet(extended_file)

# =============================================================
# 3. MACRO PREDICTORS  (Welch & Goyal 2008)
# =============================================================

def download_macro_predictors() -> pd.DataFrame:
    """
    Downloads Welch-Goyal macro predictors via the tidyfinance package.
    Keeps only the 8 variables used in GKX (2020): dp, ep, bm, ntis,
    tbl, tms, dfy, svar.
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

    # Set date index (tidyfinance returns 'month' column)
    if "month" in df.columns:
        df["date"] = pd.to_datetime(df["month"]) + pd.offsets.MonthEnd(0)
        df = df.set_index("date").drop(columns=["month"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
        df = df.set_index("date")

    df = df.sort_index()

    # Compute derived columns if not already present
    if "tms" not in df.columns and "lty" in df.columns and "tbl" in df.columns:
        df["tms"] = df["lty"] - df["tbl"]
    if "dfy" not in df.columns and "baa" in df.columns and "aaa" in df.columns:
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
