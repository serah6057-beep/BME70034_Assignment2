"""
src/portfolio_construction.py
------------------------------
Constructs long-short portfolios from ML return predictions and computes
performance statistics for GKX (2020) Table 1 and Figure 9.

Logic:
  1. Each month, sort stocks into deciles based on predicted excess return.
  2. Long top decile, short bottom decile (value-weighted).
  3. Compute monthly portfolio returns, cumulative returns, and risk metrics.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.settings import (
    PORT_DECILES, LONG_DECILE, SHORT_DECILE,
    WEIGHTING, get_logger,
)

logger = get_logger(__name__)


# =============================================================
# 1. DECILE PORTFOLIO FORMATION
# =============================================================

def form_decile_portfolios(
    predictions_df: pd.DataFrame,
    n_deciles: int = PORT_DECILES,
    weighting: str = WEIGHTING,
) -> pd.DataFrame:
    """
    Forms value-weighted decile portfolios for each month based on model
    predictions of excess returns.

    Args:
        predictions_df: DataFrame with columns:
                        [date, permno, ret_excess, me, y_pred]
        n_deciles:      Number of decile bins (default 10).
        weighting:      "value" (market-cap weights) or "equal".

    Returns:
        DataFrame with columns [date, decile, port_ret] where:
          - decile ∈ {1, 2, ..., n_deciles}
          - port_ret is the realized monthly portfolio return
    """
    results = []

    for date, group in predictions_df.groupby("date"):
        if len(group) < n_deciles:
            continue  # Skip months with too few stocks

        # Assign decile based on predicted return
        group = group.copy()
        group["decile"] = pd.qcut(
            group["y_pred"],
            q=n_deciles,
            labels=False,
            duplicates="drop",
        ) + 1  # Decile 1 = lowest predicted return, 10 = highest

        for decile in range(1, n_deciles + 1):
            dec_stocks = group[group["decile"] == decile]
            if len(dec_stocks) == 0:
                continue

            if weighting == "value":
                # Value-weighted: weight by beginning-of-month market equity
                weights = dec_stocks["me"].values
                weights = weights / weights.sum()
                port_ret = np.dot(weights, dec_stocks["ret_excess"].values)
            else:
                # Equal-weighted
                port_ret = dec_stocks["ret_excess"].mean()

            results.append({
                "date":     date,
                "decile":   decile,
                "port_ret": port_ret,
                "n_stocks": len(dec_stocks),
            })

    return pd.DataFrame(results)


# =============================================================
# 2. LONG-SHORT PORTFOLIO (HIGH - LOW DECILE)
# =============================================================

def long_short_portfolio(
    decile_df: pd.DataFrame,
    long_dec:  int = LONG_DECILE,
    short_dec: int = SHORT_DECILE,
) -> pd.Series:
    """
    Computes the long-short portfolio return series:
        LS_t = return_t(decile=long_dec) - return_t(decile=short_dec)

    Args:
        decile_df: Output of form_decile_portfolios().
        long_dec:  Decile to go long (default = 10).
        short_dec: Decile to go short (default = 1).

    Returns:
        Series indexed by date with long-short monthly returns.
    """
    long_rets  = decile_df[decile_df["decile"] == long_dec].set_index("date")["port_ret"]
    short_rets = decile_df[decile_df["decile"] == short_dec].set_index("date")["port_ret"]

    # Align dates (inner join)
    ls_rets = (long_rets - short_rets).dropna()
    ls_rets.name = "ls_ret"
    return ls_rets


# =============================================================
# 3. PERFORMANCE STATISTICS (TABLE 1 METRICS)
# =============================================================

def compute_performance_stats(
    ret_series: pd.Series,
    annualization: int = 12,
) -> dict:
    """
    Computes standard portfolio performance metrics reported in GKX Table 1.

    Args:
        ret_series:    Monthly return series (excess over risk-free).
        annualization: Periods per year (12 for monthly data).

    Returns:
        Dictionary with keys:
          mean_ret:    Annualized mean excess return (%)
          std_ret:     Annualized std dev (%)
          sharpe:      Annualized Sharpe ratio
          t_stat:      t-statistic for mean ≠ 0
          max_dd:      Maximum drawdown (%)
          skew:        Monthly return skewness
          avg_n:       (if available) average number of stocks in portfolio
    """
    rets = ret_series.dropna()
    n    = len(rets)

    mean_monthly = rets.mean()
    std_monthly  = rets.std(ddof=1)

    mean_ann = mean_monthly * annualization * 100        # annualized, in %
    std_ann  = std_monthly  * np.sqrt(annualization) * 100

    sharpe   = (mean_monthly / std_monthly * np.sqrt(annualization)
                if std_monthly > 0 else np.nan)

    # t-statistic: H0: mean = 0
    t_stat   = (mean_monthly / (std_monthly / np.sqrt(n))
                if std_monthly > 0 else np.nan)

    # Maximum drawdown
    cum_rets = (1 + rets).cumprod()
    rolling_max = cum_rets.cummax()
    drawdowns   = (cum_rets - rolling_max) / rolling_max
    max_dd      = drawdowns.min() * 100   # in %

    skew = float(rets.skew())

    return {
        "mean_ret": mean_ann,
        "std_ret":  std_ann,
        "sharpe":   sharpe,
        "t_stat":   t_stat,
        "max_dd":   max_dd,
        "skew":     skew,
    }


# =============================================================
# 4. CUMULATIVE RETURN SERIES (FIGURE 9)
# =============================================================

def cumulative_return(ret_series: pd.Series, start_value: float = 1.0) -> pd.Series:
    """
    Computes the cumulative total return index starting at start_value.

    Args:
        ret_series:  Monthly excess return series.
        start_value: Initial portfolio value (default $1).

    Returns:
        pd.Series of cumulative wealth indexed by date.
    """
    return (1 + ret_series).cumprod() * start_value


# =============================================================
# 5. AGGREGATE ACROSS ALL MODELS
# =============================================================

def build_portfolio_returns(
    all_predictions: dict[str, pd.DataFrame],
    sp500_rets: Optional[pd.Series] = None,
    rf_rets:    Optional[pd.Series] = None,
) -> dict[str, pd.Series]:
    """
    Builds long-short portfolio return series for each model.

    Args:
        all_predictions: {model_name: DataFrame with [date, permno, ret_excess, me, y_pred]}
        sp500_rets:      Optional S&P 500 excess return series (to add to Figure 9).
        rf_rets:         Optional risk-free rate series (used to compute S&P500-Rf).

    Returns:
        Dictionary {model_name: long-short monthly return Series}
    """
    portfolio_rets = {}

    for model_name, pred_df in all_predictions.items():
        logger.info(f"Forming portfolios for model: {model_name}")
        decile_df = form_decile_portfolios(pred_df)
        ls_rets   = long_short_portfolio(decile_df)
        portfolio_rets[model_name] = ls_rets

    # Add S&P 500 excess return for Figure 9
    if sp500_rets is not None and rf_rets is not None:
        sp500_excess = (sp500_rets - rf_rets).dropna()
        portfolio_rets["SP500-Rf"] = sp500_excess
    elif sp500_rets is not None:
        portfolio_rets["SP500-Rf"] = sp500_rets

    return portfolio_rets
