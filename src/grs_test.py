"""
src/grs_test.py
---------------
Gibbons, Ross, and Shanken (1989) GRS test for evaluating whether ML
model portfolios generate alphas that are jointly zero.

This is referenced in GKX (2020) Table 1 (t-statistics for portfolio returns).

Note: The GRS test itself is not the primary output of the assignment,
but it is used as a robustness check and provides the t-statistics
reported in Table 1.
"""

import numpy as np
import pandas as pd
from scipy import stats
from src.settings import get_logger

logger = get_logger(__name__)


def grs_test(
    portfolio_rets: pd.DataFrame,
    factor_rets:    pd.DataFrame,
) -> dict:
    """
    Computes the Gibbons-Ross-Shanken (1989) GRS F-statistic to test
    whether a set of portfolio alphas are jointly zero.

    H₀: α₁ = α₂ = ... = αN = 0 (no risk-adjusted abnormal returns)

    Formula (GRS 1989, Equation 9):
        GRS = (T-N-K)/N × (1 + μ̂'Ω̂⁻¹μ̂)⁻¹ × α̂'Σ̂⁻¹α̂

    where:
        T = number of time periods
        N = number of portfolios
        K = number of factors
        μ̂ = sample mean of factor returns
        Ω̂ = sample covariance of factor returns
        α̂ = estimated alphas (intercepts from time-series regressions)
        Σ̂ = covariance matrix of regression residuals

    Args:
        portfolio_rets: DataFrame (T × N) of portfolio excess returns.
        factor_rets:    DataFrame (T × K) of factor excess returns.

    Returns:
        dict with keys:
          grs_stat:  GRS F-statistic
          p_value:   p-value from F(N, T-N-K) distribution
          alphas:    Series of estimated alphas per portfolio
          t_stats:   Series of t-statistics for each alpha
    """
    # Align on common dates
    common_idx = portfolio_rets.index.intersection(factor_rets.index)
    R = portfolio_rets.loc[common_idx].values   # (T, N)
    F = factor_rets.loc[common_idx].values      # (T, K)

    T, N = R.shape
    K    = F.shape[1]

    # Add intercept column to factor matrix
    F_aug = np.hstack([np.ones((T, 1)), F])    # (T, K+1)

    # OLS: estimate alpha and beta for each portfolio
    beta_mat, _, _, _ = np.linalg.lstsq(F_aug, R, rcond=None)  # (K+1, N)
    alphas   = beta_mat[0, :]                                   # (N,)
    residuals = R - F_aug @ beta_mat                            # (T, N)

    # Residual covariance matrix Σ̂
    Sigma = residuals.T @ residuals / (T - K - 1)               # (N, N)

    # Factor mean and covariance
    mu_f  = F.mean(axis=0)                                      # (K,)
    Omega = np.cov(F, rowvar=False, ddof=1)                     # (K, K)
    if K == 1:
        Omega = Omega.reshape(1, 1)

    # Sharpe ratio squared of tangency portfolio (factor)
    sh2_f = float(mu_f @ np.linalg.solve(Omega, mu_f))

    # GRS statistic
    Sigma_inv = np.linalg.pinv(Sigma)
    alpha_quad = float(alphas @ Sigma_inv @ alphas)
    grs_stat   = (T - N - K) / N * alpha_quad / (1 + sh2_f)

    # p-value from F distribution
    p_value = 1 - stats.f.cdf(grs_stat, dfn=N, dfd=T - N - K)

    # Individual t-statistics for each alpha
    se_alpha = np.sqrt(np.diag(Sigma) / T * (1 + sh2_f))
    t_stats  = pd.Series(alphas / se_alpha,
                         index=portfolio_rets.columns,
                         name="t_stat")
    alpha_s  = pd.Series(alphas,
                         index=portfolio_rets.columns,
                         name="alpha")

    logger.info(f"GRS F-statistic: {grs_stat:.4f}, p-value: {p_value:.4f}")
    return {
        "grs_stat": grs_stat,
        "p_value":  p_value,
        "alphas":   alpha_s,
        "t_stats":  t_stats,
    }


def portfolio_alpha_tstat(
    port_ret: pd.Series,
    factor_rets: pd.DataFrame,
) -> tuple[float, float]:
    """
    Regresses a single portfolio return on factors and returns
    (alpha, t-stat of alpha).

    Useful for Table 1 where we report alpha t-stats for each model's
    long-short portfolio.

    Args:
        port_ret:    Monthly long-short portfolio excess returns.
        factor_rets: Factor excess returns (e.g., FF5 factors).

    Returns:
        (alpha, t_stat) — both scalars.
    """
    common = port_ret.index.intersection(factor_rets.index)
    y = port_ret.loc[common].values
    X = factor_rets.loc[common].values
    T = len(y)

    X_aug = np.hstack([np.ones((T, 1)), X])
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    resid = y - X_aug @ beta

    # Standard error via OLS formula
    s2    = np.sum(resid**2) / (T - X_aug.shape[1])
    cov_b = s2 * np.linalg.pinv(X_aug.T @ X_aug)
    se_alpha = np.sqrt(cov_b[0, 0])

    alpha = beta[0]
    t_stat = alpha / se_alpha if se_alpha > 0 else np.nan
    return float(alpha * 12 * 100), float(t_stat)  # annualized alpha in %
