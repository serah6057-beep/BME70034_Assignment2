"""
src/outputs.py
--------------
Generates all final outputs for the GKX (2020) replication:
  - Table 1 : Monthly OOS stock-level prediction performance
  - Figure (Table 1 visualized as bar chart)
  - Figure 4: Variable importance by model
  - Figure 9: Cumulative return of ML portfolios

All outputs are saved to results/figures/ and results/tables/.
No manual editing of figures or numbers — everything is directly generated
from model predictions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import YearLocator, DateFormatter
from pathlib import Path

from src.settings import (
    FIGURES_DIR, TABLES_DIR, MODELS_TABLE1, MODELS_FIG4, MODELS_FIG9,
    DPI, get_logger,
)
from src.portfolio_construction import (
    compute_performance_stats, cumulative_return, build_portfolio_returns,
    form_decile_portfolios, long_short_portfolio,
)
from src.rolling_regression import compute_oos_r2_table

logger = get_logger(__name__)

# ---- Colour palette (consistent across all figures) ----
MODEL_COLORS = {
    "OLS+H":   "#1f77b4",
    "OLS-3+H": "#aec7e8",
    "PCR":     "#ff7f0e",
    "ENet+H":  "#ffbb78",
    "RF":      "#2ca02c",
    "NN2":     "#d62728",
    "NN4":     "#9467bd",
    "SP500-Rf":"#8c564b",
}


# =============================================================
# TABLE 1 : OOS PREDICTION PERFORMANCE
# =============================================================

def generate_table1(
    all_predictions: dict[str, pd.DataFrame],
    period_label:    str,
) -> pd.DataFrame:
    """
    Generates Table 1: Monthly Out-of-Sample Stock-Level Prediction Performance.

    Columns:
      OOS R²  : Aggregate out-of-sample R² (%)
      Mean Ret: Annualized mean long-short return (%)
      Std     : Annualized std dev (%)
      Sharpe  : Annualized Sharpe ratio
      t-stat  : t-stat of mean return
      Max DD  : Maximum drawdown (%)

    Args:
        all_predictions: {model_name: DataFrame[date, ret_excess, y_pred, me]}
        period_label:    e.g., "1971_2016" — used in filenames.

    Returns:
        Table 1 as a DataFrame.
    """
    logger.info(f"Generating Table 1 for period {period_label}...")

    # OOS R²
    r2_df = compute_oos_r2_table(all_predictions)

    # Portfolio statistics
    port_stats_rows = []
    for model_name in MODELS_TABLE1:
        if model_name not in all_predictions:
            continue
        pred_df   = all_predictions[model_name]
        decile_df = form_decile_portfolios(pred_df)
        ls_rets   = long_short_portfolio(decile_df)
        stats     = compute_performance_stats(ls_rets)
        stats["Model"] = model_name
        port_stats_rows.append(stats)

    port_df = pd.DataFrame(port_stats_rows).set_index("Model")

    # Combine
    table1 = r2_df.join(port_df, how="outer")
    table1 = table1.rename(columns={
        "OOS_R2 (%)": "OOS R² (%)",
        "mean_ret":   "Mean Ret (%)",
        "std_ret":    "Std (%)",
        "sharpe":     "Sharpe",
        "t_stat":     "t-stat",
        "max_dd":     "Max DD (%)",
        "skew":       "Skew",
    })

    # Save LaTeX table
    latex_path = TABLES_DIR / f"table1_{period_label}.tex"
    table1.round(3).to_latex(str(latex_path), float_format="%.3f")
    logger.info(f"Table 1 saved: {latex_path}")

    # Save CSV
    csv_path = TABLES_DIR / f"table1_{period_label}.csv"
    table1.round(3).to_csv(str(csv_path))
    logger.info(f"Table 1 (CSV) saved: {csv_path}")

    return table1


def generate_table1_figure(
    table1: pd.DataFrame,
    period_label: str,
) -> None:
    """
    Visualizes Table 1 as a grouped bar chart of OOS R² and Sharpe ratio,
    following the GKX paper's Figure that accompanies Table 1.
    """
    logger.info("Generating Table 1 figure...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Table 1: Monthly OOS Prediction Performance  [{period_label.replace('_', '–')}]",
        fontsize=14, fontweight="bold", y=1.02,
    )

    models = [m for m in MODELS_TABLE1 if m in table1.index]
    colors = [MODEL_COLORS.get(m, "gray") for m in models]
    x      = np.arange(len(models))

    # ---- Panel A: OOS R² ----
    ax1 = axes[0]
    r2_vals = table1.loc[models, "OOS R² (%)"].values
    bars = ax1.bar(x, r2_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax1.set_ylabel("Out-of-Sample R² (%)", fontsize=11)
    ax1.set_title("Panel A: Predictive R²", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)
    # Annotate bars
    for bar, val in zip(bars, r2_vals):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001,
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=8)

    # ---- Panel B: Sharpe Ratio ----
    ax2 = axes[1]
    sh_vals = table1.loc[models, "Sharpe"].values
    bars2 = ax2.bar(x, sh_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax2.set_ylabel("Annualized Sharpe Ratio", fontsize=11)
    ax2.set_title("Panel B: Sharpe Ratio (Long-Short)", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, sh_vals):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_figure(fig, f"table1_figure_{period_label}")
    plt.close(fig)


# =============================================================
# FIGURE 4 : VARIABLE IMPORTANCE
# =============================================================

def generate_figure4(
    feature_importances: dict[str, list],
    feat_cols: list[str],
    period_label: str,
    top_n: int = 20,
) -> None:
    """
    Generates Figure 4: Variable Importance by Model.

    For each model in MODELS_FIG4, shows the top-N most important features
    averaged across all rolling windows.

    Args:
        feature_importances: {model_name: [pd.Series per test year]}
        feat_cols:           List of all feature names.
        period_label:        For filename.
        top_n:               Number of top features to display.
    """
    logger.info(f"Generating Figure 4 for period {period_label}...")

    # Compute average importance across rolling windows for each model
    avg_importance: dict[str, pd.Series] = {}
    for model_name in MODELS_FIG4:
        fi_list = feature_importances.get(model_name, [])
        if not fi_list:
            logger.warning(f"No feature importances for {model_name}")
            continue
        fi_df  = pd.DataFrame(fi_list).fillna(0)
        avg_fi = fi_df.mean(axis=0).sort_values(ascending=False)
        # Normalize to sum to 100%
        avg_fi = avg_fi / avg_fi.sum() * 100
        avg_importance[model_name] = avg_fi.head(top_n)

    if not avg_importance:
        logger.warning("No feature importances to plot for Figure 4.")
        return

    n_models = len(avg_importance)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5 * n_models, 7),
        sharey=False,
    )
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        f"Figure 4: Variable Importance by Model  [{period_label.replace('_', '–')}]",
        fontsize=14, fontweight="bold",
    )

    for ax, (model_name, fi) in zip(axes, avg_importance.items()):
        color = MODEL_COLORS.get(model_name, "steelblue")
        ax.barh(
            fi.index[::-1],
            fi.values[::-1],
            color=color, edgecolor="black", linewidth=0.4,
        )
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance (%)", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    save_figure(fig, f"figure4_{period_label}")
    plt.close(fig)


# =============================================================
# FIGURE 9 : CUMULATIVE RETURN OF ML PORTFOLIOS
# =============================================================

def generate_figure9(
    all_predictions:  dict[str, pd.DataFrame],
    sp500_rets:       pd.Series,
    rf_rets:          pd.Series,
    period_label:     str,
    nber_recessions:  list[tuple] = None,
) -> None:
    """
    Generates Figure 9: Cumulative Return of Machine Learning Portfolios.
    """
    logger.info(f"Generating Figure 9 for period {period_label}...")

    # ---- Build long-short portfolio returns per model ----
    port_rets = {}
    for model_name in MODELS_FIG9:
        if model_name not in all_predictions:
            continue
        pred_df   = all_predictions[model_name]
        decile_df = form_decile_portfolios(pred_df)
        ls_rets   = long_short_portfolio(decile_df)
        port_rets[model_name] = ls_rets

    if not port_rets:
        logger.warning("No model portfolios available for Figure 9")
        return

    # ---- Determine the OOS date range from model predictions ----
    all_dates = set()
    for s in port_rets.values():
        all_dates.update(s.index)
    oos_start = min(all_dates)
    oos_end   = max(all_dates)
    logger.info(f"  OOS window: {oos_start.date()} → {oos_end.date()}")

    # ---- Add SP500-Rf benchmark, restricted to OOS window ----
    if not sp500_rets.empty and not rf_rets.empty:
        common = rf_rets.index.intersection(sp500_rets.index)
        sp500_excess = (sp500_rets.loc[common] - rf_rets.loc[common])
        sp500_excess = sp500_excess[
            (sp500_excess.index >= oos_start) & (sp500_excess.index <= oos_end)
        ]
        if len(sp500_excess) > 0:
            port_rets["SP500-Rf"] = sp500_excess
            logger.info(f"  SP500-Rf benchmark added: {len(sp500_excess)} months")
        else:
            logger.warning("  SP500-Rf has no overlap with OOS window")
    else:
        logger.warning("  SP500 or RF data missing; skipping SP500-Rf")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14, 7))

    for model_name, ret_series in port_rets.items():
        # Filter to OOS window only (for SP500-Rf too)
        s = ret_series[(ret_series.index >= oos_start) & (ret_series.index <= oos_end)]
        cum_ret = np.log(cumulative_return(s))
        color   = MODEL_COLORS.get(model_name, "black")
        ls      = "--" if model_name == "SP500-Rf" else "-"
        lw      = 2.0  if model_name == "SP500-Rf" else 1.4
        ax.plot(
            cum_ret.index, cum_ret.values,
            label=model_name, color=color, linestyle=ls, linewidth=lw,
        )

    # ---- NBER recession shading (only inside OOS window) ----
    if nber_recessions:
        for rec_start, rec_end in nber_recessions:
            rs = pd.Timestamp(rec_start)
            re = pd.Timestamp(rec_end)
            # Only shade if recession overlaps OOS window
            if re >= oos_start and rs <= oos_end:
                ax.axvspan(
                    max(rs, oos_start), min(re, oos_end),
                    color="gray", alpha=0.20, zorder=0,
                )

    # ---- Axis cosmetics ----
    ax.set_title(
        f"Figure 9: Cumulative Return of ML Portfolios  [{period_label.replace('_', '–')}]",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Log Cumulative Excess Return", fontsize=12)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

    # Tight x-axis to OOS window
    ax.set_xlim(oos_start, oos_end)

    # Yearly ticks
    ax.xaxis.set_major_locator(YearLocator(1))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.xticks(rotation=30)

    plt.tight_layout()
    save_figure(fig, f"figure9_{period_label}")
    plt.close(fig)


# =============================================================
# NBER RECESSION DATES (for Figure 9 shading)
# =============================================================

# NBER recession dates (peak → trough): https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
NBER_RECESSIONS = [
    ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]


# =============================================================
# UTILITY: SAVE FIGURE
# =============================================================

def save_figure(fig, name: str) -> None:
    """
    Saves a figure as both PDF and PNG to the figures directory.

    Args:
        fig:  matplotlib Figure object.
        name: Base filename (no extension).
    """
    for ext in ["pdf", "png"]:
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
        logger.info(f"Figure saved: {path}")


# =============================================================
# MASTER OUTPUT FUNCTION
# =============================================================

def generate_all_outputs(
    all_predictions:     dict[str, pd.DataFrame],
    feature_importances: dict[str, list],
    feat_cols:           list[str],
    sp500_rets:          pd.Series,
    rf_rets:             pd.Series,
    period_label:        str,
) -> pd.DataFrame:
    """
    Runs all output generation functions for one sample period.

    Args:
        all_predictions:     Prediction DataFrames per model.
        feature_importances: Feature importance lists per model.
        feat_cols:           Feature column names.
        sp500_rets:          S&P 500 returns.
        rf_rets:             Risk-free rates.
        period_label:        E.g., "1971_2016".

    Returns:
        Table 1 DataFrame.
    """
    logger.info(f"Generating all outputs for period: {period_label}")

    # 1. Table 1 + accompanying bar chart
    table1 = generate_table1(all_predictions, period_label)
    generate_table1_figure(table1, period_label)

    # 2. Figure 4: Variable importance
    generate_figure4(feature_importances, feat_cols, period_label)

    # 3. Figure 9: Cumulative returns
    generate_figure9(
        all_predictions=all_predictions,
        sp500_rets=sp500_rets,
        rf_rets=rf_rets,
        period_label=period_label,
        nber_recessions=NBER_RECESSIONS,
    )

    logger.info(f"All outputs saved to {FIGURES_DIR} and {TABLES_DIR}")
    return table1
