# Assignment 3 — Empirical Asset Pricing via Machine Learning

**Course:** BME.70034: Empirical Asset Pricing (Spring 2026)  
**Instructor:** Sean Shin  
**Paper:** Gu, Kelly, and Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS 33(5)

---

## Overview

This project replicates key results from Gu, Kelly, and Xiu (2020):

| Output      | Description                                                  | Models                                      |
|-------------|--------------------------------------------------------------|---------------------------------------------|
| **Table 1** | Monthly OOS stock-level prediction performance               | OLS+H, OLS-3+H, PCR, ENet+H, RF, NN2, NN4 |
| **Figure 4**| Variable importance by model                                 | PCR, ENet+H, RF, NN2, NN4                  |
| **Figure 9**| Cumulative return of ML portfolios + S&P 500 benchmark       | OLS-3+H, PCR, ENet+H, RF, NN2, NN4         |

Results are produced for **two sample periods**:
- 1971–2016 (original paper window)
- 1971–2025 (extended window)

---

## Project Structure

```
assignment3/
├── main.py                        # Central pipeline controller
├── requirements.txt               # Python dependencies
├── .gitignore                     # Excludes raw data (CRSP, Compustat)
├── src/
│   ├── settings.py                # Config: paths, hyperparameters, logging
│   ├── data_download.py           # Downloads CRSP, characteristics, macro data
│   ├── data_processing.py         # Cleaning, rank-scaling, rolling splits
│   ├── models.py                  # OLS, PCR, ENet, RF, NN2, NN4
│   ├── rolling_regression.py      # Orchestrates OOS prediction loop
│   ├── portfolio_construction.py  # Decile portfolios, Sharpe, drawdown
│   ├── grs_test.py                # GRS F-test and alpha t-statistics
│   └── outputs.py                 # Table 1, Figure 4, Figure 9 generation
├── data/                          # Raw data (git-ignored — local only)
└── results/
    ├── figures/                   # PDF + PNG figures
    ├── tables/                    # LaTeX + CSV tables
    └── logs/
        └── run.log                # Full execution log
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd assignment3
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure WRDS access

CRSP data is accessed via the WRDS Python API. Create a `~/.pgpass` file or
set environment variables with your WRDS credentials:

```bash
export WRDS_USERNAME=your_username
```

On first run, the WRDS library will prompt for your password and cache it.

> **Note:** Raw CRSP/Compustat data files are stored in `data/` which is  
> listed in `.gitignore` and **never pushed to GitHub**.

---

## Running the Pipeline

### Full run (both sample periods):

```bash
python main.py
```

### Single period:

```bash
python main.py --period 1971_2016
python main.py --period 1971_2025
```

### Skip data download (use cached data):

```bash
python main.py --skip-download
```

### Run specific models only:

```bash
python main.py --models RF NN2 NN4
```

---

## Architecture

The pipeline follows a modular layer separation:

```
Data Layer:          data_download.py  →  data_processing.py
                          ↓
Model Layer:         models.py  (OLS+H, PCR, ENet+H, RF, NN2, NN4)
                          ↓
Prediction Loop:     rolling_regression.py  (expanding-window OOS)
                          ↓
Portfolio Layer:     portfolio_construction.py
                          ↓
Output Layer:        outputs.py  (Table 1, Figure 4, Figure 9)
```

All steps are controlled by `main.py`. Hyperparameters and paths are
centralized in `src/settings.py`.

---

## Algorithm Flowchart

```
Data → Cleaning → Features → Rolling OOS → Portfolio → Output
 ↓         ↓          ↓          ↓              ↓          ↓
CRSP    NYSE only   Rank-      Train/Val/    Decile    Table 1
Chars   common      scale      Test split   L-S port  Figure 4
Macro   stocks      Macro      Fit model    Sharpe    Figure 9
                   interact    Predict      Cum ret
```

**Step-by-step:**

| Step | Input | Process | Output |
|------|-------|---------|--------|
| 1 | CRSP raw | Filter NYSE common stocks | Clean returns panel |
| 2 | Returns + Chars + Macro | Rank-scale, merge, macro interactions | Feature matrix Z_it |
| 3 | Feature matrix | Rolling expanding-window train/val/test split | Splits per year |
| 4 | Training split | Fit model (grid search on val set) | Fitted model |
| 5 | Test split | Out-of-sample prediction | ŷ_it per stock-month |
| 6 | Predictions + returns | Form value-weighted decile portfolios | Monthly L-S returns |
| 7 | L-S returns | Compute OOS R², Sharpe, alpha | Table 1 statistics |
| 8 | L-S returns | Cumulative compounding | Figure 9 |
| 9 | Model weights | Feature importance extraction | Figure 4 |

---

## Data Sources

| Data | Source | Access |
|------|--------|--------|
| CRSP monthly returns | WRDS | Requires subscription |
| Stock characteristics | [Dacheng Xiu's website](https://dachxiu.chicagobooth.edu) | Free download |
| Macro predictors | [Welch-Goyal (2008)](http://www.hec.unil.ch/agoyal/) | Free download |
| Risk-free rate | [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Free download |

---

## Key Assumptions

1. **Sample:** NYSE common stocks (shrcd ∈ {10, 11}, exchcd = 1), 1971–2025.
2. **Missing values:** Filled with 0 after rank-scaling (GKX 2020, Section 2.3).
3. **Portfolio weighting:** Value-weighted by beginning-of-month market equity.
4. **OOS R² benchmark:** Null model = mean excess return of 0 (GKX Eq. 2).
5. **Macro interactions:** Each of 14 macro vars × P characteristics = full feature set.
6. **Validation window:** 12 years prior to test year (for hyperparameter tuning).

---

## Output Files

All outputs are saved to `results/`:

```
results/
├── figures/
│   ├── table1_figure_1971_2016.pdf
│   ├── table1_figure_1971_2025.pdf
│   ├── figure4_1971_2016.pdf
│   ├── figure4_1971_2025.pdf
│   ├── figure9_1971_2016.pdf
│   └── figure9_1971_2025.pdf
├── tables/
│   ├── table1_1971_2016.tex
│   ├── table1_1971_2016.csv
│   ├── table1_1971_2025.tex
│   └── table1_1971_2025.csv
└── logs/
    └── run.log
```
