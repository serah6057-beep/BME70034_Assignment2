"""
src/models.py
-------------
All machine learning models used in the GKX (2020) replication.

Models implemented:
  - OLS+H      : Ordinary Least Squares with Huber loss (robust regression)
  - OLS-3+H    : OLS restricted to 3 factors (Size, B/M, Momentum) + Huber
  - PCR         : Principal Component Regression (grid-searched n_components)
  - ENet+H      : Elastic Net with Huber loss
  - RF          : Random Forest (gradient-boosted ensemble of trees)
  - NN2         : Neural Network with 2 hidden layers (ReLU + batch norm + dropout)
  - NN4         : Neural Network with 4 hidden layers

Each model class implements the interface:
    .fit(X_train, y_train, X_val, y_val)  → sets hyperparameters, trains
    .predict(X)                            → returns predicted excess returns
    .feature_importance(feat_cols)         → returns pd.Series (for Figure 4)
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional

# scikit-learn imports
from sklearn.linear_model import HuberRegressor, ElasticNet
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator

# PyTorch imports (used for NN2, NN4)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.settings import (
    OLS_PARAMS, OLS3_FEATURES, PCR_PARAMS, ENET_PARAMS,
    RF_PARAMS, NN_PARAMS, NN_BATCH_NORM, NN_ENSEMBLE_N, NN_RANDOM_SEED,
    get_logger,
)

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================
# UTILITY: WEIGHTED R² (out-of-sample)
# =============================================================

def oos_r2(y_true: np.ndarray, y_pred: np.ndarray,
           weights: Optional[np.ndarray] = None) -> float:
    """
    Computes out-of-sample R² following GKX (2020) Equation (2):

        R²_oos = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - 0)²

    The benchmark prediction is the historical mean return (set to 0
    since we use demeaned excess returns in the GKX convention).

    Args:
        y_true:   Realized excess returns (N,)
        y_pred:   Model predictions (N,)
        weights:  Optional sample weights (e.g., market equity)

    Returns:
        Scalar OOS R².
    """
    if weights is None:
        weights = np.ones(len(y_true))
    weights = weights / weights.sum()

    ss_res  = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot  = np.sum(weights * y_true ** 2)          # benchmark = 0 (null model)
    return float(1 - ss_res / ss_tot)


# =============================================================
# 1. OLS + HUBER LOSS
# =============================================================

class OLSHuber:
    """
    Ordinary Least Squares with Huber loss for robustness to outliers.

    The "+H" suffix in GKX denotes the Huber loss variant, which down-weights
    observations with large residuals (mitigating the influence of extreme
    stock returns on coefficient estimates).
    """

    def __init__(self, epsilon: float = OLS_PARAMS["huber_epsilon"],
                 feature_subset: Optional[list[str]] = None):
        self.epsilon        = epsilon
        self.feature_subset = feature_subset   # None = use all features
        self.model          = HuberRegressor(
            epsilon=epsilon,
            max_iter=500,
            fit_intercept=True,
        )
        self.feat_cols_: Optional[list[str]] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            feat_cols: Optional[list[str]] = None) -> "OLSHuber":
        """
        Fits the Huber regressor on the training set.
        If feature_subset is specified, uses only those column indices.
        """
        self.feat_cols_ = feat_cols

        # Select feature subset if restricted model (e.g., OLS-3)
        if self.feature_subset and feat_cols:
            idx = [feat_cols.index(f) for f in self.feature_subset
                   if f in feat_cols]
            X_train = X_train[:, idx]
            self.subset_idx_ = idx
        else:
            self.subset_idx_ = None

        logger.debug(f"OLS+H fitting on {X_train.shape} ...")
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted excess returns."""
        if self.subset_idx_ is not None:
            X = X[:, self.subset_idx_]
        return self.model.predict(X)

    def feature_importance(self, feat_cols: list[str]) -> pd.Series:
        """Returns absolute coefficient values as feature importance."""
        coef = self.model.coef_
        if self.subset_idx_ is not None:
            cols = [feat_cols[i] for i in self.subset_idx_]
        else:
            cols = feat_cols[:len(coef)]
        return pd.Series(np.abs(coef), index=cols).sort_values(ascending=False)


# =============================================================
# 2. PRINCIPAL COMPONENT REGRESSION (PCR)
# =============================================================

class PCRModel:
    """
    Principal Component Regression:
      1. Standardize features.
      2. Extract top-K principal components from training set.
      3. Regress returns on the K components (OLS).

    The number of components K is selected by maximizing OOS R² on
    the validation set.
    """

    def __init__(self, n_components_grid: list[int] = PCR_PARAMS["n_components_grid"]):
        self.n_components_grid = n_components_grid
        self.best_n_components_: Optional[int] = None
        self.pipeline_: Optional[Pipeline] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            feat_cols: Optional[list[str]] = None) -> "PCRModel":
        """
        Grid-searches n_components on the validation set.
        Falls back to n_components=10 if no validation data.
        """
        best_r2 = -np.inf
        best_n  = self.n_components_grid[0]

        for n in self.n_components_grid:
            # Clip n to max feasible components
            n = min(n, X_train.shape[1], X_train.shape[0] - 1)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca",    PCA(n_components=n, svd_solver="randomized", random_state=42)),
                ("reg",    HuberRegressor(epsilon=1.345, max_iter=300)),
            ])
            pipe.fit(X_train, y_train)

            if X_val is not None and y_val is not None:
                y_hat = pipe.predict(X_val)
                r2    = oos_r2(y_val, y_hat)
                if r2 > best_r2:
                    best_r2 = r2
                    best_n  = n
                    self.pipeline_ = pipe
            else:
                self.pipeline_ = pipe
                best_n = n
                break

        self.best_n_components_ = best_n
        logger.debug(f"PCR best n_components={best_n} (val R²={best_r2:.4f})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline_.predict(X)

    def feature_importance(self, feat_cols: list[str]) -> pd.Series:
        """
        Feature importance = |PCA loading × regression coefficient|,
        summed across components (approximation following GKX 2020).
        """
        pca   = self.pipeline_.named_steps["pca"]
        reg   = self.pipeline_.named_steps["reg"]
        coef  = reg.coef_                           # shape (n_components,)
        # Weighted contribution of each original feature
        importance = np.abs(pca.components_.T @ coef)  # shape (n_features,)
        n = min(len(importance), len(feat_cols))
        return pd.Series(importance[:n], index=feat_cols[:n]).sort_values(ascending=False)


# =============================================================
# 3. ELASTIC NET + HUBER LOSS (ENet+H)
# =============================================================

class ENetHuber:
    """
    Elastic Net with Huber loss (ENet+H in GKX 2020).

    Elastic net combines L1 (lasso) and L2 (ridge) penalties:
        penalty = α[ρ||β||₁ + (1-ρ)||β||₂²]

    Parameters α (overall strength) and ρ (L1 ratio) are tuned on the
    validation set.
    """

    def __init__(self,
                 l1_ratio_grid: list[float] = ENET_PARAMS["l1_ratio_grid"],
                 alpha_grid:    list[float] = ENET_PARAMS["alpha_grid"],
                 epsilon:       float       = ENET_PARAMS["huber_epsilon"]):
        self.l1_ratio_grid = l1_ratio_grid
        self.alpha_grid    = alpha_grid
        self.epsilon       = epsilon
        self.best_params_: dict = {}
        self.scaler_       = StandardScaler()
        self.model_:       Optional[HuberRegressor] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            feat_cols: Optional[list[str]] = None) -> "ENetHuber":
        """
        Fits elastic-net Huber regressor with grid-searched hyperparameters.

        Note: scikit-learn's HuberRegressor uses L2 penalty only;
        to add L1 we combine it with a coordinate descent step using ElasticNet
        for feature selection, then refit with Huber on the selected features.
        """
        X_train_s = self.scaler_.fit_transform(X_train)
        X_val_s   = self.scaler_.transform(X_val) if X_val is not None else None

        best_r2    = -np.inf
        best_alpha = self.alpha_grid[0]
        best_l1    = self.l1_ratio_grid[0]

        for alpha in self.alpha_grid:
            for l1 in self.l1_ratio_grid:
                # Step 1: ENet for feature selection mask
                enet = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=1000,
                                  fit_intercept=True)
                enet.fit(X_train_s, y_train)
                mask = np.abs(enet.coef_) > 1e-8

                if mask.sum() == 0:
                    continue

                # Step 2: Refit with Huber on selected features
                model = HuberRegressor(epsilon=self.epsilon, max_iter=300)
                model.fit(X_train_s[:, mask], y_train)

                if X_val_s is not None and y_val is not None:
                    y_hat = model.predict(X_val_s[:, mask])
                    r2    = oos_r2(y_val, y_hat)
                    if r2 > best_r2:
                        best_r2    = r2
                        best_alpha = alpha
                        best_l1    = l1
                        self.model_  = model
                        self.mask_   = mask
                else:
                    self.model_ = model
                    self.mask_  = mask
                    break

        self.best_params_ = {"alpha": best_alpha, "l1_ratio": best_l1}
        logger.debug(f"ENet+H best params: {self.best_params_} (val R²={best_r2:.4f})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler_.transform(X)
        return self.model_.predict(X_s[:, self.mask_])

    def feature_importance(self, feat_cols: list[str]) -> pd.Series:
        """
        Importance = |Huber coefficient| restricted to non-zero ENet features.
        """
        coef = self.model_.coef_
        selected_cols = [feat_cols[i] for i in np.where(self.mask_)[0]
                         if i < len(feat_cols)]
        imp = pd.Series(np.abs(coef[:len(selected_cols)]), index=selected_cols)
        return imp.sort_values(ascending=False)


# =============================================================
# 4. RANDOM FOREST (RF)
# =============================================================

class RFModel:
    """
    Random Forest regression as in GKX (2020) Section 2.4.

    Key design choices from the paper:
      - max_features: number of features considered at each split (tuned)
      - min_samples_leaf: minimum leaf size for regularisation (tuned)
      - Feature importance: mean decrease in node impurity (MDI)
    """

    def __init__(self,
                 n_estimators:         int       = RF_PARAMS["n_estimators"],
                 max_features_grid:    list      = RF_PARAMS["max_features_grid"],
                 min_samples_leaf_grid: list[int] = RF_PARAMS["min_samples_leaf_grid"],
                 n_jobs:               int       = RF_PARAMS["n_jobs"],
                 random_state:         int       = RF_PARAMS["random_state"]):
        self.n_estimators         = n_estimators
        self.max_features_grid    = max_features_grid
        self.min_samples_leaf_grid = min_samples_leaf_grid
        self.n_jobs               = n_jobs
        self.random_state         = random_state
        self.model_: Optional[RandomForestRegressor] = None
        self.best_params_: dict = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            feat_cols: Optional[list[str]] = None) -> "RFModel":
        """
        Fits Random Forest; grid-searches max_features and min_samples_leaf.
        """
        best_r2 = -np.inf

        for mf in self.max_features_grid:
            for msl in self.min_samples_leaf_grid:
                # Use a smaller n_estimators for tuning to save time
                rf = RandomForestRegressor(
                    n_estimators=50,          # fast tuning pass
                    max_features=mf,
                    min_samples_leaf=msl,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
                rf.fit(X_train, y_train)

                if X_val is not None and y_val is not None:
                    r2 = oos_r2(y_val, rf.predict(X_val))
                    if r2 > best_r2:
                        best_r2 = r2
                        self.best_params_ = {"max_features": mf, "min_samples_leaf": msl}
                else:
                    self.best_params_ = {"max_features": mf, "min_samples_leaf": msl}
                    break

        logger.debug(f"RF best params: {self.best_params_} (val R²={best_r2:.4f})")

        # Retrain with full n_estimators using best params
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **self.best_params_,
        )
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def feature_importance(self, feat_cols: list[str]) -> pd.Series:
        """
        Feature importance = Mean Decrease in Impurity (MDI) from sklearn.
        Normalized to sum to 1.
        """
        imp = self.model_.feature_importances_
        n   = min(len(imp), len(feat_cols))
        s   = pd.Series(imp[:n], index=feat_cols[:n])
        return (s / s.sum()).sort_values(ascending=False)


# =============================================================
# 5. NEURAL NETWORK (NN2, NN4)  — PyTorch implementation
# =============================================================

class _NNBlock(nn.Module):
    """
    A single hidden layer block: Linear → [BatchNorm] → ReLU → Dropout.
    Follows GKX (2020) architecture (Section 2.5).
    """
    def __init__(self, in_dim: int, out_dim: int,
                 dropout: float = 0.05, use_batch_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _NeuralNet(nn.Module):
    """
    Fully-connected feedforward neural network with variable depth.
    Input layer → hidden blocks → linear output.
    """
    def __init__(self, in_dim: int, hidden_dims: list[int],
                 dropout: float, use_batch_norm: bool):
        super().__init__()
        dims   = [in_dim] + hidden_dims
        blocks = [
            _NNBlock(dims[i], dims[i + 1], dropout=dropout,
                     use_batch_norm=use_batch_norm)
            for i in range(len(dims) - 1)
        ]
        blocks.append(nn.Linear(dims[-1], 1))  # output layer (no activation)
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NNModel:
    """
    Neural network model ensemble for stock return prediction.

    Following GKX (2020):
      - ReLU activations in all hidden layers
      - Batch normalisation for training stability
      - Dropout for regularisation
      - L1 (lasso) penalty on weights
      - Ensemble of NN_ENSEMBLE_N randomly-initialised networks
      - Early stopping on validation loss

    Args:
        model_name: "NN2" or "NN4" (controls network depth)
    """

    def __init__(self, model_name: str = "NN2"):
        assert model_name in NN_PARAMS, f"Unknown model: {model_name}"
        cfg              = NN_PARAMS[model_name]
        self.model_name  = model_name
        self.hidden_dims = cfg["hidden_layers"]
        self.dropout     = cfg["dropout"]
        self.batch_size  = cfg["batch_size"]
        self.lr          = cfg["lr"]
        self.epochs      = cfg["epochs"]
        self.patience    = cfg["patience"]
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ensemble_: list[_NeuralNet] = []

    def _train_single(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        seed: int,
    ) -> _NeuralNet:
        """
        Trains a single network with the given random seed.
        Uses early stopping based on validation MSE.
        """
        torch.manual_seed(seed)
        in_dim  = X_train.shape[1]
        net     = _NeuralNet(in_dim, self.hidden_dims, self.dropout, NN_BATCH_NORM).to(self.device)
        opt     = optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        # Convert to tensors
        def _to_tensor(a):
            return torch.tensor(a, dtype=torch.float32, device=self.device)

        X_tr = _to_tensor(X_train);  y_tr = _to_tensor(y_train)
        X_vl = _to_tensor(X_val);    y_vl = _to_tensor(y_val)

        loader  = DataLoader(TensorDataset(X_tr, y_tr),
                             batch_size=self.batch_size, shuffle=True)

        best_val_loss  = np.inf
        best_state     = None
        patience_count = 0

        for epoch in range(self.epochs):
            net.train()
            for X_b, y_b in loader:
                opt.zero_grad()
                loss = loss_fn(net(X_b), y_b)

                # L1 penalty on all linear weights (lasso regularisation)
                l1 = sum(p.abs().sum() for name, p in net.named_parameters()
                         if "weight" in name)
                (loss + 1e-5 * l1).backward()
                opt.step()

            # Early stopping on validation
            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(X_vl), y_vl).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss  = val_loss
                best_state     = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    logger.debug(f"Early stop at epoch {epoch}")
                    break

        if best_state:
            net.load_state_dict(best_state)
        return net

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            feat_cols: Optional[list[str]] = None) -> "NNModel":
        """
        Trains an ensemble of NN_ENSEMBLE_N networks with different random seeds.
        """
        logger.debug(f"{self.model_name}: training ensemble of {NN_ENSEMBLE_N} ...")
        self.ensemble_ = []

        # Use zero arrays for val if not provided
        if X_val is None:
            X_val = X_train[:100]
            y_val = y_train[:100]

        for i in range(NN_ENSEMBLE_N):
            seed = NN_RANDOM_SEED + i
            net  = self._train_single(X_train, y_train, X_val, y_val, seed=seed)
            self.ensemble_.append(net)
            logger.debug(f"  Ensemble member {i+1}/{NN_ENSEMBLE_N} done")

        # Store input dimension for feature importance
        self.in_dim_    = X_train.shape[1]
        self.feat_cols_ = feat_cols
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the ensemble-averaged prediction (mean across members).
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        preds = []
        for net in self.ensemble_:
            net.eval()
            with torch.no_grad():
                preds.append(net(X_t).cpu().numpy())
        return np.mean(preds, axis=0)

    def feature_importance(self, feat_cols: list[str]) -> pd.Series:
        """
        Feature importance via permutation: for each feature, zero it out
        and measure the drop in in-sample R² (averaged across ensemble members).

        This is computationally expensive; a fast approximation using input
        gradient magnitude is used here.
        """
        if not self.ensemble_:
            return pd.Series(dtype=float)

        # Use gradient-based sensitivity as a proxy for importance
        net    = self.ensemble_[0]
        in_dim = self.in_dim_

        # Random probe inputs (sample from standard normal)
        torch.manual_seed(0)
        probe = torch.randn(1000, in_dim, requires_grad=True, device=self.device)
        out   = net(probe).sum()
        out.backward()

        importance = probe.grad.abs().mean(dim=0).cpu().detach().numpy()
        n          = min(len(importance), len(feat_cols))
        s          = pd.Series(importance[:n], index=feat_cols[:n])
        return (s / s.sum()).sort_values(ascending=False)


# =============================================================
# MODEL REGISTRY
# =============================================================

def get_model(name: str) -> object:
    """
    Factory function: returns a fresh (unfitted) model instance by name.

    Args:
        name: One of ["OLS+H", "OLS-3+H", "PCR", "ENet+H", "RF", "NN2", "NN4"]

    Returns:
        An unfitted model object with .fit() and .predict() methods.
    """
    registry = {
        "OLS+H":   lambda: OLSHuber(feature_subset=None),
        "OLS-3+H": lambda: OLSHuber(feature_subset=OLS3_FEATURES),
        "PCR":     lambda: PCRModel(),
        "ENet+H":  lambda: ENetHuber(),
        "RF":      lambda: RFModel(),
        "NN2":     lambda: NNModel("NN2"),
        "NN4":     lambda: NNModel("NN4"),
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry.keys())}")
    return registry[name]()
