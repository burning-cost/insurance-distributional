"""
NeuralGaussianMixture: energy score-guided neural Gaussian mixture model.

Implements the NE-GMM framework from Yang, Ji, Li & Deng (2026) arXiv:2603.27672.
The core idea: an MLP maps risk features X to per-observation mixture weights,
means, and variances of a K-component Gaussian mixture. Training minimises a
convex combination of negative log-likelihood (NLL) and analytic energy score (ES).

Why this matters for insurance pricing:
  A Gamma GBM enforces a fixed distributional shape: variance is proportional to
  mu^2. That's fine for homogeneous severity, but motor claims in London vs rural
  Scotland don't share that constraint. NeuralGMM learns the full conditional shape
  per risk — heavier tails where claims are more variable, lighter where they are
  not — without imposing a parametric family on the shape.

  The energy score component matters because NLL is insensitive to the spread of
  the distribution near the mode. A mixture could assign low probability mass to
  the actual observed value while still minimising NLL via a sharp spike nearby.
  The energy score penalises distributional spread directly, yielding better
  calibrated uncertainty estimates for XL pricing and risk capital.

Analytic energy score formula (O(K^2) in components, O(n) in observations):
  ES(F, y) = sum_m pi_m * A_m(y) - 0.5 * sum_m sum_l pi_m * pi_l * B_ml

  A_m(y) = (mu_m - y) * (2*Phi(z_m) - 1) + 2*sigma_m * phi(z_m)
    where z_m = (mu_m - y) / sigma_m

  B_ml uses d = mu_m - mu_l, sv = sqrt(sigma2_m + sigma2_l):
    same formula with mean d and variance sv^2

This avoids Monte Carlo at training time — crucial for the Pi's CPU constraints.

References:
    Yang, Ji, Li & Deng (2026). Energy Score-Guided Neural Gaussian Mixture Model
    for Predictive Uncertainty Quantification. arXiv:2603.27672.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

    # Provide stub so class definitions using nn.Module don't raise NameError
    # at import time when torch is absent.
    class _NNStub:
        class Module:
            pass

        class Sequential:
            pass

        class Linear:
            pass

        class ReLU:
            pass

        class Softmax:
            pass

        class utils:
            @staticmethod
            def clip_grad_norm_(*args, **kwargs):
                pass

    nn = _NNStub()  # type: ignore[assignment]


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for NeuralGaussianMixture. "
            "Install with: pip install insurance-distributional[neural]"
        )


# ---------------------------------------------------------------------------
# Neural network backbone
# ---------------------------------------------------------------------------


class _GMMNetwork(nn.Module):  # type: ignore[misc]
    """MLP backbone with three parallel output heads for K Gaussian components.

    Architecture:
        Input (n_features) -> Linear+ReLU x n_layers -> hidden_size
        Then three separate linear heads:
          - weights head:  hidden_size -> K  (softmax -> pi_k)
          - means head:    hidden_size -> K  (identity -> mu_k)
          - vars head:     hidden_size -> K  (softplus + var_eps -> sigma2_k)

    Keeping the heads separate rather than concatenated avoids weight sharing
    between very different activation functions (softmax vs identity vs softplus).
    """

    def __init__(
        self,
        n_features: int,
        n_components: int,
        hidden_size: int,
        n_layers: int,
        var_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.var_eps = var_eps

        # Shared backbone
        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        self.backbone = nn.Sequential(*layers)

        # Three output heads
        self.weights_head = nn.Linear(hidden_size, n_components)
        self.means_head = nn.Linear(hidden_size, n_components)
        self.vars_head = nn.Linear(hidden_size, n_components)

    def forward(
        self, x: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Args:
            x: (batch_size, n_features)
        Returns:
            pi:     (batch_size, K)  mixture weights (sum to 1)
            mu:     (batch_size, K)  component means
            sigma2: (batch_size, K)  component variances (positive)
        """
        h = self.backbone(x)
        pi = torch.softmax(self.weights_head(h), dim=1)
        mu = self.means_head(h)
        sigma2 = torch.nn.functional.softplus(self.vars_head(h)) + self.var_eps
        return pi, mu, sigma2


# ---------------------------------------------------------------------------
# Loss functions (analytic, no MC at training time)
# ---------------------------------------------------------------------------


def _nll_loss(
    pi: "torch.Tensor",
    mu: "torch.Tensor",
    sigma2: "torch.Tensor",
    y: "torch.Tensor",
) -> "torch.Tensor":
    """Mixture NLL via log-sum-exp trick for numerical stability.

    log p(y|x) = logsumexp_k [ log(pi_k) + Normal(mu_k, sigma2_k).log_prob(y) ]

    Args:
        pi:     (n, K)
        mu:     (n, K)
        sigma2: (n, K)
        y:      (n,)
    Returns:
        scalar mean NLL
    """
    y_col = y.unsqueeze(1)  # (n, 1)
    log_pi = torch.log(pi + 1e-40)
    # Normal log_prob: -0.5*log(2*pi*sigma2) - 0.5*(y-mu)^2/sigma2
    log_comp = (
        -0.5 * torch.log(2.0 * torch.pi * sigma2)
        - 0.5 * (y_col - mu) ** 2 / sigma2
    )
    log_probs = log_pi + log_comp  # (n, K)
    log_mixture = torch.logsumexp(log_probs, dim=1)  # (n,)
    return -log_mixture.mean()


def _energy_score_loss(
    pi: "torch.Tensor",
    mu: "torch.Tensor",
    sigma2: "torch.Tensor",
    y: "torch.Tensor",
) -> "torch.Tensor":
    """Analytic energy score for a Gaussian mixture (O(K^2) formula).

    ES(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]
             = sum_m pi_m * A_m(y) - 0.5 * sum_m sum_l pi_m * pi_l * B_ml

    where for a single Gaussian component N(mu_m, sigma2_m):
      A_m(y) = |mu_m - y| * (2*Phi(z_m) - 1) + 2*sigma_m * phi(z_m)
               where z_m = (mu_m - y) / sigma_m

    and B_ml (cross-component term) uses the same formula with:
      d = mu_m - mu_l,  sv = sqrt(sigma2_m + sigma2_l)

    This is exact and requires no sampling. See Yang et al. (2026) eq. (5-7).

    Args:
        pi:     (n, K)
        mu:     (n, K)
        sigma2: (n, K)
        y:      (n,)
    Returns:
        scalar mean energy score
    """
    y_col = y.unsqueeze(1)  # (n, 1)
    sigma = torch.sqrt(sigma2)  # (n, K)

    # --- A_m(y): E[ |X_m - y| ] for component m ---
    z = (mu - y_col) / sigma  # (n, K)
    normal = torch.distributions.Normal(0.0, 1.0)
    Phi_z = normal.cdf(z)          # (n, K)
    phi_z = torch.exp(normal.log_prob(z))  # (n, K)
    A = (mu - y_col) * (2.0 * Phi_z - 1.0) + 2.0 * sigma * phi_z  # (n, K)
    term1 = (pi * A).sum(dim=1)  # (n,)

    # --- B_ml: E[ |X_m - X_l| ] cross-component term ---
    # Expand to (n, K, K) for all pairs
    mu_row = mu.unsqueeze(2)      # (n, K, 1)
    mu_col = mu.unsqueeze(1)      # (n, 1, K)
    s2_row = sigma2.unsqueeze(2)  # (n, K, 1)
    s2_col = sigma2.unsqueeze(1)  # (n, 1, K)

    d = mu_row - mu_col                     # (n, K, K)
    sv = torch.sqrt(s2_row + s2_col)        # (n, K, K)
    z_ml = d / sv                           # (n, K, K)
    Phi_ml = normal.cdf(z_ml)
    phi_ml = torch.exp(normal.log_prob(z_ml))
    B = d * (2.0 * Phi_ml - 1.0) + 2.0 * sv * phi_ml  # (n, K, K)

    pi_row = pi.unsqueeze(2)  # (n, K, 1)
    pi_col = pi.unsqueeze(1)  # (n, 1, K)
    term2 = 0.5 * (pi_row * pi_col * B).sum(dim=(1, 2))  # (n,)

    es = term1 - term2  # (n,)
    return es.mean()


# ---------------------------------------------------------------------------
# Prediction container
# ---------------------------------------------------------------------------


@dataclass
class GMMPrediction:
    """
    Container for NeuralGaussianMixture predictions.

    Holds the per-observation K-component Gaussian mixture parameters and provides
    derived actuarial quantities: mean, variance, CoV, quantiles, samples, and
    layer pricing.

    Attributes
    ----------
    weights : np.ndarray, shape (n, K)
        Mixture weights pi_k(x). Each row sums to 1.
    means : np.ndarray, shape (n, K)
        Component means mu_k(x). If log_transform was used at fit time, these
        are in log-space — the mean/variance properties account for this.
    vars : np.ndarray, shape (n, K)
        Component variances sigma2_k(x). Strictly positive.
    log_transform : bool
        Whether the model was trained on log(y). If True, samples are
        exponentiated before quantile/layer calculations.
    """

    weights: np.ndarray   # (n, K)
    means: np.ndarray     # (n, K)
    vars: np.ndarray      # (n, K)
    log_transform: bool = False

    @property
    def mean(self) -> np.ndarray:
        """Conditional mean E[Y|X] = sum_k pi_k * mu_k.

        In log-space: if log_transform=True, returns E[exp(T)] via lognormal
        identity: E[exp(mu_k + 0.5*sigma2_k)].
        """
        if self.log_transform:
            # E[Y] = sum_k pi_k * exp(mu_k + 0.5 * sigma2_k)
            return (self.weights * np.exp(self.means + 0.5 * self.vars)).sum(axis=1)
        return (self.weights * self.means).sum(axis=1)

    @property
    def variance(self) -> np.ndarray:
        """Conditional variance Var[Y|X].

        Law of total variance: Var[Y] = E[Var[Y|K]] + Var[E[Y|K]]
          = sum_k pi_k * sigma2_k + sum_k pi_k * (mu_k - mu_mix)^2

        In log-space: uses lognormal second moment.
        """
        if self.log_transform:
            # E[Y^2] = sum_k pi_k * exp(2*mu_k + 2*sigma2_k)
            e2 = (self.weights * np.exp(2.0 * self.means + 2.0 * self.vars)).sum(axis=1)
            e1 = self.mean
            return np.maximum(e2 - e1 ** 2, 0.0)
        mu_mix = self.mean[:, None]  # (n, 1)
        within = (self.weights * self.vars).sum(axis=1)
        between = (self.weights * (self.means - mu_mix) ** 2).sum(axis=1)
        return within + between

    @property
    def std(self) -> np.ndarray:
        """Conditional standard deviation sqrt(Var[Y|X])."""
        return np.sqrt(self.variance)

    @property
    def cov(self) -> np.ndarray:
        """Coefficient of variation SD / mean. Dimensionless risk volatility."""
        return self.std / (self.mean + 1e-12)

    def volatility_score(self) -> np.ndarray:
        """
        Dimensionless per-risk volatility ranking: CoV = SD / mean.

        The key actuarial output of distributional modelling: identifies which
        risks are intrinsically more volatile beyond what their expected loss
        implies. Use for safety loading calibration, underwriter referrals,
        IFRS 17 risk adjustment, and XL attachment optimisation.
        """
        return self.cov

    def sample(self, n_samples: int = 2000, seed: int = 42) -> np.ndarray:
        """
        Draw samples from the fitted Gaussian mixture per observation.

        For each observation:
          1. Sample component index k ~ Categorical(pi_k)
          2. Sample y ~ Normal(mu_k, sigma2_k)

        If log_transform=True, samples are exponentiated.

        Parameters
        ----------
        n_samples : int
            Samples per observation.
        seed : int
            Random seed.

        Returns
        -------
        np.ndarray, shape (n, n_samples)
        """
        rng = np.random.default_rng(seed)
        n = len(self.weights)
        K = self.weights.shape[1]

        # Sample component indices: (n, n_samples) integer array
        # Cumulative weights for vectorised search
        cum_w = np.cumsum(self.weights, axis=1)  # (n, K)
        u = rng.random((n, n_samples))           # (n, n_samples)
        # For each obs i and sample j, find first k where cum_w[i,k] >= u[i,j]
        k_idx = (u[:, :, None] > cum_w[:, None, :]).sum(axis=2)  # (n, n_samples)
        k_idx = np.clip(k_idx, 0, K - 1)

        # Gather component parameters per sampled index
        mu_sel = np.take_along_axis(self.means, k_idx, axis=1)     # (n, n_samples)
        sigma2_sel = np.take_along_axis(self.vars, k_idx, axis=1)  # (n, n_samples)

        draws = rng.normal(mu_sel, np.sqrt(sigma2_sel))  # (n, n_samples)
        if self.log_transform:
            draws = np.exp(draws)
        return draws

    def quantile(self, q: float, n_samples: int = 2000, seed: int = 42) -> np.ndarray:
        """
        Monte Carlo quantile estimate from the fitted mixture.

        Parameters
        ----------
        q : float
            Quantile level, e.g. 0.95 for 95th percentile.
        n_samples : int
            Samples per observation.
        seed : int
            Random seed.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        draws = self.sample(n_samples=n_samples, seed=seed)
        return np.quantile(draws, q, axis=1)

    def price_layer(
        self,
        attachment: float,
        limit: float,
        n_samples: int = 2000,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Per-risk expected loss for a per-occurrence XL layer [attachment, attachment+limit].

        E[ min(max(Y - attachment, 0), limit) ]

        Computed via Monte Carlo from the fitted mixture. Suitable for excess of
        loss treaty pricing where the layer structure varies by cedant or class.

        Parameters
        ----------
        attachment : float
            Layer attachment point (deductible).
        limit : float
            Layer width (not upper limit).
        n_samples : int
            Monte Carlo samples per risk.
        seed : int
            Random seed.

        Returns
        -------
        np.ndarray, shape (n,)
            Expected layer loss per observation.
        """
        draws = self.sample(n_samples=n_samples, seed=seed)  # (n, n_samples)
        layer_losses = np.minimum(np.maximum(draws - attachment, 0.0), limit)
        return layer_losses.mean(axis=1)

    def __repr__(self) -> str:
        n, K = self.weights.shape
        return (
            f"GMMPrediction(n={n}, K={K}, "
            f"mean=[{self.mean.min():.4g}, {self.mean.max():.4g}], "
            f"log_transform={self.log_transform})"
        )


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


class NeuralGaussianMixture:
    """
    Neural Gaussian Mixture Model for insurance loss distributions.

    Trains a feedforward neural network that maps risk features X to
    K-component Gaussian mixture parameters (weights, means, variances)
    using a hybrid NLL + Energy Score loss. Returns full conditional
    distribution per risk — not just the mean.

    The energy score component (Gneiting & Raftery 2007) is computed analytically
    via the O(K^2) formula of Yang et al. (2026), avoiding Monte Carlo at training
    time. This gives stable gradients and scales to large portfolios on Databricks.

    Parameters
    ----------
    n_components : int
        Number of Gaussian mixture components K. Default 3. For bimodal
        severity (e.g. attritional + large loss), K=2 suffices. Increase
        to K=5-8 for heavy-tailed motor severity.
    hidden_size : int
        Width of hidden layers. Default 64.
    n_layers : int
        Number of hidden layers. Default 2.
    energy_weight : float
        Weight on energy score in hybrid loss: L = energy_weight * L_ES + (1 - energy_weight) * L_NLL.
        Default 0.5. Set to 0 for pure NLL, 1 for pure energy score.
    learning_rate : float
        Adam learning rate. Default 1e-3.
    epochs : int
        Training epochs. Default 200. Use 50-100 for fast experiments.
    batch_size : int
        Mini-batch size. Default 256.
    var_eps : float
        Minimum variance floor (added after softplus). Default 1e-6.
    log_transform : bool
        If True, trains on log(y) instead of y. Recommended for heavy-tailed
        insurance severity. Predictions are back-transformed automatically.
        Default False.
    random_state : int
        Random seed for reproducibility. Default 42.
    verbose : bool
        Print epoch loss every 50 epochs. Default False.

    References
    ----------
    Yang, Ji, Li & Deng (2026). Energy Score-Guided Neural Gaussian Mixture
    Model for Predictive Uncertainty Quantification. arXiv:2603.27672.

    Gneiting & Raftery (2007). Strictly Proper Scoring Rules, Prediction, and
    Estimation. JASA 102(477):359-378.
    """

    def __init__(
        self,
        n_components: int = 3,
        hidden_size: int = 64,
        n_layers: int = 2,
        energy_weight: float = 0.5,
        learning_rate: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 256,
        var_eps: float = 1e-6,
        log_transform: bool = False,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        _require_torch()
        if not 0.0 <= energy_weight <= 1.0:
            raise ValueError(f"energy_weight must be in [0, 1], got {energy_weight}")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.n_components = n_components
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.energy_weight = energy_weight
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.var_eps = var_eps
        self.log_transform = log_transform
        self.random_state = random_state
        self.verbose = verbose

        self._net: Optional["_GMMNetwork"] = None
        self._is_fitted: bool = False
        self._n_features: Optional[int] = None
        self.training_losses_: list[float] = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralGaussianMixture":
        """
        Fit the neural Gaussian mixture to training data.

        Parameters
        ----------
        X : array-like, shape (n, p)
            Feature matrix. Accepts numpy arrays and Polars DataFrames.
        y : array-like, shape (n,)
            Observed loss values. Must be positive if log_transform=True.

        Returns
        -------
        self
        """
        _require_torch()
        X_arr, y_arr = self._validate_inputs(X, y)

        if self.log_transform:
            if (y_arr <= 0).any():
                raise ValueError(
                    "log_transform=True requires all y > 0, but found non-positive values."
                )
            y_arr = np.log(y_arr)

        n, p = X_arr.shape
        self._n_features = p

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_t = torch.tensor(X_arr, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        net = _GMMNetwork(
            n_features=p,
            n_components=self.n_components,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            var_eps=self.var_eps,
        )
        opt = Adam(net.parameters(), lr=self.learning_rate)

        self.training_losses_ = []

        for epoch in range(self.epochs):
            net.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in loader:
                opt.zero_grad()
                pi, mu, sigma2 = net(batch_X)
                loss = self._hybrid_loss(pi, mu, sigma2, batch_y)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.training_losses_.append(avg_loss)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}  loss={avg_loss:.6f}")

        self._net = net
        self._is_fitted = True
        return self

    def _hybrid_loss(
        self,
        pi: "torch.Tensor",
        mu: "torch.Tensor",
        sigma2: "torch.Tensor",
        y: "torch.Tensor",
    ) -> "torch.Tensor":
        """Weighted combination: energy_weight * ES + (1 - energy_weight) * NLL."""
        if self.energy_weight == 0.0:
            return _nll_loss(pi, mu, sigma2, y)
        elif self.energy_weight == 1.0:
            return _energy_score_loss(pi, mu, sigma2, y)
        else:
            nll = _nll_loss(pi, mu, sigma2, y)
            es = _energy_score_loss(pi, mu, sigma2, y)
            return (1.0 - self.energy_weight) * nll + self.energy_weight * es

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> GMMPrediction:
        """
        Predict full Gaussian mixture distribution for each observation in X.

        Parameters
        ----------
        X : array-like, shape (m, p)
            Feature matrix.

        Returns
        -------
        GMMPrediction
            Container with mixture weights, means, and variances per observation.
        """
        _require_torch()
        self._check_fitted()
        X_arr = self._validate_X(X)

        self._net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_arr, dtype=torch.float32)
            pi, mu, sigma2 = self._net(X_t)

        return GMMPrediction(
            weights=pi.numpy(),
            means=mu.numpy(),
            vars=sigma2.numpy(),
            log_transform=self.log_transform,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def log_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Mean log-likelihood score on held-out data.

        Parameters
        ----------
        X : array-like, shape (n, p)
        y : array-like, shape (n,)

        Returns
        -------
        float
            Mean log p(y|x). Higher is better.
        """
        _require_torch()
        self._check_fitted()
        X_arr, y_arr = self._validate_inputs(X, y)
        if self.log_transform:
            y_arr = np.log(y_arr)

        self._net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_arr, dtype=torch.float32)
            y_t = torch.tensor(y_arr, dtype=torch.float32)
            pi, mu, sigma2 = self._net(X_t)
            nll = _nll_loss(pi, mu, sigma2, y_t)
        return float(-nll.item())

    def energy_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Mean analytic energy score on held-out data.

        Uses the exact O(K^2) formula — no Monte Carlo sampling.
        Lower is better (energy score is a loss, not a gain).

        Parameters
        ----------
        X : array-like, shape (n, p)
        y : array-like, shape (n,)

        Returns
        -------
        float
            Mean energy score. Lower is better.
        """
        _require_torch()
        self._check_fitted()
        X_arr, y_arr = self._validate_inputs(X, y)
        if self.log_transform:
            y_arr = np.log(y_arr)

        self._net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_arr, dtype=torch.float32)
            y_t = torch.tensor(y_arr, dtype=torch.float32)
            pi, mu, sigma2 = self._net(X_t)
            es = _energy_score_loss(pi, mu, sigma2, y_t)
        return float(es.item())

    def crps(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 2000,
        seed: int = 42,
    ) -> float:
        """
        Monte Carlo CRPS estimate on held-out data.

        CRPS(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]
        Estimated via MC: draw n_samples from the fitted mixture, then use
        the sample-based CRPS formula.

        Parameters
        ----------
        X : array-like, shape (n, p)
        y : array-like, shape (n,)
        n_samples : int
            MC samples per observation. Default 2000.
        seed : int
            Random seed. Default 42.

        Returns
        -------
        float
            Mean CRPS. Lower is better. Must be positive.
        """
        self._check_fitted()
        pred = self.predict(X)
        _, y_arr = self._validate_inputs(X, y)

        draws = pred.sample(n_samples=n_samples, seed=seed)  # (n, S)
        # If log_transform, draws are already exponentiated by .sample()
        # so y_arr stays in original space

        y_col = y_arr[:, None]  # (n, 1)
        term1 = np.abs(draws - y_col).mean(axis=1)              # (n,)
        # Unbiased second term: E[|X - X'|] ~ mean of all pairwise |xi - xj|
        # Efficient O(n*S*log(S)) via sorted approach
        draws_sorted = np.sort(draws, axis=1)  # (n, S)
        S = n_samples
        # Sum of absolute differences for sorted samples:
        # sum_{i<j} |x_i - x_j| = sum_k x_k * (2k - S + 1) for 0-indexed k
        k = np.arange(S)
        weights = 2.0 * k - S + 1  # (S,)
        term2 = (draws_sorted * weights[None, :]).sum(axis=1) / (S * (S - 1))

        crps_vals = term1 - term2  # (n,)  term2 = sum_{i<j}|xi-xj|/(S*(S-1)) = 0.5*E[|X-X'|]
        return float(np.mean(crps_vals))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self, X: object, y: object
    ) -> tuple[np.ndarray, np.ndarray]:
        X_arr = self._validate_X(X)
        try:
            import polars as pl
            if isinstance(y, pl.Series):
                y = y.to_numpy()
        except ImportError:
            pass
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y_arr.shape}")
        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"X and y must have the same number of rows: "
                f"{len(X_arr)} vs {len(y_arr)}"
            )
        return X_arr, y_arr

    def _validate_X(self, X: object) -> np.ndarray:
        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                X = X.to_numpy()
        except ImportError:
            pass
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        if self._is_fitted and self._n_features is not None:
            if X_arr.shape[1] != self._n_features:
                raise ValueError(
                    f"X has {X_arr.shape[1]} features, expected {self._n_features}"
                )
        return X_arr

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._net is None:
            raise RuntimeError(
                "NeuralGaussianMixture is not fitted. Call fit() first."
            )

    def __repr__(self) -> str:
        return (
            f"NeuralGaussianMixture("
            f"n_components={self.n_components}, "
            f"hidden_size={self.hidden_size}, "
            f"n_layers={self.n_layers}, "
            f"energy_weight={self.energy_weight}, "
            f"epochs={self.epochs}, "
            f"log_transform={self.log_transform}, "
            f"fitted={self._is_fitted})"
        )
