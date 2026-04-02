"""
Generative Adversarial Regression (GAR) for conditional risk scenario generation.

Implements the minimax framework of Asadi & Li (2026) arXiv:2603.08553, adapted for
UK P&C insurance scenario generation. The core idea: instead of training a generator
to fool a discriminator (GAN-style), we train it against an adversarial *policy* that
tries to maximally expose the generator's risk-functional mismatch. The generator
minimises worst-case proper-scoring-function loss.

This matters for Solvency II internal model work: you need scenario distributions that
are aligned with the specific risk functional you care about (VaR, ES at 99.5%), not
just distributional realism in a general sense.

Mathematical framework:
    min_theta max_{phi in Phi} E[ S( rho(Pi_phi(G_theta(Z,C)) | C), Pi_phi(Y) ) ]

where:
    G_theta(Z, C) -- generator: latent noise Z + context C -> synthetic scenario
    Pi_phi(y)     -- adversarial policy: weighted sum of loss components
    rho(. | C)    -- conditional risk functional (VaR, ES, expectile)
    S(a, l)       -- strictly consistent scoring function for rho

References:
    Asadi & Li (2026). Generative Adversarial Regression: Learning Conditional Risk
    Scenarios. arXiv:2603.08553.

    Fissler & Ziegel (2016). Higher order elicitability and Osband's principle.
    Annals of Statistics, 44(4):1680-1707.
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for GARScenarioGenerator. "
            "Install with: pip install insurance-distributional[gar]"
        )


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------


class _LinearGenerator(nn.Module):
    """Simple feedforward generator: concat(Z, C) -> Linear -> ReLU -> Linear -> Y.

    Architecture: noise Z and context C are concatenated and passed through a two-layer
    MLP with ReLU activation. This is the 'Simple-Linear' encoder from Asadi & Li (2026).
    """

    def __init__(self, n_assets: int, context_size: int, latent_dim: int, hidden_size: int) -> None:
        super().__init__()
        in_dim = latent_dim + context_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_assets),
        )

    def forward(self, Z: "torch.Tensor", C: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            Z: (batch, n_mc, latent_dim)
            C: (batch, n_mc, context_size)
        Returns:
            (batch, n_mc, n_assets)
        """
        x = torch.cat([Z, C], dim=-1)
        return self.net(x)


class _LSTMGenerator(nn.Module):
    """LSTM encoder generator for sequential context.

    The LSTM processes context history C (treated as a sequence along the last
    two dimensions if C is 2D, or a single-step sequence for 1D context). The
    hidden state h_C is concatenated with latent noise Z before decoding.

    This corresponds to the 'Encoder-LSTM' architecture in Asadi & Li (2026),
    which achieves the best performance on financial time series.
    """

    def __init__(self, n_assets: int, context_size: int, latent_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(context_size, hidden_size, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_assets),
        )

    def forward(self, Z: "torch.Tensor", C: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            Z: (batch, n_mc, latent_dim)
            C: (batch, n_mc, context_size) — for LSTM, context treated as 1-step sequence
        Returns:
            (batch, n_mc, n_assets)
        """
        batch_size, n_mc, context_size = C.shape
        # Reshape to run LSTM: treat each (batch * n_mc) as a 1-step sequence
        C_flat = C.reshape(batch_size * n_mc, 1, context_size)
        _, (h_n, _) = self.lstm(C_flat)  # h_n: (1, batch*n_mc, hidden)
        h_C = h_n.squeeze(0).reshape(batch_size, n_mc, self.hidden_size)
        xin = torch.cat([Z, h_C], dim=-1)
        return self.decoder(xin)


class _PolicyNetwork(nn.Module):
    """Adversarial policy network: scenarios -> portfolio payout.

    Represents the adversarial policy class Pi_phi. For a batch of (n_assets,)-dimensional
    scenarios, outputs a scalar payout per scenario. The policy is a small MLP with
    Tanh activation; weights are not explicitly constrained to unit gross exposure here
    because the normalisation is applied at training time.

    For insurance: interpret this as a reinsurance treaty payout function parameterised
    by attachment/limit, which the adversary adjusts to expose worst-case risk.
    """

    def __init__(self, n_assets: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_assets, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x: (..., n_assets)
        Returns:
            (...,) scalar payout per scenario
        """
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Scoring functions (Fissler-Ziegel and quantile/expectile)
# ---------------------------------------------------------------------------


def _score_var(rho_hat: "torch.Tensor", pi_y: "torch.Tensor", alpha: float) -> "torch.Tensor":
    """Quantile (VaR) scoring function.

    S_alpha(a, l) = |alpha - 1_{l <= a}| * |l - a|

    Strictly consistent for the alpha-quantile (Gneiting 2011).

    Args:
        rho_hat: (B,) estimated VaR from synthetic distribution
        pi_y: (B,) policy payout on real outcomes
        alpha: tail level (e.g. 0.05 for 95th percentile VaR)
    """
    indicator = (pi_y <= rho_hat).float()
    return torch.abs(alpha - indicator) * torch.abs(pi_y - rho_hat)


def _score_expectile(rho_hat: "torch.Tensor", pi_y: "torch.Tensor", alpha: float) -> "torch.Tensor":
    """Expectile scoring function.

    S_tau(a, l) = |tau - 1_{l <= a}| * (l - a)^2

    Strictly consistent for the tau-expectile (Newey & Powell 1987).

    Args:
        rho_hat: (B,) estimated expectile
        pi_y: (B,) policy payout on real outcomes
        alpha: asymmetry parameter tau
    """
    indicator = (pi_y <= rho_hat).float()
    return torch.abs(alpha - indicator) * (pi_y - rho_hat) ** 2


def _score_var_es(rho_hat: "torch.Tensor", pi_y: "torch.Tensor", alpha: float, s: float = 1.0) -> "torch.Tensor":
    """Fissler-Ziegel joint (VaR, ES) scoring function.

    S_alpha((v, e), l) =
        (1_{l<=v} - alpha)(H1(v) - H1(l))
        + (1/alpha) H2'(e) * 1_{l<=v} * (v - l)
        + H2'(e)(e - v) - H2(e)

    with H1(x) = x, H2(x) = s * exp(x/s), H2'(x) = exp(x/s).

    Strictly consistent for (VaR_alpha, ES_alpha) jointly (Fissler & Ziegel 2016).

    Args:
        rho_hat: (B, 2) — first column VaR, second column ES
        pi_y: (B,) policy payout on real outcomes
        alpha: tail level
        s: scale parameter for H2; larger s reduces gradient magnitude
    """
    v = rho_hat[:, 0]
    e = rho_hat[:, 1]
    ind = (pi_y <= v).float()
    dH2_e = torch.exp(e / s)
    H2_e = s * torch.exp(e / s)
    term1 = (ind - alpha) * (v - pi_y)
    term2 = (1.0 / alpha) * dH2_e * ind * (v - pi_y)
    term3 = dH2_e * (e - v) - H2_e
    return term1 + term2 + term3


def _estimate_var(pi_syn: "torch.Tensor", alpha: float) -> "torch.Tensor":
    """Estimate VaR_alpha via empirical quantile over MC samples.

    Args:
        pi_syn: (B, n_mc) policy payouts on synthetic scenarios
        alpha: tail level (0.05 = 95th pctile, i.e. upper tail)
    Returns:
        (B,)
    """
    return torch.quantile(pi_syn, 1.0 - alpha, dim=1)


def _estimate_expectile(pi_syn: "torch.Tensor", alpha: float, n_iter: int = 10) -> "torch.Tensor":
    """Estimate tau-expectile via iterative reweighting (IRLS).

    Converges in ~10 Newton steps for smooth distributions.

    Args:
        pi_syn: (B, n_mc)
        alpha: tau level
        n_iter: Newton iterations
    Returns:
        (B,)
    """
    e = pi_syn.mean(dim=1, keepdim=True)
    for _ in range(n_iter):
        ind = (pi_syn <= e).float()
        w = torch.abs(alpha - ind)
        e = (w * pi_syn).sum(dim=1, keepdim=True) / (w.sum(dim=1, keepdim=True) + 1e-8)
    return e.squeeze(1)


def _estimate_var_es(pi_syn: "torch.Tensor", alpha: float) -> "torch.Tensor":
    """Estimate (VaR_alpha, ES_alpha) jointly via MC.

    Args:
        pi_syn: (B, n_mc)
        alpha: tail level
    Returns:
        (B, 2) — columns [VaR, ES]
    """
    var = torch.quantile(pi_syn, 1.0 - alpha, dim=1)  # (B,)
    # ES = E[X | X >= VaR]
    es_list = []
    for i in range(pi_syn.shape[0]):
        row = pi_syn[i]
        tail_mask = row >= var[i]
        if tail_mask.sum() == 0:
            es_list.append(var[i])
        else:
            es_list.append(row[tail_mask].mean())
    es = torch.stack(es_list)
    return torch.stack([var, es], dim=1)  # (B, 2)


def _compute_score(
    rho_hat: "torch.Tensor",
    pi_y: "torch.Tensor",
    alpha: float,
    functional: str,
    fz_scale: float,
) -> "torch.Tensor":
    """Dispatch scoring function by name."""
    if functional == "var":
        return _score_var(rho_hat, pi_y, alpha)
    elif functional == "expectile":
        return _score_expectile(rho_hat, pi_y, alpha)
    elif functional == "var_es":
        return _score_var_es(rho_hat, pi_y, alpha, s=fz_scale)
    else:
        raise ValueError(f"Unknown risk_functional: {functional!r}. Choose 'var', 'expectile', or 'var_es'.")


def _compute_risk(pi_syn: "torch.Tensor", alpha: float, functional: str) -> "torch.Tensor":
    """Dispatch risk estimator by name."""
    if functional == "var":
        return _estimate_var(pi_syn, alpha)
    elif functional == "expectile":
        return _estimate_expectile(pi_syn, alpha)
    elif functional == "var_es":
        return _estimate_var_es(pi_syn, alpha)
    else:
        raise ValueError(f"Unknown risk_functional: {functional!r}.")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class GARScenarioGenerator:
    """Generative Adversarial Regression for conditional insurance loss scenarios.

    Trains a generative model G_theta(Z, C) whose output scenarios are aligned with
    a chosen risk functional (VaR, ES, or expectile) via minimax adversarial training.
    An adversarial policy Pi_phi finds portfolio weightings that maximally expose
    risk-functional mismatch; the generator minimises this worst-case scoring loss.

    This is distinct from a standard GAN: the adversary never sees individual samples
    and cannot classify real vs fake. It only evaluates the *risk* of the generator's
    output under a proper scoring function. This means training stability is better
    (no mode collapse) but requires more MC samples per step.

    Solvency II use case:
        gar = GARScenarioGenerator(n_assets=1, context_size=10, risk_functional='var_es',
                                   alpha=0.005)
        gar.fit(C_train, y_aggregate)
        scenarios = gar.generate(C_current, n_scenarios=10_000)
        scr = np.quantile(scenarios[0, :, 0], 0.995)

    Args:
        n_assets: Dimension of output scenario. 1 for univariate aggregate loss.
        context_size: Dimension of conditioning context C (risk covariates).
        latent_dim: Dimension of latent noise Z ~ N(0, I).
        hidden_size: Width of hidden layers in generator and policy networks.
        risk_functional: Which risk measure to align with. One of:
            'var'       — VaR at level alpha (quantile scoring)
            'expectile' — expectile at level alpha
            'var_es'    — joint (VaR, ES) via Fissler-Ziegel score (recommended for SCR)
        alpha: Tail level. 0.05 = 95th percentile (upper tail). Use 0.005 for Solvency II SCR.
        n_mc: Monte Carlo samples for risk estimation per training step. Higher = more
            stable gradients but more memory. 200 is a reasonable default; use 500+ for
            production runs.
        encoder: Generator architecture. 'linear' (default) for tabular covariates;
            'lstm' for sequential context (time series of past losses).
        max_epochs: Training epochs. More epochs = better calibration at cost of runtime.
        lr_gen: Learning rate for generator Adam optimiser.
        lr_policy: Learning rate for policy Adam optimiser.
        batch_size: Training batch size.
        fz_scale: Scale parameter s in H2(e) = s*exp(e/s) for Fissler-Ziegel scoring.
            Affects gradient magnitude; default 1.0. Expose if training is unstable.
        grad_clip: Gradient clipping value applied to both networks. Reduces variance
            in minimax training. Default 1.0.
        warmup_epochs: Initial epochs where only the generator trains (no policy update).
            Helps convergence for very right-skewed insurance data.
        random_state: Random seed for reproducibility.

    References:
        Asadi & Li (2026). Generative Adversarial Regression. arXiv:2603.08553.
        Fissler & Ziegel (2016). Higher order elicitability. Ann. Statist. 44(4).
    """

    def __init__(
        self,
        n_assets: int = 1,
        context_size: int = 1,
        latent_dim: int = 32,
        hidden_size: int = 64,
        risk_functional: Literal["var", "expectile", "var_es"] = "var_es",
        alpha: float = 0.05,
        n_mc: int = 200,
        encoder: Literal["linear", "lstm"] = "linear",
        max_epochs: int = 100,
        lr_gen: float = 1e-3,
        lr_policy: float = 1e-3,
        batch_size: int = 64,
        fz_scale: float = 1.0,
        grad_clip: float = 1.0,
        warmup_epochs: int = 5,
        random_state: int = 42,
    ) -> None:
        _require_torch()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if risk_functional not in ("var", "expectile", "var_es"):
            raise ValueError(f"risk_functional must be 'var', 'expectile', or 'var_es', got {risk_functional!r}")

        self.n_assets = n_assets
        self.context_size = context_size
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.risk_functional = risk_functional
        self.alpha = alpha
        self.n_mc = n_mc
        self.encoder = encoder
        self.max_epochs = max_epochs
        self.lr_gen = lr_gen
        self.lr_policy = lr_policy
        self.batch_size = batch_size
        self.fz_scale = fz_scale
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.random_state = random_state

        self._generator: Optional["nn.Module"] = None
        self._policy: Optional["nn.Module"] = None
        self._is_fitted: bool = False
        self.training_losses_: list[float] = []

    def _build_generator(self) -> "nn.Module":
        if self.encoder == "linear":
            return _LinearGenerator(self.n_assets, self.context_size, self.latent_dim, self.hidden_size)
        elif self.encoder == "lstm":
            return _LSTMGenerator(self.n_assets, self.context_size, self.latent_dim, self.hidden_size)
        else:
            raise ValueError(f"encoder must be 'linear' or 'lstm', got {self.encoder!r}")

    def _build_policy(self) -> "nn.Module":
        return _PolicyNetwork(self.n_assets, self.hidden_size)

    def fit(self, C_train: np.ndarray, Y_train: np.ndarray) -> "GARScenarioGenerator":
        """Fit the generator via minimax adversarial training.

        Alternates between:
          1. Policy ASCENT: update Pi_phi to maximise scoring loss (find worst-case policy).
          2. Generator DESCENT: update G_theta to minimise scoring loss under current policy.

        Both networks use Adam with gradient clipping for stability.

        Args:
            C_train: Conditioning context, shape (n, context_size). For insurance:
                portfolio risk features (sum insured, peril mix, vehicle age profile).
            Y_train: Observed outcomes, shape (n, n_assets). For insurance:
                aggregate loss outcomes per observation period.

        Returns:
            self (fitted)
        """
        _require_torch()
        C_train = np.asarray(C_train, dtype=np.float32)
        Y_train = np.asarray(Y_train, dtype=np.float32)

        if C_train.ndim == 1:
            C_train = C_train[:, None]
        if Y_train.ndim == 1:
            Y_train = Y_train[:, None]

        n, ctx = C_train.shape
        if ctx != self.context_size:
            raise ValueError(f"C_train has {ctx} columns, expected context_size={self.context_size}")
        if Y_train.shape[1] != self.n_assets:
            raise ValueError(f"Y_train has {Y_train.shape[1]} columns, expected n_assets={self.n_assets}")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        C_t = torch.tensor(C_train)
        Y_t = torch.tensor(Y_train)

        dataset = TensorDataset(C_t, Y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        gen = self._build_generator()
        policy = self._build_policy()

        opt_gen = Adam(gen.parameters(), lr=self.lr_gen)
        opt_policy = Adam(policy.parameters(), lr=self.lr_policy)

        self.training_losses_ = []

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_C, batch_Y in loader:
                B = batch_C.shape[0]

                # ---- Policy ASCENT step (skip during warmup) ----
                if epoch >= self.warmup_epochs:
                    opt_policy.zero_grad()
                    with torch.no_grad():
                        Z = torch.randn(B, self.n_mc, self.latent_dim)
                        C_exp = batch_C.unsqueeze(1).expand(-1, self.n_mc, -1)
                        syn = gen(Z, C_exp)  # (B, n_mc, n_assets)

                    pi_y = policy(batch_Y)  # (B,)
                    pi_syn = policy(syn)    # (B, n_mc)

                    # Normalise to unit gross exposure
                    pi_y_norm = pi_y / (pi_y.abs().mean() + 1e-8)
                    pi_syn_norm = pi_syn / (pi_syn.abs().mean(dim=1, keepdim=True) + 1e-8)

                    rho_hat = _compute_risk(pi_syn_norm, self.alpha, self.risk_functional)
                    loss_policy = -_compute_score(rho_hat, pi_y_norm, self.alpha, self.risk_functional, self.fz_scale).mean()

                    loss_policy.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
                    opt_policy.step()

                # ---- Generator DESCENT step ----
                opt_gen.zero_grad()
                Z = torch.randn(B, self.n_mc, self.latent_dim)
                C_exp = batch_C.unsqueeze(1).expand(-1, self.n_mc, -1)
                syn = gen(Z, C_exp)

                with torch.no_grad():
                    pi_y = policy(batch_Y)
                    pi_y_norm = pi_y / (pi_y.abs().mean() + 1e-8)

                pi_syn = policy(syn)
                pi_syn_norm = pi_syn / (pi_syn.abs().mean(dim=1, keepdim=True) + 1e-8)

                rho_hat = _compute_risk(pi_syn_norm, self.alpha, self.risk_functional)
                loss_gen = _compute_score(rho_hat, pi_y_norm, self.alpha, self.risk_functional, self.fz_scale).mean()

                loss_gen.backward()
                nn.utils.clip_grad_norm_(gen.parameters(), self.grad_clip)
                opt_gen.step()

                epoch_loss += loss_gen.item()
                n_batches += 1

            if n_batches > 0:
                self.training_losses_.append(epoch_loss / n_batches)

        self._generator = gen
        self._policy = policy
        self._is_fitted = True
        return self

    def generate(self, C: np.ndarray, n_scenarios: int = 1000) -> np.ndarray:
        """Generate synthetic loss scenarios conditioned on context C.

        Args:
            C: Conditioning context, shape (m, context_size) or (context_size,) for scalar.
            n_scenarios: Number of scenarios per context row.

        Returns:
            Array of shape (m, n_scenarios, n_assets).
        """
        _require_torch()
        self._check_fitted()
        C = np.asarray(C, dtype=np.float32)
        if C.ndim == 1:
            C = C[None, :]
        m = C.shape[0]
        C_t = torch.tensor(C)

        gen = self._generator
        gen.eval()
        with torch.no_grad():
            C_exp = C_t.unsqueeze(1).expand(-1, n_scenarios, -1)  # (m, n_scenarios, ctx)
            Z = torch.randn(m, n_scenarios, self.latent_dim)
            out = gen(Z, C_exp)  # (m, n_scenarios, n_assets)
        gen.train()
        return out.numpy()

    def score(self, C_test: np.ndarray, Y_test: np.ndarray) -> float:
        """Mean scoring function evaluated at true outcomes under the trained generator-policy pair.

        Lower is better (the generator minimises this). Useful for comparing generators
        trained with different hyperparameters or architectures.

        Args:
            C_test: Context array, shape (n, context_size).
            Y_test: True outcomes, shape (n, n_assets).

        Returns:
            Mean scoring loss (scalar).
        """
        _require_torch()
        self._check_fitted()
        C_test = np.asarray(C_test, dtype=np.float32)
        Y_test = np.asarray(Y_test, dtype=np.float32)
        if C_test.ndim == 1:
            C_test = C_test[:, None]
        if Y_test.ndim == 1:
            Y_test = Y_test[:, None]

        n = C_test.shape[0]
        C_t = torch.tensor(C_test)
        Y_t = torch.tensor(Y_test)

        gen = self._generator
        policy = self._policy
        gen.eval()
        policy.eval()

        with torch.no_grad():
            C_exp = C_t.unsqueeze(1).expand(-1, self.n_mc, -1)
            Z = torch.randn(n, self.n_mc, self.latent_dim)
            syn = gen(Z, C_exp)

            pi_y = policy(Y_t)
            pi_syn = policy(syn)

            pi_y_norm = pi_y / (pi_y.abs().mean() + 1e-8)
            pi_syn_norm = pi_syn / (pi_syn.abs().mean(dim=1, keepdim=True) + 1e-8)

            rho_hat = _compute_risk(pi_syn_norm, self.alpha, self.risk_functional)
            loss = _compute_score(rho_hat, pi_y_norm, self.alpha, self.risk_functional, self.fz_scale)

        gen.train()
        policy.train()
        return float(loss.mean().item())

    def var(self, C: np.ndarray, level: Optional[float] = None, n_scenarios: int = 2000) -> np.ndarray:
        """Value-at-Risk estimate via Monte Carlo from the fitted generator.

        Args:
            C: Context array, shape (m, context_size) or (context_size,).
            level: Tail level. Defaults to self.alpha. E.g. 0.005 for 99.5% VaR.
            n_scenarios: Number of MC scenarios to use.

        Returns:
            VaR estimates, shape (m,).
        """
        if level is None:
            level = self.alpha
        scenarios = self.generate(C, n_scenarios=n_scenarios)  # (m, n_scenarios, n_assets)
        # For n_assets=1, squeeze; for multi-asset, sum as aggregate
        y = scenarios[:, :, 0] if self.n_assets == 1 else scenarios.sum(axis=-1)
        return np.quantile(y, 1.0 - level, axis=1)

    def es(self, C: np.ndarray, level: Optional[float] = None, n_scenarios: int = 2000) -> np.ndarray:
        """Expected Shortfall (CVaR) estimate via Monte Carlo from the fitted generator.

        Args:
            C: Context array, shape (m, context_size) or (context_size,).
            level: Tail level. Defaults to self.alpha.
            n_scenarios: Number of MC scenarios to use.

        Returns:
            ES estimates, shape (m,).
        """
        if level is None:
            level = self.alpha
        scenarios = self.generate(C, n_scenarios=n_scenarios)
        y = scenarios[:, :, 0] if self.n_assets == 1 else scenarios.sum(axis=-1)
        var_vals = np.quantile(y, 1.0 - level, axis=1, keepdims=True)
        # ES = mean of observations at or above VaR
        es_vals = np.array([
            y[i, y[i] >= var_vals[i, 0]].mean() if (y[i] >= var_vals[i, 0]).any() else var_vals[i, 0]
            for i in range(y.shape[0])
        ])
        return es_vals

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("GARScenarioGenerator is not fitted. Call fit() first.")

    def get_generator_state(self) -> dict:
        """Return generator state dict for serialisation."""
        self._check_fitted()
        return self._generator.state_dict()

    def get_policy_state(self) -> dict:
        """Return policy state dict for serialisation."""
        self._check_fitted()
        return self._policy.state_dict()

    def load_generator_state(self, state_dict: dict) -> None:
        """Load a previously saved generator state dict.

        After loading, generate() will produce identical output given identical seeds.
        """
        _require_torch()
        if self._generator is None:
            self._generator = self._build_generator()
        self._generator.load_state_dict(state_dict)
        if self._policy is None:
            self._policy = self._build_policy()
        self._is_fitted = True

    def __repr__(self) -> str:
        return (
            f"GARScenarioGenerator("
            f"n_assets={self.n_assets}, context_size={self.context_size}, "
            f"risk_functional={self.risk_functional!r}, alpha={self.alpha}, "
            f"encoder={self.encoder!r}, fitted={self._is_fitted})"
        )
