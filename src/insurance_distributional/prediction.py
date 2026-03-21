"""
DistributionalPrediction: container for distributional GBM outputs.

This object is what you get back from model.predict(). It bundles the predicted
distribution parameters together with derived actuarial quantities (variance,
coefficient of variation, volatility score) so pricing teams don't have to
recompute them manually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DistributionalPrediction:
    """
    Container for the output of a distributional GBM prediction.

    Attributes
    ----------
    distribution : str
        Name of the fitted distribution ("tweedie", "gamma", "zip", "negbinom").
    mu : np.ndarray
        Predicted mean E[Y|X] for each observation.
    phi : np.ndarray, optional
        Predicted dispersion parameter phi(X) (Tweedie, Gamma).
    pi : np.ndarray, optional
        Predicted zero-inflation probability pi(X) (ZIP).
    r : np.ndarray, optional
        Predicted size/overdispersion parameter r(X) (Negative Binomial).
    power : float, optional
        Tweedie power parameter p (fixed at fit time).
    """

    distribution: str
    mu: np.ndarray
    phi: Optional[np.ndarray] = field(default=None)
    pi: Optional[np.ndarray] = field(default=None)
    r: Optional[np.ndarray] = field(default=None)
    power: Optional[float] = field(default=None)

    # -------------------------------------------------------------------------
    # Core actuarial properties
    # -------------------------------------------------------------------------

    @property
    def mean(self) -> np.ndarray:
        """Predicted conditional mean E[Y|X]."""
        return self.mu

    @property
    def variance(self) -> np.ndarray:
        """
        Predicted conditional variance Var[Y|X].

        Tweedie : phi * mu^p
        Gamma   : phi * mu^2  (phi = 1/shape, so sigma^2 = mu^2 / shape)
        ZIP     : (1-pi)*lambda + (1-pi)*pi*lambda^2  = (1-pi)*lambda*(1+pi*lambda)
        NegBinom: mu + mu^2/r
        """
        dist = self.distribution.lower()
        if dist == "tweedie":
            if self.phi is None or self.power is None:
                raise ValueError("phi and power required for Tweedie variance")
            return self.phi * np.power(self.mu, self.power)
        elif dist == "gamma":
            if self.phi is None:
                raise ValueError("phi required for Gamma variance")
            # phi is the dispersion (1/shape); Var = phi * mu^2
            return self.phi * self.mu ** 2
        elif dist == "zip":
            if self.pi is None:
                raise ValueError("pi required for ZIP variance")
            lam = self.mu / (1.0 - self.pi + 1e-12)
            return (1.0 - self.pi) * lam * (1.0 + self.pi * lam)
        elif dist == "negbinom":
            if self.r is None:
                raise ValueError("r required for Negative Binomial variance")
            return self.mu + self.mu ** 2 / self.r
        else:
            raise ValueError(f"Unknown distribution: {self.distribution!r}")

    @property
    def std(self) -> np.ndarray:
        """Predicted conditional standard deviation."""
        return np.sqrt(self.variance)

    @property
    def cov(self) -> np.ndarray:
        """
        Coefficient of variation (CoV) = SD / mean.

        A dimensionless per-risk measure of relative uncertainty. Two risks
        with the same predicted mean but different CoV have different risk
        profiles — the higher-CoV risk warrants a larger safety loading.
        """
        return self.std / (self.mu + 1e-12)

    def volatility_score(self) -> np.ndarray:
        """
        Dimensionless volatility ranking: CoV per risk.

        This is the key actuarial output of distributional modelling — it
        identifies which risks are intrinsically more volatile beyond what
        their expected loss implies. Useful for:
        - Safety loading calibration
        - Underwriter referral thresholds
        - Reinsurance attachment optimisation
        - IFRS 17 risk adjustment

        Returns
        -------
        np.ndarray
            CoV = sqrt(Var[Y|x]) / E[Y|x] for each observation.
        """
        return self.cov

    def quantile(self, q: float, n_samples: int = 10_000, seed: int = 42) -> np.ndarray:
        """
        Monte Carlo quantile estimate from the fitted distribution.

        Parameters
        ----------
        q : float
            Quantile level, e.g. 0.95 for 95th percentile.
        n_samples : int
            Number of MC samples per observation (default 10,000).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Estimated q-th quantile for each observation.
        """
        rng = np.random.default_rng(seed)
        samples = self._sample(n_samples=n_samples, rng=rng)
        return np.quantile(samples, q, axis=1)

    def _sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw samples from the fitted distribution.

        Returns an (n_obs, n_samples) array.
        """
        n = len(self.mu)
        dist = self.distribution.lower()

        if dist == "tweedie":
            return self._sample_tweedie(n_samples, rng)
        elif dist == "gamma":
            if self.phi is None:
                raise ValueError("phi required for Gamma sampling")
            # shape = 1/phi, scale = mu*phi
            shape = 1.0 / (self.phi + 1e-12)
            scale = self.mu * self.phi
            return rng.gamma(
                shape[:, None], scale[:, None], size=(n, n_samples)
            )
        elif dist == "zip":
            if self.pi is None:
                raise ValueError("pi required for ZIP sampling")
            lam = self.mu / (1.0 - self.pi + 1e-12)
            zero_mask = rng.random(size=(n, n_samples)) < self.pi[:, None]
            poisson_samples = rng.poisson(lam[:, None], size=(n, n_samples))
            return np.where(zero_mask, 0, poisson_samples).astype(float)
        elif dist == "negbinom":
            return self._sample_negbinom(n_samples, rng)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution!r}")

    def _sample_tweedie(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample from Tweedie compound Poisson-Gamma distribution.

        For p in (1, 2), Tweedie(mu, phi, p) = sum of N Gamma random variables
        where N ~ Poisson(lambda_tw) and each Gamma has shape alpha, rate beta.

        Parameterisation following Jørgensen (1987):
          lambda_tw = mu^(2-p) / (phi*(2-p))
          alpha = (2-p)/(p-1)
          beta = mu^(1-p) / (phi*(p-1))

        Vectorised implementation: draws the full (n_obs, n_samples) Poisson
        count matrix in one call, then uses a max_count x n_obs x n_samples
        Gamma tensor with masking to sum compound terms — no Python loop over
        observations. This scales to n=50,000 on Databricks without timing out.

        The max_count approach caps the Gamma draw at the largest Poisson count
        seen across all observations and samples; counts below max_count are
        masked to zero before summing. Expected max_count grows as
        O(lambda_max + sqrt(lambda_max * log(n*n_samples))) which is manageable
        for typical insurance lambda values (< 100).
        """
        if self.phi is None or self.power is None:
            raise ValueError("phi and power required for Tweedie sampling")

        p = self.power
        phi = self.phi
        mu = self.mu
        n = len(mu)

        # Compound Poisson-Gamma parameters (shape: n_obs)
        lam_tw = mu ** (2 - p) / (phi * (2 - p))   # Poisson rate per obs
        alpha = (2 - p) / (p - 1)                   # Gamma shape (scalar)
        beta = mu ** (1 - p) / (phi * (p - 1))      # Gamma rate per obs (n_obs,)

        # Draw all Poisson counts in one vectorised call: shape (n_obs, n_samples)
        counts = rng.poisson(lam_tw[:, None], size=(n, n_samples))

        max_count = int(counts.max())
        if max_count == 0:
            return np.zeros((n, n_samples))

        # Draw Gamma variates: shape (n_obs, n_samples, max_count)
        # scale = 1/beta, broadcast over (n_obs, 1, 1)
        gamma_draws = rng.gamma(
            shape=alpha,
            scale=(1.0 / beta)[:, None, None],
            size=(n, n_samples, max_count),
        )

        # Mask: position k is valid only if k < counts[i, j]
        # k_idx shape: (1, 1, max_count) — broadcasts against (n, n_samples, max_count)
        k_idx = np.arange(max_count)[None, None, :]
        mask = k_idx < counts[:, :, None]           # (n_obs, n_samples, max_count)

        # Sum valid Gamma terms along the last axis
        return (gamma_draws * mask).sum(axis=2)     # (n_obs, n_samples)

    def _sample_negbinom(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample from Negative Binomial distribution.

        scipy nbinom.rvs accepts array-valued n and p arguments, so the entire
        (n_obs, n_samples) result can be drawn in a single call — no Python
        loop over observations.

        Parameterisation: NB(r, p) where p = r/(r+mu), matching scipy's
        convention (n=r, p=success probability).
        """
        if self.r is None:
            raise ValueError("r required for NegBinom sampling")

        from scipy.stats import nbinom

        n_obs = len(self.mu)
        p = self.r / (self.r + self.mu + 1e-12)    # shape (n_obs,)

        # scipy nbinom.rvs broadcasts n and p against size, giving
        # shape (n_obs, n_samples) directly. Use a single integer seed
        # derived from the rng state so results are reproducible.
        seed = int(rng.integers(1 << 31))
        return nbinom.rvs(
            n=self.r[:, None],
            p=p[:, None],
            size=(n_obs, n_samples),
            random_state=seed,
        ).astype(float)

    def __repr__(self) -> str:
        n = len(self.mu)
        parts = [f"distribution={self.distribution!r}", f"n={n}"]
        parts.append(f"mean=[{self.mu.min():.4g}, {self.mu.max():.4g}]")
        if self.phi is not None:
            parts.append(f"phi=[{self.phi.min():.4g}, {self.phi.max():.4g}]")
        if self.pi is not None:
            parts.append(f"pi=[{self.pi.min():.4g}, {self.pi.max():.4g}]")
        if self.r is not None:
            parts.append(f"r=[{self.r.min():.4g}, {self.r.max():.4g}]")
        return f"DistributionalPrediction({', '.join(parts)})"
