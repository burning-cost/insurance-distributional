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
            if self.r is None:
                raise ValueError("r required for NegBinom sampling")
            # scipy parameterisation: p = r/(r+mu), n=r
            p = self.r / (self.r + self.mu + 1e-12)
            from scipy.stats import nbinom
            result = np.empty((n, n_samples))
            for i in range(n):
                result[i, :] = nbinom.rvs(
                    n=self.r[i], p=p[i], size=n_samples, random_state=int(rng.integers(1e9))
                )
            return result
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
        """
        if self.phi is None or self.power is None:
            raise ValueError("phi and power required for Tweedie sampling")

        p = self.power
        phi = self.phi
        mu = self.mu
        n = len(mu)

        # Compound Poisson-Gamma parameters
        lam_tw = mu ** (2 - p) / (phi * (2 - p))
        alpha = (2 - p) / (p - 1)
        beta = mu ** (1 - p) / (phi * (p - 1))  # rate param

        result = np.zeros((n, n_samples))
        for i in range(n):
            # Draw Poisson counts
            counts = rng.poisson(lam_tw[i], size=n_samples)
            # For each sample, sum 'count' gamma variates
            max_count = int(counts.max()) if counts.max() > 0 else 0
            if max_count > 0:
                # Draw a (n_samples x max_count) gamma matrix, mask by count
                gamma_matrix = rng.gamma(
                    shape=alpha, scale=1.0 / beta[i],
                    size=(n_samples, max_count)
                )
                mask = np.arange(max_count)[None, :] < counts[:, None]
                result[i, :] = (gamma_matrix * mask).sum(axis=1)
            # where counts==0, result stays 0

        return result

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
