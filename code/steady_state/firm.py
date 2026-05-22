"""
Monopsonistic Firm Problem
==========================
Solves the firm's optimization problem combining:
  (1) Monopoly power in goods markets (standard Calvo pricing)
  (2) Monopsony power in labor markets (wage markdown)

The firm maximizes profit subject to:
  - Upward-sloping firm-level labor supply curve: n_j = (w_j/w̄)^ε * N̄
  - Standard Calvo price-setting friction (probability ξ of not re-pricing)
  - Multi-sector input-output structure for intermediate goods

Key equations:
  w_j* = (ε/(ε+1)) * MPL_j   [wage markdown condition]
  P_j* = (ε_p/(ε_p-1)) * mc_j / μ^w  [reset price with true marginal cost]
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class FirmParameters:
    """Structural parameters for the firm block."""

    # Goods market
    epsilon_p: float = 6.0     # Elasticity of substitution across goods varieties
    xi: float = 0.75           # Calvo probability of NOT re-pricing (quarterly)

    # Labor market monopsony
    epsilon_w_bar: float = 4.0  # Mean firm-level labor supply elasticity
    gamma_w: float = 0.8        # Countercyclical markdown intensity
    u_bar: float = 0.05         # Steady-state unemployment rate

    # Production technology
    alpha_n: float = 0.65       # Labor share in value added
    alpha_m: float = 0.35       # Intermediate input share
    theta: float = 0.5          # Elasticity of substitution: labor vs. intermediates
    eta: float = 2.0            # Elasticity of substitution across intermediate varieties
    A: float = 1.0              # TFP level

    # Markup (goods market)
    @property
    def mu_p(self) -> float:
        """Price markup (goods market monopoly)."""
        return self.epsilon_p / (self.epsilon_p - 1)

    def wage_markdown(self, u: float) -> float:
        """
        Wage markdown as a function of unemployment rate.

        μ^w(u) = ε(u) / (ε(u) + 1)

        where ε(u) = ε̄ - γ(u - ū) is countercyclical:
        - When u rises (recession), ε falls → markdown deepens
        - When u falls (boom), ε rises → markdown shrinks

        Empirical calibration: Manning (2021), Dube et al. (2020)
        """
        epsilon_u = self.epsilon_w_bar - self.gamma_w * (u - self.u_bar)
        epsilon_u = max(epsilon_u, 0.5)  # Floor to prevent division issues
        return epsilon_u / (epsilon_u + 1)

    def steady_state_markdown(self) -> float:
        """Steady-state wage markdown."""
        return self.wage_markdown(self.u_bar)


class FirmSolver:
    """
    Solves the firm's static optimization problem.

    In steady state, firm j in sector s chooses (n_j, m_j, w_j, P_j) to
    maximize profit subject to the production function and market power constraints.
    """

    def __init__(self, params: FirmParameters):
        self.p = params

    def production(
        self,
        n: float,
        M: float,
        z: float = 1.0
    ) -> float:
        """
        Nested CES production function:
            y = z * [α_n^{1/θ} * n^{(θ-1)/θ} + α_m^{1/θ} * M^{(θ-1)/θ}]^{θ/(θ-1)}
        """
        p = self.p
        term_n = p.alpha_n ** (1/p.theta) * n ** ((p.theta-1)/p.theta)
        term_m = p.alpha_m ** (1/p.theta) * M ** ((p.theta-1)/p.theta)
        return z * self.p.A * (term_n + term_m) ** (p.theta / (p.theta-1))

    def MPL(self, n: float, M: float, z: float = 1.0) -> float:
        """Marginal product of labor."""
        p = self.p
        y = self.production(n, M, z)
        # ∂y/∂n = α_n^{1/θ} * n^{-1/θ} * (y/z/A)^{1/θ} * z * A
        return (
            p.alpha_n ** (1/p.theta)
            * n ** (-1/p.theta)
            * (y / (z * p.A)) ** (1 - 1/p.theta)
        )

    def optimal_factor_demands(
        self,
        w: float,
        P_M: float,
        P: float,
        mu_w: float,
        u: float,
        z: float = 1.0,
    ) -> Tuple[float, float, float]:
        """
        Compute optimal labor demand, intermediate demand, and output.

        Under monopsony:
            w* = μ^w * MPL  →  MPL = w / μ^w
            MPM = P_M (cost-minimizing intermediate demand)

        Returns (n*, M*, y*)
        """
        p = self.p
        # True markdown
        if mu_w is None:
            mu_w = p.wage_markdown(u)

        # Relative input price ratio (CES cost minimization)
        # Cost-minimizing ratio: n/M = (α_n/α_m) * (P_M/(w/μ^w))^θ
        relative_price = (w / mu_w) / P_M
        ratio_nm = (p.alpha_n / p.alpha_m) * relative_price ** (-p.theta)

        # From production function and relative demand:
        # Normalize output to 1 for factor intensity ratios
        # In equilibrium, scale by actual demand
        cost_index = (
            p.alpha_n * (w / mu_w) ** (1 - p.theta)
            + p.alpha_m * P_M ** (1 - p.theta)
        ) ** (1 / (1 - p.theta))

        # Optimal quantities (per unit output)
        n_star_unit = (p.alpha_n ** (1/p.theta)) * ((w/mu_w) / cost_index) ** (-p.theta)
        M_star_unit = (p.alpha_m ** (1/p.theta)) * (P_M / cost_index) ** (-p.theta)

        # Marginal cost of production
        mc = cost_index / (z * p.A)

        # Output from profit-maximizing pricing
        # P* = (μ^p / μ^w) * mc  (goods markup over true marginal cost)
        P_star = (p.epsilon_p / (p.epsilon_p - 1)) * mc

        # Scale: assume aggregate demand Y=1 for normalization
        y_star = 1.0
        n_star = n_star_unit * y_star
        M_star = M_star_unit * y_star

        return n_star, M_star, mc

    def phillips_curve_slope(self, mu_w: float, u: float) -> float:
        """
        Compute effective NKPC slope κ^mono.

        Standard slope: κ^comp = (1-ξ)(1-βξ)/ξ * (1-α_n)/α_n

        Monopsony-adjusted slope:
            κ^mono = κ^comp / (1 + γ/(ε̄(ε̄+1)) * (∂w/∂n)(n/w))

        The denominator > 1 when monopsony is present, so κ^mono < κ^comp.
        """
        p = self.p
        beta = 0.985  # Standard quarterly

        # Standard slope (log-linear Calvo NK)
        kappa_comp = (1 - p.xi) * (1 - beta * p.xi) / p.xi

        # Monopsony adjustment factor
        epsilon_u = p.epsilon_w_bar - p.gamma_w * (u - p.u_bar)
        epsilon_u = max(epsilon_u, 0.5)

        # Derivative of markup w.r.t. unemployment
        d_mu_w_d_u = -p.gamma_w / (epsilon_u + 1)**2

        # Elasticity of markup w.r.t. unemployment (negative)
        adjustment = 1 - (u / mu_w) * d_mu_w_d_u * (1 / p.alpha_n)

        return kappa_comp / adjustment

    def sacrifice_ratio(self, mu_w: float, u: float) -> float:
        """
        Sacrifice ratio: output cost per percentage point of disinflation.
        SR = 1 / κ (inverse of Phillips curve slope).
        """
        kappa = self.phillips_curve_slope(mu_w, u)
        return 1.0 / kappa if kappa > 0 else np.inf

    def markup_decomposition(self, u_grid: np.ndarray) -> dict:
        """
        Decompose total markup into goods-market and labor-market components
        across the business cycle.

        Total markup: μ^total = μ^p / μ^w(u)
        - μ^p = ε_p/(ε_p-1): goods market (constant)
        - μ^w(u) = ε(u)/(ε(u)+1): labor market (countercyclical)
        """
        p = self.p
        mu_p = p.mu_p
        mu_w_arr = np.array([p.wage_markdown(u) for u in u_grid])
        mu_total = mu_p / mu_w_arr

        return {
            'u': u_grid,
            'mu_p': np.full_like(u_grid, mu_p),
            'mu_w': mu_w_arr,
            'mu_total': mu_total,
            'labor_wedge': 1 / mu_w_arr,  # Excess markup from monopsony
        }


class MonopsonyMarkdown:
    """
    Analytic characterization of the monopsony wage markdown and its
    macroeconomic implications. Used for comparative statics figures.
    """

    @staticmethod
    def markdown_schedule(
        epsilon_bar: float = 4.0,
        gamma: float = 0.8,
        u_range: tuple = (0.02, 0.12),
        n_points: int = 100,
    ) -> dict:
        """
        Markdown schedule μ^w(u) over the business cycle.
        """
        u_grid = np.linspace(*u_range, n_points)
        u_bar = 0.05

        epsilon_u = np.maximum(epsilon_bar - gamma * (u_grid - u_bar), 0.5)
        mu_w = epsilon_u / (epsilon_u + 1)

        return {
            'u': u_grid,
            'mu_w': mu_w,
            'epsilon': epsilon_u,
            'labor_share_penalty': 1 - mu_w,  # Gap from competitive baseline
        }

    @staticmethod
    def cross_sectional_markdown(
        hhi_grid: np.ndarray,
        baseline_epsilon: float = 8.0,
        slope: float = -4.0 / 10000,
    ) -> np.ndarray:
        """
        Cross-sectional relationship between labor market HHI and wage markdown.
        Calibrated to Azar et al. (2022) estimates.

        ε(HHI) = ε_0 + slope * HHI  (markup deepens as market concentrates)
        """
        epsilon_hhi = np.maximum(baseline_epsilon + slope * hhi_grid, 0.5)
        return epsilon_hhi / (epsilon_hhi + 1)


if __name__ == '__main__':
    params = FirmParameters()
    solver = FirmSolver(params)

    print("=== Firm Block Diagnostics ===\n")
    print(f"Goods market markup μ^p: {params.mu_p:.4f}")
    print(f"SS wage markdown μ^w:    {params.steady_state_markdown():.4f}")
    print(f"Total markup μ^total:    {params.mu_p/params.steady_state_markdown():.4f}")

    u_grid = np.linspace(0.02, 0.12, 50)
    decomp = solver.markup_decomposition(u_grid)
    print(f"\nMarkdown range over business cycle:")
    print(f"  Low unemployment (u=2%): μ^w = {decomp['mu_w'][0]:.4f}")
    print(f"  High unemployment (u=12%): μ^w = {decomp['mu_w'][-1]:.4f}")

    print(f"\nPhillips curve slopes:")
    print(f"  Competitive:   κ = {solver.phillips_curve_slope(1.0, 0.05):.4f}")
    print(f"  SS monopsony:  κ = {solver.phillips_curve_slope(params.steady_state_markdown(), 0.05):.4f}")
    print(f"  Sacrifice ratio (competitive):  SR = {solver.sacrifice_ratio(1.0, 0.05):.2f}")
    print(f"  Sacrifice ratio (monopsonistic): SR = {solver.sacrifice_ratio(params.steady_state_markdown(), 0.05):.2f}")
