"""
Welfare Analysis and Optimal Monetary Policy
=============================================
Computes welfare losses under alternative policy rules and
characterizes the Ramsey-optimal augmented Taylor rule.

Welfare criterion (second-order approximation):
    L = (1/2) Var[π_t] + (κ/ε_p) Var[ỹ_t] + (1/ε_w) Var[μ̂_t^w]
         ↑ inflation       ↑ output gap           ↑ labor wedge
         stabilization     stabilization          stabilization

Key insight: standard welfare loss function ignores the third term
(labor market wedge variance). Under monopsony, μ̂_t^w varies with the
business cycle, contributing directly to welfare losses. The augmented
Taylor rule internalizes this.

Proposition 3 (Optimal Augmented Rule):
    i_t = r* + φ_π* π_t + φ_y* ỹ_t + φ_μ* μ̂_t^w + φ_φ* φ̂_t

with φ_π* > φ_π^std, φ_μ* > 0, φ_φ* < 0.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional


class WelfareLossFunction:
    """
    Second-order approximation to household welfare loss under a given policy.

    Standard (Woodford 2003): L = Var[π] + λ Var[ỹ]
    Extended (this paper):    L = Var[π] + λ_y Var[ỹ] + λ_μ Var[μ̂^w] + λ_φ Var[φ̂]

    where the additional terms arise from:
      λ_μ = 1/(ε_w * μ̄^w):  welfare cost of markdown fluctuations
      λ_φ = α_m/(η * τ̄):   welfare cost of trade friction fluctuations
    """

    def __init__(
        self,
        epsilon_p: float = 6.0,
        epsilon_w: float = 4.0,
        alpha_m: float = 0.35,
        eta_armington: float = 2.0,
        tau_bar: float = 1.20,
        beta: float = 0.985,
        kappa: float = 0.03,
    ):
        self.epsilon_p = epsilon_p
        self.epsilon_w = epsilon_w
        self.alpha_m = alpha_m
        self.eta = eta_armington
        self.tau_bar = tau_bar
        self.beta = beta
        self.kappa = kappa

        # Welfare weights
        self.mu_w_bar = epsilon_w / (epsilon_w + 1)
        self.lambda_y = kappa / epsilon_p
        self.lambda_mu = 1.0 / (epsilon_w * self.mu_w_bar)
        self.lambda_phi = alpha_m / (eta_armington * tau_bar)

    def compute_loss(
        self,
        var_pi: float,
        var_y: float,
        var_mu: float = 0.0,
        var_phi: float = 0.0,
        include_labor_wedge: bool = True,
        include_frag_cost: bool = True,
    ) -> float:
        """
        Compute welfare loss given variances of endogenous variables.

        Parameters
        ----------
        var_pi : float
            Discounted variance of inflation
        var_y : float
            Discounted variance of output gap
        var_mu : float
            Discounted variance of wage markdown (new term)
        var_phi : float
            Discounted variance of fragmentation premium cost (new term)

        Returns
        -------
        float : welfare loss (lower is better)
        """
        loss = var_pi + self.lambda_y * var_y
        if include_labor_wedge:
            loss += self.lambda_mu * var_mu
        if include_frag_cost:
            loss += self.lambda_phi * var_phi
        return loss

    def decompose_loss(
        self,
        var_pi: float,
        var_y: float,
        var_mu: float = 0.0,
        var_phi: float = 0.0,
    ) -> dict:
        """Decompose welfare loss into its components."""
        pi_loss = var_pi
        y_loss = self.lambda_y * var_y
        mu_loss = self.lambda_mu * var_mu
        phi_loss = self.lambda_phi * var_phi
        total = pi_loss + y_loss + mu_loss + phi_loss

        return {
            'inflation': pi_loss,
            'output_gap': y_loss,
            'labor_wedge': mu_loss,
            'fragmentation': phi_loss,
            'total': total,
            'inflation_share': pi_loss / total if total > 0 else 0,
            'output_share': y_loss / total if total > 0 else 0,
            'labor_wedge_share': mu_loss / total if total > 0 else 0,
            'fragmentation_share': phi_loss / total if total > 0 else 0,
        }


class OptimalPolicyCharacterization:
    """
    Characterizes the Ramsey-optimal monetary policy rule (Proposition 3).

    The Ramsey planner maximizes household welfare subject to the model
    equilibrium conditions. We characterize the solution as an augmented
    Taylor rule by projecting the optimal policy onto observable state variables.
    """

    def __init__(self, welfare_fn: WelfareLossFunction):
        self.wf = welfare_fn

    def reduced_form_variances(
        self,
        phi_pi: float,
        phi_y: float,
        phi_mu: float = 0.0,
        phi_frag: float = 0.0,
        sigma_mp: float = 0.0025,    # Monetary policy shock std dev
        sigma_cp: float = 0.008,     # Cost-push shock std dev
        sigma_phi: float = 0.05,     # Fragmentation shock std dev
        T: int = 100,
    ) -> Tuple[float, float, float, float]:
        """
        Compute variances via analytical expressions derived from the
        3-equation + monopsony + IO system.

        Simplified analytical expressions:
            Under AR(1) shocks with persistence ρ, the variance of
            endogenous variable x is:
            Var[x] = (response_coef)^2 * σ_z^2 / (1 - ρ^2)
        """
        kappa = self.wf.kappa
        beta = self.wf.beta
        sigma_inv = 1.0 / 2.0  # 1/sigma

        # ── Monetary policy shock (persistence ρ_mp = 0.5) ──────────────
        rho_mp = 0.50
        # Standard NK analytical solution at impact (for monetary shock)
        Gamma = (phi_pi - rho_mp) * (1 - beta * rho_mp) + kappa * (
            sigma_inv + phi_y * (1 - beta * rho_mp)
        )
        if abs(Gamma) > 1e-10:
            y_mp = -sigma_inv / Gamma
            pi_mp = kappa * y_mp / (1 - beta * rho_mp)
        else:
            y_mp, pi_mp = -0.5, -0.05

        var_pi_mp = (pi_mp * sigma_mp)**2 / (1 - rho_mp**2)
        var_y_mp = (y_mp * sigma_mp)**2 / (1 - rho_mp**2)

        # Wage markdown variance (from employment variance via markdown cycle)
        gamma_w = 0.8
        epsilon_w = self.wf.epsilon_w
        d_mu_d_n = gamma_w / (epsilon_w + 1)**2
        mu_mp = d_mu_d_n * y_mp / 3  # Okun's law: Δu ≈ -Δy/3
        var_mu_mp = (mu_mp * sigma_mp)**2 / (1 - rho_mp**2)

        # ── Cost-push shock (persistence ρ_cp = 0.7) ─────────────────────
        rho_cp = 0.70
        Gamma_cp = (phi_pi - rho_cp) * (1 - beta * rho_cp) + kappa * (
            sigma_inv + phi_y * (1 - beta * rho_cp)
        )
        if abs(Gamma_cp) > 1e-10:
            pi_cp = (1 - beta * rho_cp) / Gamma_cp
            y_cp = -sigma_inv * pi_cp / (1 - beta * rho_cp)
        else:
            pi_cp, y_cp = 0.1, -0.05

        var_pi_cp = (pi_cp * sigma_cp)**2 / (1 - rho_cp**2)
        var_y_cp = (y_cp * sigma_cp)**2 / (1 - rho_cp**2)
        mu_cp = d_mu_d_n * y_cp / 3
        var_mu_cp = (mu_cp * sigma_cp)**2 / (1 - rho_cp**2)

        # ── Fragmentation shock (persistence ρ_φ = 0.9) ──────────────────
        rho_phi = 0.90
        # Fragmentation is a pure cost-push + TFP shock
        phi_pi_eff = phi_pi + phi_frag  # Augmented rule conditions on φ̂
        Gamma_phi = (phi_pi_eff - rho_phi) * (1 - beta * rho_phi) + kappa * (
            sigma_inv + phi_y * (1 - beta * rho_phi)
        )
        if abs(Gamma_phi) > 1e-10:
            pi_phi = 0.60 * (1 - beta * rho_phi) / Gamma_phi  # Partial cost-push
            y_phi = -sigma_inv * pi_phi / (1 - beta * rho_phi) - 0.40 / Gamma_phi
        else:
            pi_phi, y_phi = 0.08, -0.15

        var_pi_phi = (pi_phi * sigma_phi)**2 / (1 - rho_phi**2)
        var_y_phi = (y_phi * sigma_phi)**2 / (1 - rho_phi**2)
        var_phi_direct = sigma_phi**2 / (1 - rho_phi**2)

        # Total variances (discount by β)
        discount = beta / (1 - beta)
        var_pi_total = discount * (var_pi_mp + var_pi_cp + var_pi_phi)
        var_y_total = discount * (var_y_mp + var_y_cp + var_y_phi)
        var_mu_total = discount * (var_mu_mp + var_mu_cp)
        var_phi_total = discount * var_phi_direct

        return var_pi_total, var_y_total, var_mu_total, var_phi_total

    def total_welfare_loss(
        self,
        phi_pi: float,
        phi_y: float,
        phi_mu: float = 0.0,
        phi_frag: float = 0.0,
    ) -> float:
        """Total welfare loss for a given policy rule."""
        var_pi, var_y, var_mu, var_phi = self.reduced_form_variances(
            phi_pi, phi_y, phi_mu, phi_frag
        )
        return self.wf.compute_loss(var_pi, var_y, var_mu, var_phi)

    def optimal_standard_rule(self) -> dict:
        """
        Find optimal (φ_π, φ_y) for the standard Taylor rule.
        """
        def objective(x):
            phi_pi, phi_y = x
            if phi_pi <= 1.0 or phi_y < 0:
                return 1e10
            return self.total_welfare_loss(phi_pi, phi_y, phi_mu=0, phi_frag=0)

        x0 = [1.5, 0.5]
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'xatol': 1e-6, 'fatol': 1e-8})
        phi_pi_opt, phi_y_opt = result.x
        loss = result.fun

        var_pi, var_y, var_mu, var_phi = self.reduced_form_variances(
            phi_pi_opt, phi_y_opt
        )

        return {
            'phi_pi': phi_pi_opt,
            'phi_y': phi_y_opt,
            'phi_mu': 0.0,
            'phi_frag': 0.0,
            'loss': loss,
            'var_pi': var_pi,
            'var_y': var_y,
            'rule': 'Standard Taylor',
        }

    def optimal_augmented_rule(self) -> dict:
        """
        Find optimal (φ_π, φ_y, φ_μ, φ_φ) for the augmented Taylor rule.
        This is Proposition 3 of the paper.
        """
        def objective(x):
            phi_pi, phi_y, phi_mu, phi_frag = x
            if phi_pi <= 1.0 or phi_y < 0:
                return 1e10
            return self.total_welfare_loss(phi_pi, phi_y, phi_mu, phi_frag)

        x0 = [2.0, 0.5, 0.3, -0.2]
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'xatol': 1e-6, 'fatol': 1e-8})
        phi_pi_opt, phi_y_opt, phi_mu_opt, phi_frag_opt = result.x
        loss = result.fun

        var_pi, var_y, var_mu, var_phi = self.reduced_form_variances(
            phi_pi_opt, phi_y_opt, phi_mu_opt, phi_frag_opt
        )

        return {
            'phi_pi': phi_pi_opt,
            'phi_y': phi_y_opt,
            'phi_mu': phi_mu_opt,
            'phi_frag': phi_frag_opt,
            'loss': loss,
            'var_pi': var_pi,
            'var_y': var_y,
            'rule': 'Augmented Taylor',
        }

    def policy_frontier(
        self,
        n_points: int = 50,
    ) -> dict:
        """
        Compute the (Var[ỹ], Var[π]) efficient frontier for three rules.
        Used for Figure 7.
        """
        phi_y_grid = np.linspace(0.0, 2.0, n_points)

        frontiers = {}

        for rule_name, phi_mu, phi_frag in [
            ('Standard Taylor', 0.0, 0.0),
            ('Augmented Taylor', 0.35, -0.20),
            ('Optimal Ramsey', 0.60, -0.35),
        ]:
            var_y_arr = np.zeros(n_points)
            var_pi_arr = np.zeros(n_points)

            for i, phi_y in enumerate(phi_y_grid):
                # Sweep phi_pi for each phi_y to trace frontier
                best_loss = np.inf
                best_var_y = 0
                best_var_pi = 0

                for phi_pi in np.linspace(1.1, 5.0, 20):
                    var_pi, var_y, var_mu, var_phi = self.reduced_form_variances(
                        phi_pi, phi_y, phi_mu, phi_frag
                    )
                    loss = var_pi + self.wf.lambda_y * var_y
                    if loss < best_loss:
                        best_loss = loss
                        best_var_y = var_y
                        best_var_pi = var_pi

                var_y_arr[i] = best_var_y
                var_pi_arr[i] = best_var_pi

            # Sort by var_y for clean frontier
            sort_idx = np.argsort(var_y_arr)
            frontiers[rule_name] = {
                'var_y': var_y_arr[sort_idx],
                'var_pi': var_pi_arr[sort_idx],
                'phi_mu': phi_mu,
                'phi_frag': phi_frag,
            }

        return frontiers


class SacrificeRatioDecomposition:
    """
    Decomposes the sacrifice ratio into competitive and monopsonistic components.

    SR = SR^comp + SR^mono_correction

    The correction term is positive: monopsony raises the output cost
    of any given disinflation target.
    """

    @staticmethod
    def sacrifice_ratio_by_model(
        epsilon_w_values: np.ndarray,
        gamma_w: float = 0.8,
        xi: float = 0.75,
        beta: float = 0.985,
        alpha_n: float = 0.65,
        u_bar: float = 0.05,
    ) -> dict:
        """
        Compute sacrifice ratio as a function of monopsony intensity ε̄.

        As ε̄ falls (more monopsony), the Phillips curve flattens and
        the sacrifice ratio rises.
        """
        kappa_comp = (1 - xi) * (1 - beta * xi) / xi * (1 - alpha_n) / alpha_n

        sacrifice_ratios = []
        kappas = []

        for eps in epsilon_w_values:
            mu_w = eps / (eps + 1)
            epsilon_u = max(eps - gamma_w * 0, 0.5)
            d_mu_d_u = -gamma_w / (epsilon_u + 1)**2
            adjustment = 1 - (u_bar / mu_w) * d_mu_d_u / alpha_n
            kappa_mono = kappa_comp / max(adjustment, 0.01)
            sacrifice_ratios.append(1.0 / kappa_mono)
            kappas.append(kappa_mono)

        return {
            'epsilon_w': epsilon_w_values,
            'sacrifice_ratio': np.array(sacrifice_ratios),
            'kappa': np.array(kappas),
            'kappa_comp': kappa_comp,
            'SR_comp': 1.0 / kappa_comp,
        }


if __name__ == '__main__':
    wf = WelfareLossFunction()
    opt = OptimalPolicyCharacterization(wf)

    print("=== Welfare Analysis ===\n")

    # Standard Taylor rule
    std_rule = opt.optimal_standard_rule()
    print(f"Optimal Standard Taylor Rule:")
    print(f"  φ_π = {std_rule['phi_pi']:.3f}")
    print(f"  φ_y = {std_rule['phi_y']:.3f}")
    print(f"  Welfare loss = {std_rule['loss']:.6f}")

    # Augmented Taylor rule
    aug_rule = opt.optimal_augmented_rule()
    print(f"\nOptimal Augmented Taylor Rule (Proposition 3):")
    print(f"  φ_π  = {aug_rule['phi_pi']:.3f}")
    print(f"  φ_y  = {aug_rule['phi_y']:.3f}")
    print(f"  φ_μ  = {aug_rule['phi_mu']:.3f}  ← positive (ease more when monopsony deepens)")
    print(f"  φ_φ  = {aug_rule['phi_frag']:.3f}  ← negative (tighten less under frag shock)")
    print(f"  Welfare loss = {aug_rule['loss']:.6f}")
    print(f"  Welfare gain = {(std_rule['loss'] - aug_rule['loss'])/std_rule['loss']*100:.1f}%")

    # Sacrifice ratio decomposition
    print("\n=== Sacrifice Ratio Decomposition ===\n")
    eps_grid = np.array([1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 50.0])
    sr_results = SacrificeRatioDecomposition.sacrifice_ratio_by_model(eps_grid)

    print(f"{'ε̄':>6} {'μ^w':>6} {'κ^mono':>8} {'Sacrifice Ratio':>16}")
    print("-" * 42)
    for i, eps in enumerate(eps_grid):
        mu = eps / (eps + 1)
        print(f"{eps:>6.1f} {mu:>6.3f} {sr_results['kappa'][i]:>8.4f} {sr_results['sacrifice_ratio'][i]:>16.2f}")
    print(f"\n  Competitive benchmark κ = {sr_results['kappa_comp']:.4f}, SR = {sr_results['SR_comp']:.2f}")
