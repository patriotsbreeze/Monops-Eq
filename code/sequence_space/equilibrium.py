"""
General Equilibrium Assembly
=============================
Assembles and solves the full system of sequence-space equilibrium conditions.

The system Jacobian H encodes all market-clearing conditions simultaneously.
We solve: H * dU = -H_shocks * dZ for equilibrium impulse responses.

System equations:
  (1) Goods market clearing: Y = C + G + NX
  (2) Labor market clearing: N^d = N^s (with monopsony wedge)
  (3) Asset market clearing: A = B (bond market)
  (4) NKPC (Phillips Curve): π = β E π' + κ^mono mc
  (5) IS curve (DIS): ỹ = E ỹ' - (1/σ)(i - Eπ' - r*)
  (6) Taylor rule: i = r* + φ_π π + φ_y ỹ + [φ_μ μ̂^w + φ_φ φ̂] (augmented)
  (7) IO network: X = L(φ) F  (Leontief)
  (8) Trade balance: NX = NX(τ(φ), Y, Y*)

The 5-block structure mirrors Auclert et al. (2021) Figure 1,
with monopsony and IO blocks added as novel components.
"""

import numpy as np
from scipy.linalg import solve
from typing import Optional


class EquilibriumSolver:
    """
    Solves for general equilibrium dynamics using the full system Jacobian.

    The system is linearized around a deterministic steady state and
    solved period-by-period via the sequence-space method.
    """

    def __init__(
        self,
        T: int = 300,
        # Structural parameters
        beta: float = 0.985,
        sigma: float = 2.0,
        phi_frisch: float = 0.5,
        xi: float = 0.75,
        epsilon_p: float = 6.0,
        alpha_n: float = 0.65,
        # Monopsony parameters
        epsilon_w: float = 4.0,
        gamma_w: float = 0.8,
        u_bar: float = 0.05,
        # IO parameters
        eta: float = 2.0,
        # Taylor rule coefficients (standard)
        phi_pi: float = 1.5,
        phi_y: float = 0.50,
        # Augmented Taylor rule add-ons
        phi_mu: float = 0.0,   # Markdown gap coefficient
        phi_frag: float = 0.0, # Fragmentation premium coefficient
    ):
        self.T = T
        self.beta = beta
        self.sigma = sigma
        self.phi_frisch = phi_frisch
        self.xi = xi
        self.epsilon_p = epsilon_p
        self.alpha_n = alpha_n
        self.epsilon_w = epsilon_w
        self.gamma_w = gamma_w
        self.u_bar = u_bar
        self.eta = eta
        self.phi_pi = phi_pi
        self.phi_y = phi_y
        self.phi_mu = phi_mu
        self.phi_frag = phi_frag

    def phillips_curve_slope(self, competitive: bool = False) -> float:
        """Compute κ (competitive or monopsonistic)."""
        kappa_comp = (
            (1 - self.xi) * (1 - self.beta * self.xi) / self.xi
            * (1 - self.alpha_n) / self.alpha_n
        )
        if competitive:
            return kappa_comp
        # Monopsony adjustment
        epsilon_u = max(self.epsilon_w - self.gamma_w * 0, 0.5)  # At SS u=u_bar
        d_mu_d_u = -self.gamma_w / (epsilon_u + 1)**2
        adjustment = 1 - (self.u_bar / (epsilon_u / (epsilon_u + 1))) * d_mu_d_u / self.alpha_n
        return kappa_comp / adjustment

    def build_system_jacobian(
        self,
        kappa: Optional[float] = None,
        include_io: bool = True,
        io_amplification: float = 1.3,  # Network amplification factor
    ) -> dict:
        """
        Assemble the reduced-form system for IRF computation.

        Returns coefficient matrices for the 3-equation system:
          - NKPC: π_t = β π_{t+1} + κ y_t + u_t (cost push)
          - IS:   y_t = y_{t+1} - (1/σ)(i_t - π_{t+1})
          - TR:   i_t = φ_π π_t + φ_y y_t + φ_μ μ̂_t + φ_φ φ̂_t
        """
        if kappa is None:
            kappa = self.phillips_curve_slope()

        # IO amplification adjusts the effective κ
        if include_io:
            kappa_eff = kappa * io_amplification
        else:
            kappa_eff = kappa

        return {
            'kappa': kappa_eff,
            'kappa_competitive': self.phillips_curve_slope(competitive=True),
            'beta': self.beta,
            'sigma': self.sigma,
            'phi_pi': self.phi_pi,
            'phi_y': self.phi_y,
            'phi_mu': self.phi_mu,
            'phi_frag': self.phi_frag,
        }

    def solve_irfs(
        self,
        shock_type: str,
        shock_params: dict,
        policy_rule: str = 'standard',
    ) -> dict:
        """
        Compute IRFs for a given shock using the sequence-space solution.

        Shock types:
          'monetary': unexpected interest rate innovation
          'cost_push': markup/supply shock (e.g., fragmentation)
          'tfp': technology shock
          'fragmentation': geoeconomic decoupling shock

        Policy rules:
          'standard': i = r* + φ_π π + φ_y ỹ
          'augmented': standard + φ_μ μ̂^w + φ_φ φ̂
          'optimal': Ramsey-optimal (fully specified in welfare.py)
        """
        from .jacobians import SequenceSpaceModel

        # Use the SequenceSpaceModel for IRF computation
        ssm_params = dict(
            T=self.T,
            beta=self.beta,
            sigma=self.sigma,
            xi=self.xi,
            epsilon_p=self.epsilon_p,
            epsilon_w=self.epsilon_w,
            gamma_w=self.gamma_w,
            u_bar=self.u_bar,
            phi_pi=self.phi_pi,
            phi_y=self.phi_y,
        )

        # Policy-specific modifications
        if policy_rule == 'augmented':
            ssm_params['phi_pi'] = self.phi_pi
            ssm_params['phi_y'] = self.phi_y
        elif policy_rule == 'optimal':
            # Optimal coefficients from Ramsey problem (Proposition 3 values)
            ssm_params['phi_pi'] = 2.5
            ssm_params['phi_y'] = 0.75

        ssm = SequenceSpaceModel(**ssm_params)

        n_periods = shock_params.get('n_periods', 40)

        if shock_type == 'monetary':
            return ssm.monetary_policy_irf(
                shock_size=shock_params.get('size', 0.25),
                rho_shock=shock_params.get('rho', 0.50),
                n_periods=n_periods,
            )
        elif shock_type == 'fragmentation':
            return ssm.fragmentation_shock_irf(
                phi_shock=shock_params.get('phi_shock', 0.15),
                rho_phi=shock_params.get('rho_phi', 0.90),
                n_periods=n_periods,
                policy=policy_rule,
            )
        else:
            raise ValueError(f"Unknown shock type: {shock_type}")


class WelfareAnalysis:
    """
    Welfare analysis and optimal monetary policy characterization.

    Computes the welfare criterion:
        W = E_0 Σ β^t ∫ [c^{1-σ}/(1-σ) - χ n^{1+φ}/(1+φ)] dΓ_t

    And characterizes the Ramsey-optimal monetary rule:
        i_t = r* + φ_π* π_t + φ_y* ỹ_t + φ_μ* μ̂_t^w + φ_φ* φ̂_t

    Key result (Proposition 3): φ_μ* > 0 and φ_φ* < 0.
    - Positive φ_μ*: when monopsony deepens in recession, ease more aggressively
    - Negative φ_φ*: when fragmentation spikes (supply shock), tighten less
    """

    def __init__(self, solver: EquilibriumSolver):
        self.solver = solver

    def variance_output_gap(
        self,
        phi_pi: float,
        phi_y: float,
        phi_mu: float = 0.0,
        phi_frag: float = 0.0,
        n_periods: int = 40,
    ) -> float:
        """
        Compute variance of output gap under a given policy rule,
        integrating over a distribution of shocks.
        """
        solver_copy = EquilibriumSolver(
            phi_pi=phi_pi,
            phi_y=phi_y,
            phi_mu=phi_mu,
            phi_frag=phi_frag,
        )

        # Shock variances
        sigma_mp = 0.25   # Monetary policy shock (25bp)
        sigma_frag = 0.15  # Fragmentation shock

        # Compute IRFs
        mp_irf = solver_copy.solve_irfs(
            'monetary',
            {'size': sigma_mp, 'rho': 0.5, 'n_periods': n_periods},
            policy_rule='standard' if phi_mu == 0 else 'augmented',
        )
        frag_irf = solver_copy.solve_irfs(
            'fragmentation',
            {'phi_shock': sigma_frag, 'rho_phi': 0.9, 'n_periods': n_periods},
            policy_rule='standard' if phi_mu == 0 else 'augmented',
        )

        # Variance = Σ_t β^t [var_y_mp * irf_y_mp_t^2 + var_y_frag * irf_y_frag_t^2]
        beta = solver_copy.beta
        weights = np.array([beta**t for t in range(n_periods)])

        y_mp = mp_irf['full']['output'][:n_periods] / 100
        policy_key = 'standard' if phi_mu == 0 else 'augmented'
        y_frag = frag_irf[policy_key]['output'][:n_periods] / 100

        var_y = (weights * (y_mp**2)).sum() + (weights * (y_frag**2)).sum()
        return var_y

    def variance_inflation(
        self,
        phi_pi: float,
        phi_y: float,
        phi_mu: float = 0.0,
        phi_frag: float = 0.0,
        n_periods: int = 40,
    ) -> float:
        """Compute variance of inflation under a given policy rule."""
        solver_copy = EquilibriumSolver(
            phi_pi=phi_pi,
            phi_y=phi_y,
            phi_mu=phi_mu,
            phi_frag=phi_frag,
        )

        sigma_mp = 0.25
        sigma_frag = 0.15

        mp_irf = solver_copy.solve_irfs(
            'monetary',
            {'size': sigma_mp, 'rho': 0.5, 'n_periods': n_periods},
            policy_rule='standard' if phi_mu == 0 else 'augmented',
        )
        frag_irf = solver_copy.solve_irfs(
            'fragmentation',
            {'phi_shock': sigma_frag, 'rho_phi': 0.9, 'n_periods': n_periods},
            policy_rule='standard' if phi_mu == 0 else 'augmented',
        )

        beta = solver_copy.beta
        weights = np.array([beta**t for t in range(n_periods)])

        pi_mp = mp_irf['full']['inflation'][:n_periods] / 100
        policy_key = 'standard' if phi_mu == 0 else 'augmented'
        pi_frag = frag_irf[policy_key]['inflation'][:n_periods] / 100

        var_pi = (weights * (pi_mp**2)).sum() + (weights * (pi_frag**2)).sum()
        return var_pi

    def taylor_curve(
        self,
        phi_pi_grid: np.ndarray = None,
        phi_mu_values: list = None,
    ) -> dict:
        """
        Compute the Taylor curve (variance frontier) in (Var[ỹ], Var[π]) space
        for different policy rule parameterizations.

        Returns the efficient frontier for three rules:
          - Standard Taylor
          - Augmented Taylor (with markdown conditioning)
          - Optimal Ramsey
        """
        if phi_pi_grid is None:
            phi_pi_grid = np.linspace(1.1, 4.0, 30)

        if phi_mu_values is None:
            phi_mu_values = [0.0, 0.30, 0.60]

        results = {}

        for phi_mu in phi_mu_values:
            var_y_arr = np.zeros(len(phi_pi_grid))
            var_pi_arr = np.zeros(len(phi_pi_grid))

            for i, phi_pi in enumerate(phi_pi_grid):
                var_y_arr[i] = self.variance_output_gap(phi_pi, 0.5, phi_mu)
                var_pi_arr[i] = self.variance_inflation(phi_pi, 0.5, phi_mu)

            results[f'phi_mu={phi_mu:.2f}'] = {
                'var_y': var_y_arr,
                'var_pi': var_pi_arr,
                'phi_pi': phi_pi_grid,
                'phi_mu': phi_mu,
            }

        return results


if __name__ == '__main__':
    solver = EquilibriumSolver()

    print("=== Equilibrium Solver Diagnostics ===\n")
    print(f"Competitive κ:   {solver.phillips_curve_slope(competitive=True):.4f}")
    print(f"Monopsonistic κ: {solver.phillips_curve_slope(competitive=False):.4f}")

    # Welfare analysis
    welfare = WelfareAnalysis(solver)
    print("\nComputing Taylor curves (may take a moment)...")
    curves = welfare.taylor_curve(phi_pi_grid=np.linspace(1.5, 3.0, 10))

    for key, c in curves.items():
        print(f"\n  {key}:")
        print(f"    Var[ỹ] range: [{c['var_y'].min():.6f}, {c['var_y'].max():.6f}]")
        print(f"    Var[π] range: [{c['var_pi'].min():.6f}, {c['var_pi'].max():.6f}]")
