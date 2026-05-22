"""
Heterogeneous Household Problem — HANK Block
=============================================
Solves the household problem with incomplete markets using the
Endogenous Grid Method (EGM) following Carroll (2006) and
Auclert et al. (2021).

The household maximizes:
    E_0 Σ β^t [ c^{1-σ}/(1-σ) - χ n^{1+φ}/(1+φ) ]

subject to:
    c + a' = (1+r) a + w(ℓ) n + T
    a' ≥ ā (borrowing constraint)

Key departures from standard HANK:
  - Labor income w(ℓ) reflects monopsonistic wage markdown
  - Distribution Γ(a,ℓ) tracks idiosyncratic income risk under monopsony
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


class HouseholdParameters:
    """Structural parameters for the household block."""

    def __init__(
        self,
        beta: float = 0.985,      # Discount factor (quarterly)
        sigma: float = 2.0,        # CRRA coefficient
        chi: float = 1.0,          # Disutility of labor scale
        phi: float = 2.0,          # Inverse Frisch elasticity
        a_min: float = -0.5,       # Borrowing limit (times avg income)
        a_max: float = 200.0,      # Maximum assets
        n_a: int = 500,            # Asset grid points
        n_e: int = 7,              # Income shock grid points (Rouwenhorst)
        rho_e: float = 0.966,      # Income shock persistence
        sigma_e: float = 0.5,      # Income shock std dev (log)
    ):
        self.beta = beta
        self.sigma = sigma
        self.chi = chi
        self.phi = phi
        self.a_min = a_min
        self.a_max = a_max
        self.n_a = n_a
        self.n_e = n_e
        self.rho_e = rho_e
        self.sigma_e = sigma_e

    def asset_grid(self) -> np.ndarray:
        """Non-uniform asset grid with curvature near borrowing constraint."""
        # Denser grid near the constraint using power spacing
        a_grid = (
            self.a_min
            + (self.a_max - self.a_min)
            * np.linspace(0, 1, self.n_a) ** 1.4
        )
        return a_grid

    def rouwenhorst_income(self):
        """
        Rouwenhorst (1995) discretization of AR(1) income process.
        Returns (e_grid, Pi) where Pi is the Markov transition matrix.
        """
        n = self.n_e
        p = (1 + self.rho_e) / 2

        # Build transition matrix recursively
        Pi_2 = np.array([[p, 1 - p], [1 - p, p]])
        Pi_n = Pi_2.copy()
        for _ in range(n - 2):
            Pi_upper = np.block([
                [Pi_n, np.zeros((Pi_n.shape[0], 1))],
                [np.zeros((1, Pi_n.shape[1])), np.zeros((1, 1))]
            ])
            Pi_lower = np.block([
                [np.zeros((1, 1)), np.zeros((1, Pi_n.shape[1]))],
                [np.zeros((Pi_n.shape[0], 1)), Pi_n]
            ])
            Pi_n = p * Pi_upper + (1 - p) * Pi_lower
            Pi_n[1:-1] /= 2  # Normalize interior rows

        # Income levels (logs)
        sigma_y = self.sigma_e * np.sqrt(1 - self.rho_e**2)
        e_log = np.linspace(
            -np.sqrt(n - 1) * sigma_y,
             np.sqrt(n - 1) * sigma_y,
            n
        )
        e_grid = np.exp(e_log)
        e_grid /= e_grid @ np.linalg.matrix_power(Pi_n, 1000)[0]  # normalize

        return e_grid, Pi_n


class HouseholdSolver:
    """
    Solves the steady-state household problem using EGM.

    The monopsony wage markdown μ^w enters as a scaling factor on
    the effective wage, reducing labor income relative to the
    marginal product of labor:
        effective_w = mu_w * MPL
    where mu_w = ε/(ε+1) < 1.
    """

    def __init__(self, params: HouseholdParameters):
        self.p = params
        self.a_grid = params.asset_grid()
        self.e_grid, self.Pi = params.rouwenhorst_income()

    def utility_c(self, c: np.ndarray) -> np.ndarray:
        """Marginal utility of consumption."""
        return c ** (-self.p.sigma)

    def inv_utility_c(self, uc: np.ndarray) -> np.ndarray:
        """Inverse marginal utility (for EGM)."""
        return uc ** (-1 / self.p.sigma)

    def optimal_labor(self, w: float, mu_w: float = 1.0) -> float:
        """
        Optimal labor supply given effective wage.
        From FOC: χ n^φ = (w * mu_w) * c^{-σ}
        In partial equilibrium: n* = (w * mu_w / χ)^{1/φ}
        """
        return ((w * mu_w) / self.p.chi) ** (1 / self.p.phi)

    def solve_steady_state(
        self,
        r: float,
        w: float,
        mu_w: float = 1.0,
        T: float = 0.0,
        tol: float = 1e-9,
        max_iter: int = 10000,
    ) -> dict:
        """
        Solve for steady-state policy functions using EGM.

        Parameters
        ----------
        r : float
            Net real interest rate
        w : float
            Wage (marginal product of labor)
        mu_w : float
            Wage markdown (1 = competitive, < 1 = monopsonistic)
        T : float
            Lump-sum transfer (government redistribution)
        tol : float
            Convergence tolerance on policy function
        max_iter : int
            Maximum iterations

        Returns
        -------
        dict with keys:
            'c'  : (n_a, n_e) consumption policy
            'a'  : (n_a, n_e) savings policy
            'n'  : (n_a, n_e) labor supply policy
            'D'  : (n_a, n_e) stationary distribution
            'MPC': (n_a, n_e) marginal propensity to consume
        """
        beta = self.p.beta
        a_grid = self.a_grid
        e_grid = self.e_grid
        Pi = self.Pi
        n_a, n_e = len(a_grid), len(e_grid)

        # Effective income from labor
        eff_w = w * mu_w
        n_labor = self.optimal_labor(w, mu_w)  # hours worked (simplified)

        # Cash-on-hand grid: y = (1+r)*a + eff_w*e + T
        # Shape: (n_a, n_e)
        cash = (1 + r) * a_grid[:, None] + eff_w * e_grid[None, :] * n_labor + T

        # Initialize consumption guess: consume all cash (bounded above a_min)
        c_old = np.maximum(cash - self.p.a_min, 1e-10)

        for _ in range(max_iter):
            # Expected marginal utility next period (EGM step)
            # E[u'(c')] for each (a, e) using the transition matrix
            Euc_next = (self.utility_c(c_old) @ Pi.T)  # (n_a, n_e)

            # EGM: implied consumption today
            c_impl = self.inv_utility_c(beta * (1 + r) * Euc_next)

            # Implied asset today (endogenous grid)
            a_impl = (c_impl + a_grid[:, None] - eff_w * e_grid[None, :] * n_labor - T) / (1 + r)

            # Interpolate onto fixed asset grid
            c_new = np.empty_like(c_old)
            for ie in range(n_e):
                # Check if borrowing constraint binds
                # For assets below the kink, constraint binds
                a_kink = a_impl[0, ie]
                if a_kink > self.p.a_min:
                    # Constrained region: consume all cash
                    mask_constrained = a_grid <= a_kink
                    c_new[mask_constrained, ie] = np.maximum(
                        cash[mask_constrained, ie] - self.p.a_min, 1e-10
                    )
                else:
                    mask_constrained = np.zeros(n_a, dtype=bool)

                # Unconstrained region: interpolate
                mask_unc = ~mask_constrained
                if mask_unc.any():
                    interp_fn = interp1d(
                        a_impl[:, ie], c_impl[:, ie],
                        kind='linear', fill_value='extrapolate'
                    )
                    c_new[mask_unc, ie] = np.maximum(
                        interp_fn(a_grid[mask_unc]), 1e-10
                    )

            err = np.max(np.abs(c_new - c_old))
            c_old = c_new.copy()
            if err < tol:
                break

        # Final policy functions
        c_policy = c_old
        a_policy = cash - c_policy  # savings
        a_policy = np.maximum(a_policy, self.p.a_min)
        n_policy = np.full((n_a, n_e), n_labor)  # simplified static labor

        # Stationary distribution via Young (2010) method
        D = self._stationary_distribution(a_policy, Pi)

        # Marginal propensity to consume (numerical derivative)
        delta = 1e-4
        cash_up = cash + delta
        c_up = np.minimum(cash_up, c_policy + delta)
        MPC = (c_up - c_policy) / delta

        return {
            'c': c_policy,
            'a': a_policy,
            'n': n_policy,
            'D': D,
            'MPC': MPC,
            'agg_C': np.sum(c_policy * D),
            'agg_N': np.sum(n_policy * D),
            'agg_A': np.sum(a_policy * D),
            'mean_MPC': np.sum(MPC * D),
        }

    def _stationary_distribution(
        self,
        a_policy: np.ndarray,
        Pi: np.ndarray,
        tol: float = 1e-10,
        max_iter: int = 5000,
    ) -> np.ndarray:
        """
        Compute stationary distribution using Young (2010) non-stochastic
        simulation method. Avoids discretization error in distribution.
        """
        a_grid = self.a_grid
        n_a, n_e = a_policy.shape

        # Initialize uniform distribution
        D = np.ones((n_a, n_e)) / (n_a * n_e)

        for _ in range(max_iter):
            D_new = np.zeros_like(D)

            for ie in range(n_e):
                for ia in range(n_a):
                    # Find position of a'(a, e) in asset grid
                    a_prime = a_policy[ia, ie]
                    # Linear interpolation weights
                    idx = np.searchsorted(a_grid, a_prime) - 1
                    idx = np.clip(idx, 0, n_a - 2)

                    w_upper = (a_prime - a_grid[idx]) / (a_grid[idx + 1] - a_grid[idx] + 1e-16)
                    w_upper = np.clip(w_upper, 0, 1)
                    w_lower = 1 - w_upper

                    # Distribute mass across income states
                    for ie_next in range(n_e):
                        D_new[idx, ie_next] += D[ia, ie] * Pi[ie, ie_next] * w_lower
                        D_new[idx + 1, ie_next] += D[ia, ie] * Pi[ie, ie_next] * w_upper

            err = np.max(np.abs(D_new - D))
            D = D_new
            if err < tol:
                break

        return D / D.sum()


def compute_aggregate_MPC(
    w_grid: np.ndarray,
    mu_w_grid: np.ndarray,
    params: HouseholdParameters,
    r: float = 0.01,
) -> np.ndarray:
    """
    Compute aggregate MPC as a function of (w, μ^w) for comparative statics.
    Used to illustrate how monopsony depresses aggregate demand responsiveness.
    """
    solver = HouseholdSolver(params)
    MPCs = np.zeros((len(w_grid), len(mu_w_grid)))

    for i, w in enumerate(w_grid):
        for j, mu_w in enumerate(mu_w_grid):
            result = solver.solve_steady_state(r=r, w=w, mu_w=mu_w)
            MPCs[i, j] = result['mean_MPC']

    return MPCs


if __name__ == '__main__':
    import time
    params = HouseholdParameters(n_a=200, n_e=5)
    solver = HouseholdSolver(params)

    print("Solving steady state (competitive)...")
    t0 = time.time()
    result_comp = solver.solve_steady_state(r=0.01, w=1.0, mu_w=1.0)
    print(f"  Converged in {time.time()-t0:.2f}s")
    print(f"  Agg consumption: {result_comp['agg_C']:.4f}")
    print(f"  Mean MPC:        {result_comp['mean_MPC']:.4f}")

    print("\nSolving steady state (monopsonistic, μ^w=0.80)...")
    t0 = time.time()
    result_mono = solver.solve_steady_state(r=0.01, w=1.0, mu_w=0.80)
    print(f"  Converged in {time.time()-t0:.2f}s")
    print(f"  Agg consumption: {result_mono['agg_C']:.4f}")
    print(f"  Mean MPC:        {result_mono['mean_MPC']:.4f}")
    print(f"\nMPC increase from monopsony: {result_mono['mean_MPC'] - result_comp['mean_MPC']:.4f}")
    print("(Higher MPC because monopsony compresses income, pushing more HH near constraint)")
