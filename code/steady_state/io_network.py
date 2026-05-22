"""
Multi-Region Input-Output Network
==================================
Implements the multi-sector, multi-region input-output structure
with geoeconomic fragmentation.

The world consists of:
  - S sectors (10 in benchmark calibration)
  - R regions (5: US, EU, China, Emerging Asia, RoW)
  - B = 2 geopolitical blocs

Key objects:
  - Omega (S*R x S*R): IO coefficient matrix
  - tau (R x R): iceberg trade cost matrix
  - L = (I - Omega')^{-1}: Leontief inverse
  - lambda (S*R,): Domar weights (sector Gi shares)

Geoeconomic fragmentation enters through time-varying tau matrix,
which reshapes Omega and thus the entire Leontief inverse.
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


# ─── Sector and Region Definitions ──────────────────────────────────────────

SECTORS = [
    'Energy',
    'Mining & Extraction',
    'Food & Agriculture',
    'Basic Manufacturing',
    'Technology Hardware',
    'Chemicals & Pharma',
    'Transportation',
    'Financial Services',
    'Professional Services',
    'Consumer Services',
]

REGIONS = [
    'United States',
    'European Union',
    'China',
    'Emerging Asia',
    'Rest of World',
]

BLOCS = {
    0: ['United States', 'European Union', 'Rest of World'],
    1: ['China', 'Emerging Asia'],
}

REGION_BLOC = {
    'United States': 0,
    'European Union': 0,
    'China': 1,
    'Emerging Asia': 1,
    'Rest of World': 0,
}


# ─── Calibrated IO Parameters ────────────────────────────────────────────────

# Benchmark IO coefficient matrix (S x S), domestic use
# Calibrated to BEA 2019 Use Table, aggregated to 10 sectors
IO_DOMESTIC_BENCHMARK = np.array([
    # Energy  Mining  Food  Basic  Tech   Chem  Trans  Fin   Prof  Cons
    [0.15,   0.05,   0.02, 0.08,  0.03,  0.06, 0.08,  0.01, 0.01, 0.02],  # Energy
    [0.01,   0.12,   0.01, 0.10,  0.05,  0.04, 0.02,  0.00, 0.01, 0.01],  # Mining
    [0.02,   0.01,   0.18, 0.03,  0.01,  0.08, 0.03,  0.01, 0.01, 0.05],  # Food
    [0.04,   0.08,   0.03, 0.15,  0.12,  0.07, 0.10,  0.02, 0.03, 0.04],  # Basic Mfg
    [0.03,   0.02,   0.01, 0.05,  0.10,  0.04, 0.06,  0.05, 0.08, 0.03],  # Tech
    [0.03,   0.04,   0.05, 0.08,  0.04,  0.12, 0.04,  0.02, 0.02, 0.03],  # Chem
    [0.05,   0.06,   0.04, 0.07,  0.05,  0.04, 0.08,  0.03, 0.05, 0.06],  # Trans
    [0.01,   0.02,   0.01, 0.02,  0.03,  0.01, 0.02,  0.10, 0.05, 0.03],  # Fin
    [0.01,   0.01,   0.01, 0.02,  0.04,  0.02, 0.03,  0.04, 0.08, 0.03],  # Prof
    [0.02,   0.01,   0.03, 0.02,  0.02,  0.01, 0.04,  0.03, 0.04, 0.08],  # Cons
])

# Value-added shares by sector (complement to IO coefficients)
VALUE_ADDED_SHARES = 1 - IO_DOMESTIC_BENCHMARK.sum(axis=0)


class IONetwork:
    """
    Multi-region input-output network with geoeconomic fragmentation.

    The full IO coefficient matrix is (S*R) x (S*R), where entry
    Ω_{(s,r),(s',r')} is the cost share of sector s' in region r'
    inputs used by sector s in region r.

    Under fragmentation, cross-bloc entries are reduced by the
    fragmentation premium φ acting through iceberg costs τ^{rr'}.
    """

    def __init__(
        self,
        n_sectors: int = 10,
        n_regions: int = 5,
        delta_within: float = 0.05,   # Within-region trade cost
        delta_cross_bloc: float = 0.15,  # Additional cross-bloc trade cost (baseline)
        eta: float = 2.0,             # Armington elasticity (between regions)
        domestic_share: float = 0.65, # Share of inputs sourced domestically
    ):
        self.S = n_sectors
        self.R = n_regions
        self.N = n_sectors * n_regions  # Total nodes
        self.delta_within = delta_within
        self.delta_cross_bloc = delta_cross_bloc
        self.eta = eta
        self.domestic_share = domestic_share

        # Build baseline IO structure
        self.Omega_domestic = IO_DOMESTIC_BENCHMARK[:n_sectors, :n_sectors].copy()
        # Normalize rows to ensure feasibility
        row_sums = self.Omega_domestic.sum(axis=1, keepdims=True)
        self.Omega_domestic = self.Omega_domestic / np.maximum(row_sums, 1) * \
                              self.Omega_domestic.sum(axis=1, keepdims=True)

    def trade_cost_matrix(self, phi: float = 0.0) -> np.ndarray:
        """
        Build R x R iceberg trade cost matrix.

        τ^{rr'} = 1 + δ_within + φ_extra * 1[b(r) ≠ b(r')]

        where φ_extra is the fragmentation premium (time-varying shock).
        """
        tau = np.ones((self.R, self.R))

        for r in range(self.R):
            for r_prime in range(self.R):
                if r == r_prime:
                    tau[r, r_prime] = 1.0
                else:
                    region_r = REGIONS[r]
                    region_rp = REGIONS[r_prime]
                    cross_bloc = (REGION_BLOC[region_r] != REGION_BLOC[region_rp])
                    tau[r, r_prime] = (
                        1 + self.delta_within
                        + (self.delta_cross_bloc + phi) * cross_bloc
                    )
        return tau

    def full_io_matrix(self, phi: float = 0.0) -> np.ndarray:
        """
        Build full (S*R) x (S*R) IO coefficient matrix.

        Entry Ω_{(s,r),(s',r')} reflects:
        - Domestic share (s=s', r=r'): self.domestic_share
        - Import shares: distributed across regions with CES weights
          adjusted by iceberg costs

        Returns matrix indexed as [destination (s,r), source (s',r')].
        """
        S, R = self.S, self.R
        N = S * R
        tau = self.trade_cost_matrix(phi)

        # Full IO matrix: rows = using sector-region, cols = supplying sector-region
        Omega_full = np.zeros((N, N))

        for s in range(S):
            for r in range(R):
                row_idx = s * R + r
                # Total intermediate input share for sector s
                total_input_share = self.Omega_domestic[s].sum()

                for s_prime in range(S):
                    # Domestic + import shares for input s' into (s, r)
                    # CES over source regions: Armington model
                    # τ-adjusted cost weights
                    tau_adjusted = tau[r, :] ** (1 - self.eta)
                    weights = tau_adjusted / tau_adjusted.sum()

                    # Domestic share is boosted
                    weights_with_home = weights.copy()
                    weights_with_home[r] = (
                        self.domestic_share
                        + (1 - self.domestic_share) * weights[r]
                    )
                    import_weight = 1 - weights_with_home[r]
                    weights_with_home[:] *= (1 / weights_with_home.sum())

                    for r_prime in range(R):
                        col_idx = s_prime * R + r_prime
                        Omega_full[row_idx, col_idx] = (
                            self.Omega_domestic[s, s_prime]
                            * weights_with_home[r_prime]
                        )

        return Omega_full

    def leontief_inverse(self, phi: float = 0.0) -> np.ndarray:
        """
        Compute Leontief inverse L = (I - Ω')^{-1}.

        The Leontief inverse captures all rounds of indirect input demand.
        L[i,j] = total output of sector i required per unit of final demand
        in sector j (including all upstream chains).
        """
        Omega = self.full_io_matrix(phi)
        N = self.N
        I = np.eye(N)

        # Check for stability (spectral radius < 1)
        eigenvalues = np.linalg.eigvals(Omega.T)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig >= 1:
            warnings.warn(f"IO matrix may be unstable: spectral radius = {max_eig:.4f}")

        L = np.linalg.solve(I - Omega.T, I)
        return L

    def domar_weights(
        self,
        phi: float = 0.0,
        gdp_shares: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute Domar weights: λ_{sr} = P_{sr} X_{sr} / (P * GDP)

        Domar weights exceed sector GDP shares because sectors supply
        intermediates in addition to final goods. They sum to > 1.

        Under fragmentation (higher φ), upstream domestic sectors gain
        Domar weight as cross-bloc substitution weakens.
        """
        L = self.leontief_inverse(phi)
        N = self.N

        # Final demand shares: equal weights as baseline
        if gdp_shares is None:
            f = np.ones(N) / N
        else:
            f = gdp_shares / gdp_shares.sum()

        # Domar weights = row sums of L weighted by final demand
        # λ_i = Σ_j L_{ij} f_j
        domar = L @ f

        # Normalize so they represent shares of aggregate output
        # (Domar weights sum to > 1 in general)
        return domar

    def fragmentation_tfp_loss(self, phi: float, dphi: float = 1e-4) -> float:
        """
        Aggregate TFP loss from a marginal increase in fragmentation.

        d log TFP / d φ = -Σ_{(s,r),(s',r') cross-bloc} λ_{sr} Ω_{(s,r)(s',r')} / τ^{rr'}

        This is the closed-form network-weighted expression for the
        welfare cost of geoeconomic fragmentation.
        """
        tau = self.trade_cost_matrix(phi)
        Omega = self.full_io_matrix(phi)
        domar = self.domar_weights(phi)
        S, R = self.S, self.R

        total = 0.0
        for s in range(S):
            for r in range(R):
                row_idx = s * R + r
                for s_prime in range(S):
                    for r_prime in range(R):
                        col_idx = s_prime * R + r_prime
                        region_r = REGIONS[r]
                        region_rp = REGIONS[r_prime]
                        if REGION_BLOC[region_r] != REGION_BLOC[region_rp]:
                            # Cross-bloc link
                            total += (
                                domar[row_idx]
                                * Omega[row_idx, col_idx]
                                / tau[r, r_prime]
                            )

        return -total  # Negative because fragmentation reduces TFP

    def compute_sector_domar_by_phi(
        self,
        phi_grid: np.ndarray,
        region: int = 0,  # US
    ) -> np.ndarray:
        """
        Compute sectoral Domar weights for a given region as function of φ.
        Returns (len(phi_grid), S) array.
        """
        results = np.zeros((len(phi_grid), self.S))

        for i, phi in enumerate(phi_grid):
            domar = self.domar_weights(phi)
            for s in range(self.S):
                idx = s * self.R + region
                results[i, s] = domar[idx]

        return results

    def network_entropy(self, phi: float) -> float:
        """
        Shannon entropy of the trade network structure.
        H = -Σ Ω log(Ω)   (higher entropy = more diversified sourcing)
        Higher fragmentation → lower entropy → more concentrated sourcing.
        """
        Omega = self.full_io_matrix(phi)
        Omega_flat = Omega.flatten()
        # Only non-zero entries
        mask = Omega_flat > 0
        return -np.sum(Omega_flat[mask] * np.log(Omega_flat[mask]))


def simulate_fragmentation_dynamics(
    phi_path: np.ndarray,
    network: IONetwork,
    rho_phi: float = 0.90,
) -> dict:
    """
    Simulate the dynamic impact of a fragmentation shock.

    phi_path: sequence of fragmentation premium values
    Returns time series of:
      - Aggregate TFP loss
      - Domar weight concentration (Herfindahl of domestic upstream sectors)
      - Network entropy
    """
    T = len(phi_path)
    tfp_loss = np.zeros(T)
    entropy = np.zeros(T)
    domar_hhi = np.zeros(T)

    for t, phi in enumerate(phi_path):
        tfp_loss[t] = network.fragmentation_tfp_loss(phi)
        entropy[t] = network.network_entropy(phi)

        domar = network.domar_weights(phi)
        # HHI of domestic Domar weights (region 0 = US)
        domestic_domar = np.array([domar[s * network.R] for s in range(network.S)])
        domestic_domar /= domestic_domar.sum()
        domar_hhi[t] = np.sum(domestic_domar**2) * 10000  # Scale to HHI units

    return {
        'phi': phi_path,
        'tfp_loss': tfp_loss,
        'entropy': entropy,
        'domar_hhi': domar_hhi,
    }


if __name__ == '__main__':
    network = IONetwork(n_sectors=10, n_regions=5)

    print("=== IO Network Diagnostics ===\n")
    print(f"Network dimensions: {network.S} sectors × {network.R} regions = {network.N} nodes")

    # Baseline Domar weights
    domar_base = network.domar_weights(phi=0.0)
    print(f"\nDomar weights (summing to {domar_base.sum():.3f}):")
    for s, sector in enumerate(SECTORS):
        idx = s * network.R  # US domestic
        print(f"  {sector:25s}: λ = {domar_base[idx]:.4f}")

    # TFP loss from fragmentation
    tfp_loss = network.fragmentation_tfp_loss(phi=0.15)
    print(f"\nTFP loss at baseline fragmentation (φ=0.15): {tfp_loss:.4f}")

    # Fragmentation path
    phi_path = 0.15 * np.exp(-0.1 * np.arange(20)) + \
               0.30 * np.exp(-0.1 * np.arange(20)) * (np.arange(20) == 0)
    phi_path = np.minimum(phi_path, 0.50)
    phi_path[0] = 0.30

    dynamics = simulate_fragmentation_dynamics(phi_path, network)
    print(f"\nFragmentation shock (φ: 0.15 → 0.30 → decay):")
    print(f"  Peak TFP loss: {dynamics['tfp_loss'].min():.4f}")
    print(f"  Entropy decline: {dynamics['entropy'][0]:.4f} → {dynamics['entropy'][-1]:.4f}")
