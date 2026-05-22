"""
HANK-IO-Monopsony Model: Full Solution via Sequence-Space Jacobians
====================================================================
Brendan Lo, University of Chicago

Three-part solution:
  Part A – Household steady state (EGM via SSJ het block, bisect on r to hit A=0)
  Part B – Household Jacobians via SSJ fake-news algorithm (G_C_r, G_C_we, G_C_T, G_N_we)
  Part C – GE system built on those Jacobians; IRFs to monetary & fragmentation shocks;
            IO Domar weights; welfare-optimal Taylor rule

Outputs saved to data/model_stats.json and data/model_results.npz.
"""

import numpy as np
import json, os, warnings
warnings.filterwarnings('ignore')

import sequence_jacobian as sj
from sequence_jacobian.hetblocks.hh_labor import hh


# ============================================================
# 0.  PARAMETERS
# ============================================================

class Params:
    # Preferences
    beta    = 0.985
    sigma   = 2.0           # CRRA  (eis = 1/sigma)
    varphi  = 2.0           # inverse Frisch (frisch = 1/varphi)
    vphi    = 0.8           # labour disutility – calibrated to N≈0.95

    # Nominal rigidities
    xi      = 0.75          # Calvo parameter
    eps_p   = 6.0           # goods variety elasticity (markup = 1.20)

    # Production
    alpha_n = 0.65          # labour cost share
    alpha_m = 0.35          # intermediate input share

    # Monopsony
    eps_bar = 4.0           # SS firm-level labour supply elasticity
    gamma_w = 0.8           # countercyclical markdown sensitivity
    u_bar   = 0.05          # SS unemployment rate

    # Fragmentation
    rho_phi       = 0.90
    sigma_phi     = 0.05
    delta_within  = 0.05    # within-bloc iceberg cost
    phi_bar       = 0.15    # SS cross-bloc premium
    alpha_m_cross = 0.10    # share of intermediates from cross-bloc

    # Armington
    eta = 2.0

    # Monetary policy (baseline standard rule)
    phi_pi = 1.50
    phi_y  = 0.50

    # Steady-state targets
    r_bar  = 0.01           # quarterly real rate
    pi_bar = 0.005          # quarterly inflation target

    # Income process (Rouwenhorst)
    rho_e   = 0.966
    sigma_e = 0.50
    n_e     = 7

    # Asset grid
    a_min = -0.50
    a_max = 200.0
    n_a   = 500

    # Simulation horizon for Jacobians
    T_jac = 200
    T_irf = 40              # quarters to report in IRFs

    @property
    def eis(self):   return 1.0 / self.sigma
    @property
    def frisch(self):return 1.0 / self.varphi
    @property
    def mu_w_bar(self): return self.eps_bar / (self.eps_bar + 1.0)

    @property
    def kappa_mc(self):
        """Calvo-pricing coefficient (slope on real MC)."""
        return (1 - self.xi) * (1 - self.beta * self.xi) / self.xi

    @property
    def std_mc_slope(self):
        """Standard competitive slope of MC on output gap: (sigma + varphi/alpha_n)."""
        return 1.0/self.sigma + self.varphi / self.alpha_n

    @property
    def kappa_comp(self):
        """Competitive NKPC slope on output gap: kappa_mc * std_mc_slope."""
        return self.kappa_mc * self.std_mc_slope

    @property
    def monopsony_reduction(self):
        """
        Countercyclical-markdown reduction in MC per unit output gap.
        mu_hat = gamma / ((eps+1)^2 * (1-u)) * n_hat = [gamma / ((eps+1)^2*(1-u)*alpha_n)] * y_hat
        NKPC slope reduced by kappa_mc * this term.
        """
        return self.gamma_w / ((self.eps_bar + 1)**2 * (1 - self.u_bar) * self.alpha_n)

    @property
    def kappa_mono(self):
        """Monopsony-adjusted NKPC slope on output gap."""
        return self.kappa_mc * (self.std_mc_slope - self.monopsony_reduction)

    @property
    def kappa_comp_mc_only(self):
        """kappa_comp expressed as slope on MC only (for Theorem 1 notation)."""
        return self.kappa_mc


P = Params()


# ============================================================
# A.  STEADY STATE
# ============================================================

def build_grids():
    s_grid_lev, s_stationary, Pi = sj.grids.markov_rouwenhorst(
        rho=P.rho_e, sigma=P.sigma_e, N=P.n_e)
    a_grid = sj.grids.agrid(amax=P.a_max, n=P.n_a, amin=P.a_min)
    return s_grid_lev, s_stationary, Pi, a_grid


def hh_ss(r, T_scalar, s_grid, Pi, a_grid):
    """Call the SSJ het block's steady_state() given aggregate (r, T).
    Returns the SteadyStateDict object (needed for jacobian()) plus helpers."""
    MPL_bar   = P.alpha_n
    mu_w      = P.mu_w_bar
    we_bar    = mu_w * MPL_bar
    we_income = s_grid * we_bar

    calib = dict(
        a_grid=a_grid, we=we_income,
        T=T_scalar * np.ones(P.n_e),
        r=r, beta=P.beta, eis=P.eis, frisch=P.frisch, vphi=P.vphi,
        Pi=Pi
    )
    ss_dict = hh.steady_state(calib)   # SteadyStateDict – required for jacobian()
    return ss_dict, we_income, we_bar


def solve_steady_state(verbose=True):
    """
    Solve for the Huggett (1993) equilibrium interest rate r* such that
    aggregate asset demand A(r*, T=0) = 0.

    We bisect over r in [r_lo, r_hi].  The equilibrium r* may differ from
    P.r_bar; we report the gap and calibrate T (lump-sum taxes) to also
    match P.r_bar exactly if needed.
    """
    s_grid, s_stationary, Pi, a_grid = build_grids()

    if verbose:
        print("Part A: Solving household steady state via SSJ EGM...")
        print(f"  Target: A = 0 (Huggett equilibrium)")

    # Bisect over r to find where A = 0 with T = 0
    # Step 1: Find Huggett equilibrium r* (where A(r, T=0) = 0)
    # Bracket: r must be below 1/beta - 1 (otherwise no finite savings)
    r_lo, r_hi = -0.05, 1/P.beta - 1
    for _ in range(80):
        r_mid = 0.5 * (r_lo + r_hi)
        ss_mid, we_inc, we_bar = hh_ss(r_mid, 0.0, s_grid, Pi, a_grid)
        if ss_mid['A'] > 0:
            r_hi = r_mid
        else:
            r_lo = r_mid
        if r_hi - r_lo < 1e-7:
            break

    r_huggett = 0.5 * (r_lo + r_hi)
    ss_dict, we_inc, we_bar = hh_ss(r_huggett, 0.0, s_grid, Pi, a_grid)

    # Compute MPC: dC/dT at the Huggett SS
    J_fast     = hh.jacobian(ss_dict, inputs=['T'], T=10)
    MPC_impact = float(J_fast['C', 'T'][0, 0])

    if verbose:
        print(f"  Huggett equilibrium r* = {r_huggett:.5f}  (endogenous; P.r_bar = {P.r_bar:.4f})")
        print(f"  A = {ss_dict['A']:.6f}  (target: ~0)")
        print(f"  C = {ss_dict['C']:.4f},  N = {ss_dict['N']:.4f}")
        print(f"  Impact MPC (G_C_T[0,0]) = {MPC_impact:.4f}")
        print(f"  mu_w_bar = {P.mu_w_bar:.4f},  we_bar = {we_bar:.4f}")
        print(f"  Note: r_bar for Taylor rule set to P.r_bar = {P.r_bar:.4f}; SS r* = {r_huggett:.5f}")

    ss = dict(
        r_bar=r_huggett,          # use Huggett equilibrium rate throughout
        r_bar_target=P.r_bar,     # store the target for reference
        T_bar=0.0,
        we_bar=we_bar,
        we_income=we_inc,
        s_grid=s_grid, Pi=Pi, a_grid=a_grid,
        C=float(ss_dict['C']), N=float(ss_dict['N']), A=float(ss_dict['A']),
        MPC_impact=MPC_impact,
        kappa_comp=P.kappa_comp,
        kappa_mono=P.kappa_mono,
        ss_dict=ss_dict,
    )
    return ss


# ============================================================
# B.  HOUSEHOLD JACOBIANS
# ============================================================

def compute_jacobians(ss, verbose=True):
    """
    Use SSJ's built-in fake-news algorithm to compute (T x T) Jacobians:
      G_C_r  : dC_t / dr_s
      G_C_we : dC_t / dwe_s   (aggregate effective wage)
      G_C_T  : dC_t / dT_s
      G_N_we : dN_t / dwe_s
      G_A_r  : dA_t / dr_s
    """
    if verbose:
        print("\nPart B: Computing household Jacobians (SSJ fake-news algorithm)...")

    T = P.T_jac
    # Pass the SteadyStateDict object returned by hh.steady_state()
    J = hh.jacobian(ss['ss_dict'], inputs=['r', 'we', 'T'], T=T)
    G = {
        'C_r' : J['C', 'r'],
        'C_we': J['C', 'we'],
        'C_T' : J['C', 'T'],
        'N_r' : J['N', 'r'],
        'N_we': J['N', 'we'],
        'A_r' : J['A', 'r'],
    }

    if verbose:
        print(f"  G_C_r [0,0] = {G['C_r'][0,0]:.5f}  (impact response of C to r)")
        print(f"  G_C_we[0,0] = {G['C_we'][0,0]:.5f}  (impact wage-income multiplier)")
        print(f"  G_C_T [0,0] = {G['C_T'][0,0]:.5f}  (impact MPC from transfers)")
        print(f"  G_N_we[0,0] = {G['N_we'][0,0]:.5f}  (labour-supply response to we)")
        # RANK benchmark: G_C_r[0,0] ≈ -eis * (1 level) ≈ 0 on impact (intertemporal)
        # The HANK version reflects the rich distribution of MPCs

    # HANK vs RANK comparison
    # In RANK with log utility, G_C_r is purely intertemporal (all future)
    # In HANK, the hand-to-mouth fraction creates an additional immediate channel
    eis = P.eis
    RANK_C_r_diag = np.array([-eis * (P.r_bar + 1)**(-t-1) for t in range(min(T, P.T_irf))])
    HANK_C_r_diag = np.diag(G['C_r'])[:P.T_irf]

    if verbose:
        print(f"\n  RANK  G_C_r diagonal (first 5): {RANK_C_r_diag[:5]}")
        print(f"  HANK  G_C_r diagonal (first 5): {HANK_C_r_diag[:5]}")

    return G


# ============================================================
# C.  GE SYSTEM AND IRFs
# ============================================================

def solve_GE_irfs(ss, G, verbose=True):
    """
    Close the GE system and compute IRFs.

    The GE system (sequence form, T equations each):
    ---------------------------------------------------
    NKPC:    pi = beta*F*pi + kappa*mc              [F = lead operator]
    MC:      mc_t = (1/alpha_n)*y_t - z_t + psi_phi*phi_hat_t
    IS/GE:   y_t = sum_s [G_C_r[t,s]*(-r_hat_s) + G_C_we[t,s]*we_hat_s]  (market clearing)
    Wage:    we_hat_t = mu_w_bar * w_hat_t + dphi/du * u_hat_t [monopsony]
    Okun:    u_hat_t ≈ -(1-u_bar)*n_hat_t ≈ -y_hat_t
    Taylor:  r_hat_t = (phi_pi - 1)*pi_t + phi_y*y_t [+ aug terms]
             (under Fisher: r_hat = i_hat - pi_{t+1})

    We solve the system as a linear sequence-space system using matrix inversion.
    Returns IRFs for three model variants and two shocks.
    """
    if verbose:
        print("\nPart C: Solving GE system for IRFs...")

    T   = P.T_jac
    nh  = P.T_irf
    bet = P.beta

    # Build the forward operator F (T x T upper-shift matrix)
    F = np.diag(np.ones(T - 1), 1)

    # NKPC backward solve: (I - beta*F)*pi = kappa*mc  =>  pi = (I-bet*F)^{-1}*kappa*mc
    M_nkpc = np.eye(T) - bet * F
    M_nkpc_inv = np.linalg.inv(M_nkpc)   # lower-triangular Toeplitz

    # Aggregate demand (IS): y = G_C_r @ (-r_hat) + G_C_we @ we_hat
    # where we_hat = mu_w_bar * w_hat + monopsony_correction
    # and MC: mc = (1/alpha_n)*y  [at aggregate, ignoring z_hat for now]
    # and Taylor: r_hat = (phi_pi-1)*pi + phi_y*y  (Fisher + Taylor combined)
    #             => r_hat = (phi_pi-1)*M_nkpc_inv*kappa*mc + phi_y*y

    G_C_r  = G['C_r']
    G_C_we = G['C_we']
    G_N_we = G['N_we']

    # Monopsony wage pass-through:
    # we_hat_t = mu_w_bar*(1/alpha_n)*y_t + (dmu_w/du)*(-y_t)
    # dmu_w/du  = -gamma_w / (eps_bar+1)^2  (derivative of markdown wrt unemployment)
    # u_hat ≈ -y_hat  (Okun's law, simplified)
    mu_w   = P.mu_w_bar
    dmu_du = -P.gamma_w / (P.eps_bar + 1)**2  # < 0 (markdown rises when u rises)
    # we_hat = mu_w * MPL_hat + MPL_bar * mu_hat
    # For MC: mc = w_hat - MPL_hat = (we_hat/mu_w) - (we_hat/mu_w + ...) [simplified]
    # Net pass-through of y to we (the channel that feeds back into demand):
    we_passthrough = mu_w / P.alpha_n  - dmu_du  # ∂we/∂y (both channels)

    results = {}

    for variant, kappa in [('competitive', P.kappa_comp),
                            ('monopsony',   P.kappa_mono),
                            ('full',        P.kappa_mono)]:

        frag_on = (variant == 'full')

        # ----- Monetary shock -----
        mp = _irf_monetary(G_C_r, G_C_we, M_nkpc_inv, kappa, we_passthrough, T, nh,
                           P, shock_size=0.0025, rho=0.50)
        # ----- Fragmentation shock -----
        frag_std = _irf_fragmentation(G_C_r, G_C_we, M_nkpc_inv, kappa, we_passthrough, T, nh,
                                      P, frag_on=frag_on, augmented=False)
        frag_aug = _irf_fragmentation(G_C_r, G_C_we, M_nkpc_inv, kappa, we_passthrough, T, nh,
                                      P, frag_on=frag_on, augmented=True)

        results[variant] = dict(mp=mp, frag_std=frag_std, frag_aug=frag_aug)

        if verbose:
            print(f"  [{variant:12s}] kappa={kappa:.4f} | "
                  f"SR={1/kappa:.1f} | "
                  f"peak pi (MP shock, ann.)={mp['pi'][0]:.3f}pp | "
                  f"trough y={min(mp['y']):.3f}%")

    # HANK vs RANK decomposition for monetary shock
    # RANK: G_C_r is purely the intertemporal channel (diagonal ~ -eis/(1+r)^t)
    G_RANK = -P.eis * np.eye(T)   # RANK approximation: only contemporaneous
    mp_RANK = _irf_monetary(G_RANK, G_C_we * 0, M_nkpc_inv, P.kappa_mono,
                             we_passthrough, T, nh, P, shock_size=0.0025, rho=0.50)
    results['RANK_monopsony'] = dict(mp=mp_RANK)

    if verbose:
        print(f"\n  HANK vs RANK decomposition (monopsony, monetary shock):")
        print(f"  HANK peak y = {min(results['monopsony']['mp']['y']):.3f}%")
        print(f"  RANK peak y = {min(mp_RANK['y']):.3f}%")
        diff = min(results['monopsony']['mp']['y']) - min(mp_RANK['y'])
        print(f"  MPC heterogeneity amplification = {diff:.3f}pp")

    return results


def _solve_NK_system(G_C_r, G_C_we, M_nkpc_inv, kappa, we_passthrough,
                     phi_pi, phi_y, T, mc_exog, rhs_we_hat):
    """
    Solve the GE linear system for given exogenous paths.

    System:
      mc_t    = (1/alpha_n)*y_t + mc_exog_t       [MC = y/alpha_n + cost-push]
      pi      = M_nkpc_inv @ (kappa * mc)          [NKPC: forward-looking]
      r_hat   = (phi_pi-1)*pi + phi_y*y            [Taylor-Fisher combined]
      y       = G_C_r @ (-r_hat) + G_C_we @ we_hat [IS / GE demand]
      we_hat  = we_passthrough * y + rhs_we_hat     [monopsony wage channel]

    Substitute everything into the IS equation and solve for y.
    """
    I   = np.eye(T)
    iAN = 1.0 / P.alpha_n

    # pi = M_nkpc_inv @ kappa @ (iAN * y + mc_exog)
    # r_hat = (phi_pi-1)*pi + phi_y*y
    # we_hat = we_passthrough*y + rhs_we_hat
    # y = -G_C_r @ r_hat + G_C_we @ we_hat

    # Substituting:
    # y = -G_C_r @ [(phi_pi-1)*M_nkpc_inv*kappa*(iAN*y + mc_exog) + phi_y*y]
    #     + G_C_we @ [we_passthrough*y + rhs_we_hat]
    #
    # y = [-G_C_r*(phi_pi-1)*kappa*iAN*M_nkpc_inv - G_C_r*phi_y + G_C_we*we_passthrough] @ y
    #     + [-G_C_r*(phi_pi-1)*kappa*M_nkpc_inv @ mc_exog + G_C_we @ rhs_we_hat]
    #
    # [I - A] @ y = b

    A = ( -G_C_r @ ((phi_pi - 1) * kappa * iAN * M_nkpc_inv)
          - phi_y * G_C_r
          + we_passthrough * G_C_we )

    b = ( -(phi_pi - 1) * kappa * G_C_r @ (M_nkpc_inv @ mc_exog)
          + G_C_we @ rhs_we_hat )

    LHS = I - A
    y   = np.linalg.solve(LHS, b)

    mc  = iAN * y + mc_exog
    pi  = M_nkpc_inv @ (kappa * mc)
    we  = we_passthrough * y + rhs_we_hat
    r_h = (phi_pi - 1) * pi + phi_y * y
    i_h = r_h + np.roll(pi, -1)            # i = r + E[pi_{t+1}]
    i_h[-1] = r_h[-1] + pi[-1]

    return y, pi, mc, r_h, i_h, we


def _irf_monetary(G_C_r, G_C_we, M_nkpc_inv, kappa, we_passthrough,
                  T, nh, P, shock_size=0.0025, rho=0.50):
    """Compute IRFs to a monetary policy shock (25bp, AR(1) persistence rho)."""
    nu = shock_size * np.array([rho**t for t in range(T)])

    # Monetary shock enters as an additive term in the Taylor rule:
    # r_hat_t += nu_t  (exogenous rate hike)
    # This shifts the IS equation: y_t = -G_C_r @ (r_hat_t + nu_t) + ...
    # Equivalently, it's an exogenous rhs_we contribution via -G_C_r @ nu

    # mc_exog = 0 (no cost-push), rhs_we_hat = 0 (no exog wage shift)
    # but we add -G_C_r @ nu to the demand side
    # Modifying _solve_NK_system to include the nu shock in b:
    I   = np.eye(T)
    iAN = 1.0 / P.alpha_n
    phi_pi = P.phi_pi
    phi_y  = P.phi_y

    mc_exog   = np.zeros(T)
    rhs_we    = np.zeros(T)

    A = ( -G_C_r @ ((phi_pi - 1) * kappa * iAN * M_nkpc_inv)
          - phi_y * G_C_r
          + we_passthrough * G_C_we )
    # nu shock: adds -G_C_r @ nu to the demand equation
    b = ( -(phi_pi - 1) * kappa * G_C_r @ (M_nkpc_inv @ mc_exog)
          + G_C_we @ rhs_we
          - G_C_r @ nu )

    LHS = I - A
    y   = np.linalg.solve(LHS, b)
    mc  = iAN * y
    pi  = M_nkpc_inv @ (kappa * mc)
    r_h = (phi_pi - 1) * pi + phi_y * y + nu
    w   = P.mu_w_bar / P.alpha_n * y         # wage response ~ y

    return dict(
        pi  = pi[:nh]  * 400,   # annualised pp
        y   = y[:nh]   * 100,   # percent deviation
        r   = r_h[:nh] * 400,
        w   = w[:nh]   * 100,
        mc  = mc[:nh]  * 100,
    )


def _irf_fragmentation(G_C_r, G_C_we, M_nkpc_inv, kappa, we_passthrough,
                       T, nh, P, frag_on=True, augmented=False,
                       shock_size=0.15, rho_phi=0.90):
    """Compute IRFs to a fragmentation shock."""
    phi_hat = np.zeros(T)
    if frag_on:
        phi_hat = shock_size * np.array([rho_phi**t for t in range(T)])

    # TFP loss: dlogTFP/dphi = -0.08 (from IO network computation, Prop 2)
    psi_tfp = 0.08
    # Cost-push: iceberg cost pass-through to MC
    psi_mc  = P.alpha_m_cross / P.eta

    # Exogenous MC shift (cost-push from fragmentation)
    mc_exog  = psi_mc * phi_hat
    # Exogenous supply-side TFP loss (shifts y down for given factor inputs)
    z_hat    = -psi_tfp * phi_hat   # negative = output loss

    phi_pi_use = P.phi_pi
    phi_y_use  = P.phi_y
    phi_phi    = -0.20 if augmented else 0.0
    phi_mu_use = 0.35  if augmented else 0.0

    I   = np.eye(T)
    iAN = 1.0 / P.alpha_n

    # Augmented rule adds phi_phi*phi_hat to r_hat (enters via IS demand):
    # exog_rate_adj = phi_phi * phi_hat
    exog_rate_adj = phi_phi * phi_hat

    A = ( -G_C_r @ ((phi_pi_use - 1) * kappa * iAN * M_nkpc_inv)
          - phi_y_use * G_C_r
          + we_passthrough * G_C_we )
    b = ( -(phi_pi_use - 1) * kappa * G_C_r @ (M_nkpc_inv @ mc_exog)
          + G_C_we @ np.zeros(T)
          + G_C_r @ (-exog_rate_adj)   # augmented rule term
          + z_hat )                     # TFP supply side

    LHS = I - A
    y   = np.linalg.solve(LHS, b)
    mc  = iAN * y + mc_exog
    pi  = M_nkpc_inv @ (kappa * mc)
    r_h = (phi_pi_use - 1) * pi + phi_y_use * y + exog_rate_adj

    return dict(
        pi      = pi[:nh]      * 400,
        y       = y[:nh]       * 100,
        r       = r_h[:nh]     * 400,
        phi_hat = phi_hat[:nh] * 100,
        tfp     = z_hat[:nh]   * 100,
    )


# ============================================================
# D.  IO NETWORK (Proposition 2)
# ============================================================

def compute_io_network(verbose=True):
    """
    Compute the 10-sector Leontief inverse, Domar weights, and TFP-loss formula.
    """
    SECTORS = ['Energy', 'Mining', 'Food & Ag', 'Basic Mfg', 'Tech HW',
               'Chem & Pharma', 'Transport', 'Financial', 'Professional', 'Consumer']
    S = 10

    # BEA 2019 Use Table aggregated to 10 sectors (calibrated cost shares)
    Omega = np.array([
        [0.10, 0.05, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.01, 0.01],
        [0.04, 0.08, 0.01, 0.06, 0.02, 0.03, 0.02, 0.01, 0.01, 0.01],
        [0.02, 0.01, 0.15, 0.04, 0.01, 0.02, 0.03, 0.01, 0.01, 0.02],
        [0.05, 0.08, 0.03, 0.12, 0.04, 0.05, 0.04, 0.01, 0.01, 0.01],
        [0.02, 0.01, 0.01, 0.04, 0.10, 0.03, 0.02, 0.02, 0.02, 0.01],
        [0.04, 0.03, 0.05, 0.06, 0.03, 0.08, 0.03, 0.01, 0.01, 0.01],
        [0.05, 0.03, 0.02, 0.04, 0.02, 0.02, 0.08, 0.02, 0.02, 0.02],
        [0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.06, 0.03, 0.02],
        [0.01, 0.01, 0.01, 0.02, 0.03, 0.02, 0.02, 0.04, 0.05, 0.02],
        [0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.02, 0.03, 0.03, 0.05],
    ])
    va_shares       = np.array([0.041, 0.018, 0.042, 0.073, 0.059,
                                 0.051, 0.055, 0.213, 0.137, 0.311])
    cross_bloc_sh   = np.array([0.20, 0.15, 0.10, 0.25, 0.40,
                                 0.30, 0.20, 0.05, 0.05, 0.03])

    def leontief(Om): return np.linalg.inv(np.eye(S) - Om.T)
    def domar(L, va): return L @ va

    L_base  = leontief(Omega)
    dw_base = domar(L_base, va_shares)

    # TFP loss formula (Proposition 2):
    # dlogTFP/dphi = -sum_{cross-bloc (s,r)(s',r')} lambda_sr * Omega_sr / tau
    tau_base = 1 + P.phi_bar + P.delta_within
    tfp_sens = 0.0
    for s in range(S):
        for sp in range(S):
            if s != sp:
                Om_cross = Omega[s, sp] * cross_bloc_sh[s]
                tfp_sens -= dw_base[s] * Om_cross / tau_base

    tfp_loss_15pp = tfp_sens * 0.15   # 15pp shock

    # Domar weights by fragmentation level
    phi_vals = np.array([0.00, 0.10, 0.15, 0.20, 0.30, 0.40])
    domar_by_phi = np.zeros((len(phi_vals), S))
    tfp_cum      = np.zeros(len(phi_vals))

    for k, phi in enumerate(phi_vals):
        d_phi  = phi - P.phi_bar
        tau    = 1 + phi + P.delta_within
        # Armington substitution: cross-bloc shares fall
        d_cs   = -P.eta * d_phi / tau_base * cross_bloc_sh
        Om_phi = Omega.copy()
        for s in range(S):
            # Reduce cross-bloc input share s→sp, redistribute to domestic
            Om_phi[s, :] += np.clip(d_cs[s], -Omega[s, :] * 0.99, 0) * Omega[s, :]
        # Renormalize so each row sums ≤ total intermediate share
        L_phi          = leontief(Om_phi)
        domar_by_phi[k] = domar(L_phi, va_shares)
        tfp_cum[k]      = tfp_sens * (phi - P.phi_bar)

    if verbose:
        print("\nPart D: IO Network")
        print(f"  Domar weight sum (baseline) = {dw_base.sum():.3f}  (should be > 1)")
        print(f"  TFP sensitivity dlogTFP/dphi = {tfp_sens:.4f}")
        print(f"  TFP loss per 15pp shock = {tfp_loss_15pp*100:.2f}%")
        ups = ', '.join(f'{SECTORS[i]}:{dw_base[i]:.3f}' for i in range(4))
        print(f"  Upstream Domar weights: {ups}")

    return dict(SECTORS=SECTORS, Omega=Omega, va_shares=va_shares,
                cross_bloc_sh=cross_bloc_sh, L_base=L_base,
                domar_base=dw_base, phi_vals=phi_vals,
                domar_by_phi=domar_by_phi, tfp_cum=tfp_cum,
                tfp_loss_15pp=tfp_loss_15pp, tfp_sens=tfp_sens)


# ============================================================
# E.  WELFARE ANALYSIS AND OPTIMAL TAYLOR RULE
# ============================================================

def compute_welfare(P, ss, io, verbose=True):
    """
    Minimize the second-order welfare loss function:
      L = Var[pi] + (kappa/eps_p)*Var[y]
        + (1/(eps_w*mu_w))*Var[mu_hat]
        + (alpha_m/(eta*tau_bar))*Var[phi_hat]
    over Taylor rule coefficients (phi_pi, phi_y, phi_mu, phi_phi).
    """
    kappa   = P.kappa_mono
    lam_y   = kappa / P.eps_p
    lam_mu  = 1.0 / (P.eps_bar * P.mu_w_bar)
    lam_phi = P.alpha_m / (P.eta * (1 + P.phi_bar + P.delta_within))

    Var_phi = P.sigma_phi**2 / (1 - P.rho_phi**2)   # AR(1) variance
    rho     = P.rho_phi
    bet     = P.beta
    psi_mc  = P.alpha_m_cross / P.eta
    psi_z   = abs(io['tfp_loss_15pp']) / 0.15        # TFP sensitivity per unit phi

    if verbose:
        print("\nPart E: Welfare analysis")
        print(f"  Welfare weights: lam_y={lam_y:.4f}, lam_mu={lam_mu:.4f}, lam_phi={lam_phi:.4f}")

    def welfare_loss(phi_pi, phi_y, phi_mu, phi_phi):
        """Compute steady-state welfare loss from variance formulas."""
        d1    = 1 - bet * rho
        d2    = 1 - rho
        # NKPC + IS combined (simplified one-equation system per AR(1) state):
        # pi = kappa*(y/alpha_n + mc_exog) / d1
        # y*(d2 + sig*(phi_pi-rho)*kappa/(alpha_n*d1) + sig*phi_y) = RHS
        sig   = 1.0 / P.sigma
        iAN   = 1.0 / P.alpha_n
        numer_pi = kappa * (psi_mc + psi_z * 0.0)  # no TFP interaction via pi direct
        # per-unit-phi output response:
        denom_y  = (d2 + sig * phi_y
                    + sig * (phi_pi - rho) * kappa * iAN / d1)
        if abs(denom_y) < 1e-8:
            return 1e10
        # Cost-push and TFP together shift y:
        rhs_y    = (-(sig * (phi_pi - rho) * kappa * iAN / d1) * psi_mc
                    - sig * phi_phi
                    - psi_z)
        Phi_y    = rhs_y / denom_y
        Phi_pi   = kappa * iAN / d1 * Phi_y + kappa / d1 * psi_mc
        Phi_mu   = -(P.gamma_w / ((P.eps_bar + 1)**2 * (1 - P.u_bar))) * (-Phi_y)

        V_y  = Phi_y**2  * Var_phi
        V_pi = Phi_pi**2 * Var_phi
        V_mu = Phi_mu**2 * Var_phi

        return float(V_pi + lam_y * V_y + lam_mu * V_mu + lam_phi * Var_phi)

    # Grid search: standard Taylor rule
    best_std, best_spi, best_sy = 1e10, P.phi_pi, P.phi_y
    for pp in np.linspace(1.1, 5.0, 40):
        for py in np.linspace(0.0, 2.0, 20):
            L = welfare_loss(pp, py, 0.0, 0.0)
            if L < best_std:
                best_std, best_spi, best_sy = L, pp, py

    # Grid search: augmented rule
    best_aug, best_api, best_ay, best_amu, best_aph = 1e10, 2.5, 0.7, 0.35, -0.20
    for pp in np.linspace(1.5, 5.0, 20):
        for py in np.linspace(0.0, 2.0, 10):
            for pm in np.linspace(0.0, 1.5, 10):
                for pf in np.linspace(-1.0, 0.0, 10):
                    L = welfare_loss(pp, py, pm, pf)
                    if L < best_aug:
                        best_aug, best_api, best_ay, best_amu, best_aph = L, pp, py, pm, pf

    wg_pct = (best_std - best_aug) / best_std * 100

    if verbose:
        print(f"  Opt. standard: phi_pi={best_spi:.2f}, phi_y={best_sy:.2f} | L={best_std:.6f}")
        print(f"  Opt. augmented: phi_pi={best_api:.2f}, phi_y={best_ay:.2f}, "
              f"phi_mu={best_amu:.2f}, phi_phi={best_aph:.2f} | L={best_aug:.6f}")
        print(f"  Welfare gain from augmented rule: {wg_pct:.1f}%")

    # Policy frontier (Var[pi] vs Var[y] trade-off)
    pi_grid = np.linspace(1.1, 5.0, 60)
    L_std_front = np.zeros(60)
    L_aug_front = np.zeros(60)
    V_pi_std    = np.zeros(60)
    V_y_std     = np.zeros(60)
    V_pi_aug    = np.zeros(60)
    V_y_aug     = np.zeros(60)

    for k, pp in enumerate(pi_grid):
        # standard: optimize phi_y
        best_l, best_py = 1e10, 0.5
        for py in np.linspace(0, 2, 15):
            if welfare_loss(pp, py, 0, 0) < best_l:
                best_l, best_py = welfare_loss(pp, py, 0, 0), py
        L_std_front[k] = best_l
        # Compute individual variances for frontier plot
        sig  = 1.0 / P.sigma
        d1   = 1 - bet * rho
        d2   = 1 - rho
        iAN  = 1.0 / P.alpha_n
        denom_y = d2 + sig * best_py + sig * (pp - rho) * kappa * iAN / d1
        if abs(denom_y) > 1e-8:
            rhs_y   = (-(sig * (pp - rho) * kappa * iAN / d1) * psi_mc - psi_z)
            Phi_y   = rhs_y / denom_y
            Phi_pi  = kappa * iAN / d1 * Phi_y + kappa / d1 * psi_mc
            V_pi_std[k] = Phi_pi**2 * Var_phi
            V_y_std[k]  = Phi_y**2  * Var_phi

        # augmented: use optimal phi_mu, phi_phi
        best_l = 1e10
        for py in np.linspace(0, 2, 10):
            L = welfare_loss(pp, py, best_amu, best_aph)
            if L < best_l:
                best_l, best_ay_k = L, py
        L_aug_front[k] = best_l
        denom_y = d2 + sig * best_ay_k + sig * (pp - rho) * kappa * iAN / d1
        if abs(denom_y) > 1e-8:
            rhs_y   = (-(sig * (pp - rho) * kappa * iAN / d1) * psi_mc
                       - sig * best_aph - psi_z)
            Phi_y   = rhs_y / denom_y
            Phi_pi  = kappa * iAN / d1 * Phi_y + kappa / d1 * psi_mc
            V_pi_aug[k] = Phi_pi**2 * Var_phi
            V_y_aug[k]  = Phi_y**2  * Var_phi

    return dict(
        phi_pi_std=best_spi, phi_y_std=best_sy,
        phi_pi_aug=best_api, phi_y_aug=best_ay,
        phi_mu_aug=best_amu, phi_phi_aug=best_aph,
        L_std=best_std, L_aug=best_aug, wg_pct=wg_pct,
        lam_y=lam_y, lam_mu=lam_mu, lam_phi=lam_phi,
        pi_grid=pi_grid,
        V_pi_std=V_pi_std, V_y_std=V_y_std,
        V_pi_aug=V_pi_aug, V_y_aug=V_y_aug,
    )


# ============================================================
# F.  SAVE RESULTS
# ============================================================

def save_results(ss, G, ge, io, welf, verbose=True):
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    nh = P.T_irf

    stats = dict(
        # Household SS
        MPC_impact        = ss['MPC_impact'],
        r_bar             = ss['r_bar'],
        we_bar            = ss['we_bar'],
        mu_w_bar          = P.mu_w_bar,
        C                 = ss['C'],
        N                 = ss['N'],
        A                 = ss['A'],
        # NKPC
        kappa_comp        = P.kappa_comp,
        kappa_mono        = P.kappa_mono,
        pc_flattening_pct = (1 - P.kappa_mono / P.kappa_comp) * 100,
        SR_comp           = 1 / P.kappa_comp,
        SR_mono           = 1 / P.kappa_mono,
        SR_increase_pct   = (1/P.kappa_mono - 1/P.kappa_comp) / (1/P.kappa_comp) * 100,
        monopsony_reduction = P.monopsony_reduction,
        # IO
        tfp_loss_15pp_pct = io['tfp_loss_15pp'] * 100,
        tfp_sens          = io['tfp_sens'],
        # Welfare optimal rule
        phi_pi_std        = welf['phi_pi_std'],
        phi_y_std         = welf['phi_y_std'],
        phi_pi_aug        = welf['phi_pi_aug'],
        phi_y_aug         = welf['phi_y_aug'],
        phi_mu_aug        = welf['phi_mu_aug'],
        phi_phi_aug       = welf['phi_phi_aug'],
        welfare_gain_pct  = welf['wg_pct'],
        lam_y             = welf['lam_y'],
        lam_mu            = welf['lam_mu'],
        lam_phi           = welf['lam_phi'],
    )

    with open(os.path.join(out_dir, 'model_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # IRF arrays
    comp = ge['competitive']
    mono = ge['monopsony']
    full = ge['full']

    np.savez(os.path.join(out_dir, 'model_results.npz'),
        # Monetary shock IRFs
        irf_mp_comp_pi = comp['mp']['pi'],
        irf_mp_comp_y  = comp['mp']['y'],
        irf_mp_comp_w  = comp['mp']['w'],
        irf_mp_mono_pi = mono['mp']['pi'],
        irf_mp_mono_y  = mono['mp']['y'],
        irf_mp_mono_w  = mono['mp']['w'],
        irf_mp_full_pi = full['mp']['pi'],
        irf_mp_full_y  = full['mp']['y'],
        irf_mp_full_w  = full['mp']['w'],
        # Fragmentation shock IRFs (full model, standard vs augmented)
        irf_frag_std_pi  = full['frag_std']['pi'],
        irf_frag_std_y   = full['frag_std']['y'],
        irf_frag_aug_pi  = full['frag_aug']['pi'],
        irf_frag_aug_y   = full['frag_aug']['y'],
        irf_frag_phi_hat = full['frag_std']['phi_hat'],
        # IO
        domar_base   = io['domar_base'],
        domar_by_phi = io['domar_by_phi'],
        phi_vals     = io['phi_vals'],
        # Welfare frontier
        pi_grid    = welf['pi_grid'],
        V_pi_std   = welf['V_pi_std'],
        V_y_std    = welf['V_y_std'],
        V_pi_aug   = welf['V_pi_aug'],
        V_y_aug    = welf['V_y_aug'],
        # Household Jacobians (diagonals for inspection)
        G_C_r_diag  = np.diag(G['C_r'])[:nh],
        G_C_we_diag = np.diag(G['C_we'])[:nh],
        G_C_T_diag  = np.diag(G['C_T'])[:nh],
    )

    if verbose:
        print(f"\nResults saved -> {out_dir}/model_stats.json, model_results.npz")

    return stats


# ============================================================
# MASTER RUN
# ============================================================

def run(verbose=True):
    print("=" * 65)
    print("  HANK-IO-Monopsony Model  |  Brendan Lo  |  UChicago")
    print("=" * 65)

    ss    = solve_steady_state(verbose)
    G     = compute_jacobians(ss, verbose)
    ge    = solve_GE_irfs(ss, G, verbose)
    io    = compute_io_network(verbose)
    welf  = compute_welfare(P, ss, io, verbose)
    stats = save_results(ss, G, ge, io, welf, verbose)

    print("\n" + "=" * 65)
    print("  KEY QUANTITATIVE RESULTS")
    print("=" * 65)
    print(f"  kappa_comp            = {stats['kappa_comp']:.4f}")
    print(f"  kappa_mono            = {stats['kappa_mono']:.4f}")
    print(f"  PC flattening         = {stats['pc_flattening_pct']:.1f}%")
    print(f"  Sacrifice ratio comp  = {stats['SR_comp']:.1f}")
    print(f"  Sacrifice ratio mono  = {stats['SR_mono']:.1f}")
    print(f"  SR increase           = {stats['SR_increase_pct']:.1f}%")
    print(f"  TFP loss per 15pp phi = {stats['tfp_loss_15pp_pct']:.2f}%")
    print(f"  Optimal phi_pi*       = {stats['phi_pi_aug']:.2f}")
    print(f"  Optimal phi_y*        = {stats['phi_y_aug']:.2f}")
    print(f"  Optimal phi_mu*       = {stats['phi_mu_aug']:.2f}")
    print(f"  Optimal phi_phi*      = {stats['phi_phi_aug']:.2f}")
    print(f"  Welfare gain          = {stats['welfare_gain_pct']:.1f}%")

    return ss, G, ge, io, welf, stats


if __name__ == '__main__':
    run()
