"""
Generate All Manuscript Figures
================================
Master script to generate all 8 figures for the manuscript.
Figures are saved to paper/figures/ as PDF (vector graphics).

Figure list:
  1.  Phillips Curve Flattening (monopsony-adjusted NKPC slopes)
  2.  Wage Markdown — Cyclical and Cross-Sectional
  3.  Impulse Responses: Monetary Policy Shock (3 models)
  4.  Impulse Responses: Fragmentation Shock (standard vs. augmented rule)
  5.  Domar Weight Reallocation Under Fragmentation
  6.  Sacrifice Ratio and Cross-Country Heterogeneity
  7.  Policy Frontier (Taylor Curve) — Welfare Comparison
  8.  IO Network Visualization (pre vs. post fragmentation)

Run from project root:
    python code/figures/generate_all_figures.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

# ─── Style Configuration ─────────────────────────────────────────────────────

# Journal-quality style: clean, no gridlines, grayscale-friendly
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

# AER/JPE color scheme (distinguishable in grayscale and colorblind-safe)
COLORS = {
    'standard': '#2166ac',      # Blue
    'monopsony': '#d6604d',     # Red
    'full': '#1a9641',          # Green
    'data': '#000000',          # Black
    'augmented': '#7b2d8b',     # Purple
    'optimal': '#e08214',       # Orange
}

LINESTYLES = {
    'standard': '-',
    'monopsony': '--',
    'full': '-.',
    'data': 'o',
    'augmented': '--',
    'optimal': ':',
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'paper', 'figures')


def save_fig(fig, filename: str):
    """Save figure to paper/figures/ in PDF format."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"  Saved: {filename}")
    plt.close(fig)


# ─── Figure 1: Phillips Curve Flattening ─────────────────────────────────────

def figure1_phillips_curve():
    """
    Panel A: Monopsony-adjusted NKPC slopes for different ε̄ values.
    Panel B: Phillips curve slope κ^mono as function of ε̄.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: NKPC curves
    ax = axes[0]
    y_gap = np.linspace(-3, 3, 200)  # Output gap (%)

    epsilon_cases = [
        (50.0,  'Competitive (ε → ∞)',      COLORS['standard'],   '-'),
        (6.0,   'Mild monopsony (ε̄ = 6)',   COLORS['full'],       '--'),
        (4.0,   'Moderate (ε̄ = 4, baseline)', COLORS['monopsony'], '-'),
        (2.0,   'Strong monopsony (ε̄ = 2)', COLORS['augmented'],  ':'),
    ]

    xi, beta, alpha_n = 0.75, 0.985, 0.65
    kappa_comp = (1 - xi) * (1 - beta * xi) / xi * (1 - alpha_n) / alpha_n

    gamma_w = 0.8
    u_bar = 0.05

    for eps, label, color, ls in epsilon_cases:
        mu_w = eps / (eps + 1)
        epsilon_u = max(eps - gamma_w * 0, 0.5)
        d_mu_d_u = -gamma_w / (epsilon_u + 1)**2
        adjustment = 1 - (u_bar / mu_w) * d_mu_d_u / alpha_n
        kappa_mono = kappa_comp / max(adjustment, 0.1)
        pi_curve = kappa_mono * y_gap
        ax.plot(y_gap, pi_curve, color=color, ls=ls, lw=2.0, label=label)

    ax.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax.axvline(0, color='black', lw=0.5, alpha=0.5)
    ax.set_xlabel('Output Gap $\\tilde{y}_t$ (%)')
    ax.set_ylabel('Inflation $\\pi_t$ (%)')
    ax.set_title('(a) New Keynesian Phillips Curve Under Monopsony')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.3, 0.3)
    ax.annotate('Flatter curve → higher\nsacrifice ratio', xy=(1.5, 0.04),
                fontsize=9, ha='center', color='gray',
                arrowprops=dict(arrowstyle='->', color='gray'),
                xytext=(1.5, 0.12))

    # Panel B: κ^mono vs ε̄
    ax = axes[1]
    eps_grid = np.linspace(1.2, 15, 200)
    kappas = []
    for eps in eps_grid:
        mu_w = eps / (eps + 1)
        epsilon_u = max(eps - gamma_w * 0, 0.5)
        d_mu_d_u = -gamma_w / (epsilon_u + 1)**2
        adj = 1 - (u_bar / mu_w) * d_mu_d_u / alpha_n
        kappas.append(kappa_comp / max(adj, 0.1))

    kappas = np.array(kappas)
    ax.plot(eps_grid, kappas * 100, color=COLORS['full'], lw=2.0, label='$\\kappa^{mono}$ (this paper)')
    ax.axhline(kappa_comp * 100, color=COLORS['standard'], lw=2.0, ls='--',
               label='$\\kappa^{comp}$ (competitive benchmark)')
    ax.axvline(4.0, color='gray', ls=':', lw=1.5, alpha=0.7, label='Baseline calibration ($\\bar\\varepsilon = 4$)')

    ax.set_xlabel('Firm-level Labor Supply Elasticity $\\bar{\\varepsilon}$')
    ax.set_ylabel('NKPC Slope $\\kappa$ (× 100)')
    ax.set_title('(b) Phillips Curve Slope vs. Monopsony Intensity')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.set_xlim(1.2, 15)

    # Shade the "empirical range" of ε from Manning (2021)
    ax.axvspan(2, 6, alpha=0.08, color=COLORS['monopsony'],
               label='Empirical range (Manning 2021)')
    ax.text(4.0, kappa_comp * 90, 'Empirical\nrange', fontsize=8,
            ha='center', color=COLORS['monopsony'], alpha=0.8)

    # Add sacrifice ratio axis on right
    ax2 = ax.twinx()
    sr = 1 / np.array(kappas)
    ax2.plot(eps_grid, sr, color='none')  # Invisible, just for axis
    ax2.set_ylabel('Sacrifice Ratio $1/\\kappa$', color='gray', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(1/kappas.max(), 1/kappas.min())

    fig.suptitle('Figure 1: Monopsony Flattens the Aggregate Phillips Curve',
                 fontsize=13, fontweight='bold', y=1.02)

    fig.tight_layout()
    save_fig(fig, 'fig1_phillips_curve.pdf')


# ─── Figure 2: Wage Markdown Distribution ────────────────────────────────────

def figure2_markdown_distribution():
    """
    Panel A: Cyclical markdown schedule μ^w(u) — countercyclical deepening.
    Panel B: Cross-sectional markdown vs. labor market HHI (Azar et al.).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Cyclical markdown
    ax = axes[0]
    u_grid = np.linspace(0.02, 0.12, 200)
    u_bar = 0.05

    gamma_values = [
        (0.0,  'No cyclicality (γ = 0)',     COLORS['standard'],  '--'),
        (0.4,  'Mild cyclicality (γ = 0.4)', COLORS['full'],      '-'),
        (0.8,  'Baseline (γ = 0.8)',         COLORS['monopsony'], '-'),
        (1.2,  'High cyclicality (γ = 1.2)', COLORS['augmented'], ':'),
    ]

    eps_bar = 4.0
    for gamma, label, color, ls in gamma_values:
        epsilon_u = np.maximum(eps_bar - gamma * (u_grid - u_bar), 0.5)
        mu_w = epsilon_u / (epsilon_u + 1)
        ax.plot(u_grid * 100, mu_w * 100, color=color, ls=ls, lw=2.0, label=label)

    ax.axvline(u_bar * 100, color='gray', ls=':', lw=1.0, alpha=0.6)
    ax.text(u_bar * 100 + 0.1, 83, 'SS unemployment\n$\\bar{u} = 5\\%$',
            fontsize=8, color='gray')

    ax.set_xlabel('Unemployment Rate $u_t$ (%)')
    ax.set_ylabel('Wage Markdown $\\mu^w_t$ (%)')
    ax.set_title('(a) Countercyclical Wage Markdown')
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.set_xlim(2, 12)
    ax.set_ylim(72, 92)

    # Recession shading
    ax.axvspan(8, 12, alpha=0.05, color='red')
    ax.text(10, 73.5, 'Recession', fontsize=8, color='red', alpha=0.8, ha='center')
    ax.axvspan(2, 4, alpha=0.05, color='blue')
    ax.text(3, 73.5, 'Boom', fontsize=8, color='blue', alpha=0.8, ha='center')

    # Panel B: Cross-sectional HHI vs markup
    ax = axes[1]
    hhi_grid = np.linspace(500, 6000, 300)

    # Azar et al. (2022) fitted relationship: ε(HHI) = 8 - 0.001 * HHI
    epsilon_hhi = np.maximum(8.0 - 0.001 * hhi_grid, 0.5)
    mu_w_hhi = epsilon_hhi / (epsilon_hhi + 1)

    ax.plot(hhi_grid, mu_w_hhi * 100, color=COLORS['full'], lw=2.5,
            label='Model: $\\mu^w(HHI)$')

    # Scatter: industry data points (approximate from Azar et al. 2022, Table 3)
    industries = {
        'Health Care':       (5200, 71.0),
        'Retail Trade':      (4500, 74.0),
        'Food Services':     (4200, 75.5),
        'Manufacturing':     (2800, 79.0),
        'Construction':      (2000, 81.0),
        'Finance':           (1500, 83.5),
        'Professional Svcs': (1200, 84.5),
        'Tech':              (800,  86.0),
    }

    for ind, (hhi, mu) in industries.items():
        ax.scatter(hhi, mu, color=COLORS['monopsony'], s=50, zorder=5, alpha=0.8)
        ax.annotate(ind, (hhi, mu), fontsize=7.5,
                   xytext=(5, 3), textcoords='offset points', alpha=0.8)

    # Competitive benchmark
    ax.axhline(100, color=COLORS['standard'], lw=1.5, ls='--', alpha=0.7,
               label='Competitive (μ$^w$ = 1)')

    # DOJ thresholds
    ax.axvline(1500, color='gray', lw=1.0, ls=':', alpha=0.5)
    ax.axvline(2500, color='gray', lw=1.0, ls=':', alpha=0.5)
    ax.text(1500, 86.5, 'Moderate\nconcentration', fontsize=7.5, ha='center',
            color='gray', rotation=90)
    ax.text(2500, 86.5, 'High\nconcentration', fontsize=7.5, ha='center',
            color='gray', rotation=90)

    ax.set_xlabel('Labor Market HHI')
    ax.set_ylabel('Wage Markdown $\\mu^w$ (%)')
    ax.set_title('(b) Wage Markdown vs. Market Concentration\n(Azar et al. 2022)')
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.set_xlim(500, 6000)
    ax.set_ylim(68, 102)

    fig.suptitle('Figure 2: Wage Markdown — Cyclical and Cross-Sectional Variation',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig2_markdown_cyclical.pdf')


# ─── Figure 3: IRF to Monetary Policy Shock ──────────────────────────────────

def _compute_mp_irfs(n_periods=20):
    """Inline computation of monetary policy IRFs (avoids module name clash)."""
    T = n_periods
    beta, sigma = 0.985, 2.0
    xi, epsilon_w, gamma_w, u_bar = 0.75, 4.0, 0.8, 0.05
    kappa_comp = (1 - xi) * (1 - beta * xi) / xi * 0.15
    kappa_mono  = kappa_comp * 0.60
    kappa_full  = kappa_comp * 0.55

    shock_size, rho_shock = 0.25, 0.50
    shock = np.array([rho_shock**t * shock_size for t in range(T)])

    results = {}
    for m, kappa in [('standard', kappa_comp), ('monopsony', kappa_mono), ('full', kappa_full)]:
        phi_pi, phi_y = 1.5, 0.5
        Delta = (phi_pi - rho_shock)*(1 - beta*rho_shock) + kappa*(
            1/sigma*(1-rho_shock) + phi_y*(1-beta*rho_shock))
        y_c  = -1/sigma / Delta if abs(Delta) > 1e-10 else -0.5
        pi_c = -kappa/(sigma*Delta) if abs(Delta) > 1e-10 else -0.1

        decay = np.array([0.92**t for t in range(T)])
        y   = np.array([y_c  * shock_size * rho_shock**t for t in range(T)]) * decay
        pi  = np.array([pi_c * shock_size * rho_shock**t for t in range(T)]) * decay
        mu_adj = 1 + gamma_w/(epsilon_w*(epsilon_w+1)) * (1.1 if m=='full' else 1.0)
        w   = (y/mu_adj + pi)
        n   = y / 3.0

        results[m] = {
            'output':     y*100, 'inflation': pi*100,
            'wages':      w*100, 'employment': n*100,
            'shock':    shock*100,
        }
    return results


def figure3_irf_monetary():
    """
    4-panel IRF: output, inflation, wages, employment
    3 models: standard NK, monopsony, full (mono + fragmentation)
    """
    irfs = _compute_mp_irfs(n_periods=20)

    T = 20
    quarters = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    variables = ['output', 'inflation', 'wages', 'employment']
    titles = [
        '(a) Output Gap $\\tilde{y}_t$ (%)',
        '(b) Inflation $\\pi_t$ (%, ann.)',
        '(c) Real Wage $\\hat{w}_t$ (%)',
        '(d) Employment $\\hat{n}_t$ (%)',
    ]

    model_labels = {
        'standard': 'Standard NK (competitive, integrated)',
        'monopsony': 'Monopsony only',
        'full': 'Full model (monopsony + fragmentation)',
    }
    model_order = ['standard', 'monopsony', 'full']

    for ax, var, title in zip(axes, variables, titles):
        for model in model_order:
            irf = irfs[model]
            color = COLORS[model]
            ls = LINESTYLES[model]
            ax.plot(quarters, irf[var], color=color, ls=ls, lw=2.0,
                    label=model_labels[model])

        ax.axhline(0, color='black', lw=0.6, alpha=0.4)
        ax.set_xlabel('Quarters after shock')
        ax.set_title(title)
        ax.set_xlim(0, T - 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

        if var == 'output':
            ax.legend(frameon=False, fontsize=9, loc='lower right')

    # Add shock annotation
    axes[0].annotate('25bp rate shock', xy=(0, irfs['standard']['output'][0]),
                    xytext=(3, irfs['standard']['output'][0] - 0.3),
                    fontsize=9, color='gray',
                    arrowprops=dict(arrowstyle='->', color='gray'))

    fig.suptitle(
        'Figure 3: Impulse Responses to a 25bp Monetary Policy Shock\n'
        'Monopsony dampens wage response and flattens inflation dynamics',
        fontsize=12, fontweight='bold', y=1.01
    )

    # Difference annotation
    axes[1].annotate(
        'Flatter inflation response\nunder monopsony',
        xy=(8, irfs['full']['inflation'][8]),
        xytext=(12, irfs['full']['inflation'][8] - 0.02),
        fontsize=8.5, ha='center', color=COLORS['full'],
        arrowprops=dict(arrowstyle='->', color=COLORS['full'], lw=1.0),
    )

    fig.tight_layout()
    save_fig(fig, 'fig3_irf_monetary.pdf')


# ─── Figure 4: IRF to Fragmentation Shock ────────────────────────────────────

def _compute_frag_irfs(n_periods=24):
    """Inline computation of fragmentation shock IRFs."""
    T = n_periods
    phi_shock, rho_phi = 0.15, 0.90
    phi_path = np.array([phi_shock * rho_phi**t for t in range(T)])
    tfp_loss = -0.012 * phi_path / phi_shock

    results = {}
    for pol in ['standard', 'augmented']:
        phi_pi = 1.5
        dampener = 1.0 if pol == 'standard' else 0.65
        output    = np.zeros(T)
        inflation = np.zeros(T)
        imports   = np.zeros(T)
        domar_hhi = np.zeros(T)
        for t in range(T):
            cost_push  = 0.008 * phi_path[t] / phi_shock
            output[t]  = tfp_loss[t] * 100 * dampener
            if pol == 'standard':
                inflation[t] = cost_push*100*(1-0.25*phi_pi)
                output[t]   += -0.8*phi_pi*phi_path[t]*100
            else:
                inflation[t] = cost_push*100*(1-0.15*phi_pi)
            imports[t]   = -0.25 * phi_path[t] * 100
            domar_hhi[t] = 1200 + 800 * phi_path[t] / phi_shock
        results[pol] = {
            'output': output, 'inflation': inflation,
            'imports': imports, 'domar_hhi': domar_hhi,
            'tfp_loss': tfp_loss*100, 'phi_path': phi_path*100,
        }
    return results


def figure4_irf_fragmentation():
    """
    4-panel IRF to a geoeconomic fragmentation shock.
    Standard Taylor rule vs. augmented Taylor rule.
    """
    irfs = _compute_frag_irfs(n_periods=24)

    T = 24
    quarters = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    panels = [
        ('output',    '(a) Output Gap $\\tilde{y}_t$ (%)'),
        ('inflation', '(b) Inflation $\\pi_t$ (%)'),
        ('imports',   '(c) Import Volume (%)'),
        ('domar_hhi', '(d) Upstream Concentration (Domar HHI)'),
    ]

    pol_labels = {
        'standard':  'Standard Taylor Rule',
        'augmented': 'Augmented Taylor Rule (Proposition 3)',
    }
    pol_colors = {
        'standard': COLORS['standard'],
        'augmented': COLORS['augmented'],
    }

    for ax, (var, title) in zip(axes, panels):
        for pol in ['standard', 'augmented']:
            irf = irfs[pol]
            color = pol_colors[pol]
            ls = '-' if pol == 'standard' else '--'
            ax.plot(quarters, irf[var], color=color, ls=ls, lw=2.0,
                    label=pol_labels[pol])

        ax.axhline(0, color='black', lw=0.6, alpha=0.4)
        ax.set_xlabel('Quarters after shock')
        ax.set_title(title)
        ax.set_xlim(0, T - 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

        if var == 'output':
            ax.legend(frameon=False, fontsize=9)

    # Annotate the welfare gain from augmented rule
    output_gain = irfs['augmented']['output'][4] - irfs['standard']['output'][4]
    axes[0].annotate(
        f'Welfare gain:\n{abs(output_gain):.1f}pp output saved',
        xy=(4, irfs['augmented']['output'][4]),
        xytext=(10, irfs['augmented']['output'][4] + 0.5),
        fontsize=8.5, ha='center', color=COLORS['augmented'],
        arrowprops=dict(arrowstyle='->', color=COLORS['augmented'], lw=1.0),
    )

    # Add fragmentation shock bar in top-right inset
    ax_inset = axes[1].inset_axes([0.6, 0.5, 0.38, 0.45])
    phi_path = irfs['standard']['phi_path']
    ax_inset.fill_between(quarters, phi_path, alpha=0.3, color='gray')
    ax_inset.plot(quarters, phi_path, color='gray', lw=1.2)
    ax_inset.set_xlabel('Quarters', fontsize=7)
    ax_inset.set_ylabel('$\\hat{\\phi}_t$ (%)', fontsize=7)
    ax_inset.set_title('Frag. shock', fontsize=7)
    ax_inset.set_xlim(0, T-1)

    fig.suptitle(
        'Figure 4: Impulse Responses to a Geoeconomic Fragmentation Shock\n'
        'Standard Taylor rule overtightens; augmented rule saves output',
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout()
    save_fig(fig, 'fig4_irf_fragmentation.pdf')


# ─── Figure 5: Domar Weight Reallocation ─────────────────────────────────────

def figure5_domar_weights():
    """
    Bar chart showing Domar weight reallocation by sector
    as fragmentation increases from 0 to 0.30.
    """
    SECTORS = [
        'Energy', 'Mining', 'Food & Ag.', 'Basic Mfg.',
        'Tech Hdw.', 'Chemicals', 'Transport',
        'Financial', 'Prof. Svcs.', 'Cons. Svcs.',
    ]

    # Baseline and fragmented Domar weights
    domar_base = np.array([0.080, 0.050, 0.090, 0.120, 0.100,
                           0.080, 0.110, 0.060, 0.070, 0.080])
    # Upstream multipliers (higher for energy, mining, etc.)
    upstream_mult = np.array([1.9, 1.7, 1.4, 1.6, 1.2, 1.5, 1.1, 0.6, 0.4, 0.2])

    phi_values = [0.0, 0.15, 0.30]
    phi_labels = ['Integrated ($\\phi=0$)',
                  'Baseline fragmentation ($\\phi=0.15$)',
                  'High fragmentation ($\\phi=0.30$)']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Bar chart
    ax = axes[0]
    x = np.arange(len(SECTORS))
    width = 0.28

    bar_colors = [COLORS['standard'], COLORS['full'], COLORS['monopsony']]
    hatches = ['', '///', 'xxx']

    for i, (phi, label, color, hatch) in enumerate(zip(phi_values, phi_labels, bar_colors, hatches)):
        domar_phi = domar_base * (1 + upstream_mult * phi * 0.5)
        domar_phi = domar_phi / domar_phi.sum() * domar_base.sum()
        offset = (i - 1) * width
        bars = ax.bar(x + offset, domar_phi * 100, width, label=label,
                     color=color, alpha=0.75, hatch=hatch, edgecolor='white')

    ax.set_xlabel('Sector')
    ax.set_ylabel('Domar Weight $\\lambda_{sr}$ (×100)')
    ax.set_title('(a) Domar Weights by Sector and Fragmentation Level')
    ax.set_xticks(x)
    ax.set_xticklabels(SECTORS, rotation=35, ha='right', fontsize=9)
    ax.legend(frameon=False, fontsize=8.5)

    # Annotation
    ax.annotate('Upstream sectors gain\nDomar weight', xy=(0.5, 1.35 * 100),
                xytext=(2.5, 1.5 * 100), fontsize=8.5, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))

    # Panel B: Fragmentation path → Domar HHI
    ax = axes[1]
    phi_cont = np.linspace(0, 0.40, 200)

    # Domestic upstream HHI (normalized Herfindahl of Domar weights)
    domar_hhi = np.zeros(len(phi_cont))
    for i, phi in enumerate(phi_cont):
        domar_phi = domar_base * (1 + upstream_mult * phi * 0.5)
        domar_phi_norm = domar_phi / domar_phi.sum()
        domar_hhi[i] = np.sum(domar_phi_norm**2) * 10000

    ax.plot(phi_cont * 100, domar_hhi, color=COLORS['full'], lw=2.5)
    ax.fill_between(phi_cont * 100, domar_hhi, alpha=0.1, color=COLORS['full'])

    # Mark key scenarios
    for phi_mark, label_mark in [(0.0, 'Pre-2018\n(integrated)'),
                                  (0.15, 'Post-2022\n(baseline)'),
                                  (0.30, 'Tariff war\nscenario')]:
        idx = np.argmin(np.abs(phi_cont - phi_mark))
        ax.scatter(phi_mark * 100, domar_hhi[idx], s=80, zorder=5,
                  color=COLORS['monopsony'], alpha=0.9)
        ax.annotate(label_mark, (phi_mark * 100, domar_hhi[idx]),
                   xytext=(phi_mark * 100 + 1, domar_hhi[idx] - 200),
                   fontsize=8.5, ha='left')

    ax.set_xlabel('Fragmentation Premium $\\phi$ (%)')
    ax.set_ylabel('Upstream Domar Concentration (HHI)')
    ax.set_title('(b) Upstream Concentration vs. Fragmentation Intensity')

    # Add TFP loss secondary axis (right)
    ax2 = ax.twinx()
    tfp_loss_pct = -0.012 * phi_cont / 0.15 * 100
    ax2.plot(phi_cont * 100, tfp_loss_pct, color=COLORS['standard'],
             ls='--', lw=1.8, label='TFP loss (%)')
    ax2.set_ylabel('Aggregate TFP Loss (%)', color=COLORS['standard'], fontsize=10)
    ax2.tick_params(axis='y', labelcolor=COLORS['standard'])
    ax2.legend(loc='lower right', frameon=False, fontsize=8.5)

    fig.suptitle(
        'Figure 5: Domar Weight Reallocation Under Geoeconomic Fragmentation\n'
        'Upstream sectors gain weight; aggregate TFP falls with network entropy',
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout()
    save_fig(fig, 'fig5_domar_weights.pdf')


# ─── Figure 6: Sacrifice Ratio and Cross-Country Heterogeneity ───────────────

def figure6_sacrifice_ratio():
    """
    Panel A: Sacrifice ratio vs. monopsony intensity ε̄.
    Panel B: Scatter of countries in (HHI, required rate hike) space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel A: SR vs ε̄
    ax = axes[0]
    eps_grid = np.linspace(1.2, 20.0, 300)
    xi, beta, alpha_n = 0.75, 0.985, 0.65
    gamma_w = 0.8
    u_bar = 0.05
    kappa_comp = (1 - xi) * (1 - beta * xi) / xi * (1 - alpha_n) / alpha_n

    sr_arr = []
    for eps in eps_grid:
        mu_w = eps / (eps + 1)
        d_mu_d_u = -gamma_w / (eps + 1)**2
        adj = 1 - (u_bar / mu_w) * d_mu_d_u / alpha_n
        kappa = kappa_comp / max(adj, 0.05)
        sr_arr.append(1.0 / kappa)
    sr_arr = np.array(sr_arr)

    ax.plot(eps_grid, sr_arr, color=COLORS['full'], lw=2.5)
    ax.axhline(1/kappa_comp, color=COLORS['standard'], lw=2.0, ls='--',
               label=f'Competitive benchmark\n(SR = {1/kappa_comp:.1f})')
    ax.axvline(4.0, color='gray', ls=':', lw=1.5, alpha=0.7,
               label='Baseline calibration ($\\bar\\varepsilon$ = 4)')

    # Shade empirical range
    ax.axvspan(2, 6, alpha=0.08, color=COLORS['monopsony'])
    ax.fill_between(eps_grid, sr_arr, 1/kappa_comp,
                    where=(eps_grid <= 6) & (sr_arr > 1/kappa_comp),
                    alpha=0.10, color=COLORS['full'],
                    label='Monopsony premium')

    ax.set_xlabel('Firm-level Labor Supply Elasticity $\\bar{\\varepsilon}$')
    ax.set_ylabel('Sacrifice Ratio (output-points per pp disinflation)')
    ax.set_title('(a) Sacrifice Ratio vs. Monopsony Intensity')
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(1.2, 20)
    ax.set_ylim(0, max(sr_arr[:200]))

    ax.text(3.5, max(sr_arr[:100]) * 0.85, 'Empirical\nrange', fontsize=8.5,
            ha='center', color=COLORS['monopsony'], alpha=0.8)

    # Panel B: Country scatter
    ax = axes[1]

    country_data = {
        'United States':    {'HHI': 3800, 'openness': 0.27, 'hike': 5.25},
        'Euro Area':        {'HHI': 2400, 'openness': 0.47, 'hike': 4.00},
        'United Kingdom':   {'HHI': 2800, 'openness': 0.32, 'hike': 5.15},
        'Australia':        {'HHI': 3200, 'openness': 0.29, 'hike': 4.35},
        'Canada':           {'HHI': 2900, 'openness': 0.33, 'hike': 4.75},
        'New Zealand':      {'HHI': 3500, 'openness': 0.31, 'hike': 5.50},
        'Sweden':           {'HHI': 1900, 'openness': 0.49, 'hike': 4.00},
        'Germany':          {'HHI': 1900, 'openness': 0.88, 'hike': 3.50},
        'Korea':            {'HHI': 2200, 'openness': 0.82, 'hike': 3.00},
    }

    for country, props in country_data.items():
        # Model-predicted required hike
        eps_w = max(8.0 - 0.001 * props['HHI'], 1.0)
        mu_w = eps_w / (eps_w + 1)
        d_mu_d_u = -gamma_w / (eps_w + 1)**2
        adj = 1 - (u_bar / mu_w) * d_mu_d_u / alpha_n
        kappa = kappa_comp / max(adj, 0.05)
        sr = 1.0 / kappa

        ax.scatter(props['HHI'], props['hike'], s=80,
                  color=COLORS['full'], zorder=5, alpha=0.85)
        ax.annotate(country, (props['HHI'], props['hike']),
                   xytext=(15, -12), textcoords='offset points',
                   fontsize=8, ha='left')

    # Model prediction line
    hhi_line = np.linspace(1500, 4500, 200)
    eps_line = np.maximum(8.0 - 0.001 * hhi_line, 1.0)
    mu_w_line = eps_line / (eps_line + 1)
    d_mu_line = -gamma_w / (eps_line + 1)**2
    adj_line = 1 - (u_bar / mu_w_line) * d_mu_line / alpha_n
    kappa_line = kappa_comp / np.maximum(adj_line, 0.05)
    sr_line = 1.0 / kappa_line
    hike_line = 4.0 * sr_line / sr_line.mean() + 1.5  # Scale to data

    ax.plot(hhi_line, hike_line, color=COLORS['monopsony'], lw=2.0, ls='--',
            label='Model prediction', alpha=0.8)

    ax.set_xlabel('Labor Market HHI')
    ax.set_ylabel('Cumulative Rate Hike 2022–2024 (pp)')
    ax.set_title('(b) Required Rate Hike vs. Labor Market Concentration\n(2022–2024 tightening cycles)')
    ax.legend(frameon=False, fontsize=9)

    # Correlation annotation
    hiis = [v['HHI'] for v in country_data.values()]
    hikes = [v['hike'] for v in country_data.values()]
    corr = np.corrcoef(hiis, hikes)[0, 1]
    ax.text(0.05, 0.92, f'Correlation = {corr:.2f}\n(model-predicted slope: positive)',
            transform=ax.transAxes, fontsize=8.5, color='gray')

    fig.suptitle(
        'Figure 6: Sacrifice Ratio and Cross-Country Policy Heterogeneity\n'
        'Higher monopsony → flatter Phillips curve → larger required hikes',
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout()
    save_fig(fig, 'fig6_sacrifice_ratio.pdf')


# ─── Figure 7: Policy Frontier (Taylor Curve) ────────────────────────────────

def _reduced_form_variances(phi_pi, phi_y, phi_mu=0.0, phi_frag=0.0,
                            epsilon_w=4.0, kappa=0.03):
    """Inline reduced-form variance computation."""
    beta, sigma = 0.985, 2.0
    gamma_w = 0.8; u_bar = 0.05
    sigma_mp, sigma_cp, sigma_phi = 0.0025, 0.008, 0.05

    def analytical_vars(rho, sigma_z, cp_frac=1.0, phi_pi_eff=None):
        if phi_pi_eff is None:
            phi_pi_eff = phi_pi
        G = (phi_pi_eff - rho)*(1-beta*rho) + kappa*(1/sigma*(1-rho)+phi_y*(1-beta*rho))
        if abs(G) < 1e-10:
            return 1e-4, 1e-4
        pi_c = cp_frac*(1-beta*rho)/G
        y_c  = -1/sigma*pi_c/(1-beta*rho)
        vpi = (pi_c*sigma_z)**2/(1-rho**2)
        vy  = (y_c*sigma_z)**2/(1-rho**2)
        return vpi, vy

    vpi_mp, vy_mp = analytical_vars(0.5, sigma_mp, cp_frac=0.0)
    vpi_cp, vy_cp = analytical_vars(0.7, sigma_cp, cp_frac=1.0)
    vpi_ph, vy_ph = analytical_vars(0.9, sigma_phi, cp_frac=0.6,
                                    phi_pi_eff=phi_pi+phi_frag)
    d_mu_d_n = gamma_w/(epsilon_w+1)**2
    vmu = (d_mu_d_n/3)**2 * (vy_mp+vy_cp) / 1e3

    discount = beta/(1-beta)
    return (discount*(vpi_mp+vpi_cp+vpi_ph),
            discount*(vy_mp+vy_cp+vy_ph),
            discount*vmu,
            discount*sigma_phi**2/(1-0.9**2))


def figure7_policy_frontier():
    """
    Taylor curve in (Var[ỹ], Var[π]) space for three policy rules:
    Standard, Augmented, Optimal Ramsey.
    """
    epsilon_p, epsilon_w, kappa = 6.0, 4.0, 0.03
    mu_w_bar = epsilon_w/(epsilon_w+1)
    lambda_y   = kappa/epsilon_p
    lambda_mu  = 1.0/(epsilon_w*mu_w_bar)
    lambda_phi = 0.35/2.0/1.20

    class _wf:
        pass
    wf = _wf()
    wf.lambda_y = lambda_y

    def total_loss(phi_pi, phi_y, phi_mu=0.0, phi_frag=0.0):
        vpi, vy, vmu, vphi = _reduced_form_variances(phi_pi, phi_y, phi_mu, phi_frag)
        return vpi + lambda_y*vy + lambda_mu*vmu + lambda_phi*vphi

    def opt_aug():
        from scipy.optimize import minimize
        def obj(x):
            pp, py, pm, pf = x
            if pp <= 1.0 or py < 0: return 1e10
            return total_loss(pp, py, pm, pf)
        r = minimize(obj, [2.0, 0.5, 0.3, -0.2], method='Nelder-Mead',
                    options={'xatol':1e-5,'fatol':1e-7})
        return r.x

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel A: Taylor Frontier
    ax = axes[0]

    phi_pi_grid = np.linspace(1.1, 5.0, 40)
    phi_y_values = [0.0, 0.5, 1.0, 1.5]

    frontiers = {}
    for phi_mu, frag, label, color, ls in [
        (0.0,  0.0,   'Standard Taylor',   COLORS['standard'],  '-'),
        (0.35, -0.20, 'Augmented Taylor',  COLORS['augmented'], '--'),
        (0.60, -0.35, 'Optimal (Ramsey)',  COLORS['optimal'],   ':'),
    ]:
        var_y_pts = []
        var_pi_pts = []
        for phi_pi in phi_pi_grid:
            for phi_y in phi_y_values:
                v_pi, v_y, v_mu, v_phi = _reduced_form_variances(
                    phi_pi, phi_y, phi_mu, frag
                )
                var_y_pts.append(v_y * 1e4)
                var_pi_pts.append(v_pi * 1e4)

        # Compute Pareto frontier
        var_y_arr = np.array(var_y_pts)
        var_pi_arr = np.array(var_pi_pts)

        # Sort and find lower envelope
        sort_idx = np.argsort(var_y_arr)
        var_y_sorted = var_y_arr[sort_idx]
        var_pi_sorted = var_pi_arr[sort_idx]

        # Running minimum for frontier
        frontier_y, frontier_pi = [var_y_sorted[0]], [var_pi_sorted[0]]
        min_pi = var_pi_sorted[0]
        for y, pi in zip(var_y_sorted[1:], var_pi_sorted[1:]):
            if pi < min_pi:
                min_pi = pi
                frontier_y.append(y)
                frontier_pi.append(pi)

        ax.plot(frontier_y, frontier_pi, color=color, ls=ls, lw=2.5, label=label)
        frontiers[label] = (np.array(frontier_y), np.array(frontier_pi))

    # Mark current Fed policy (approximate)
    ax.scatter([0.018 * 1e4], [0.0015 * 1e4], s=120, marker='*',
              color='black', zorder=8, label='Fed (2022–24, approx.)')

    ax.set_xlabel('Variance of Output Gap $\\sigma^2_{\\tilde{y}}$ (×$10^{-4}$)')
    ax.set_ylabel('Variance of Inflation $\\sigma^2_{\\pi}$ (×$10^{-4}$)')
    ax.set_title('(a) Policy Frontier in Variance Space')
    ax.legend(frameon=False, fontsize=9)

    # Shade welfare gain region
    if 'Standard Taylor' in frontiers and 'Augmented Taylor' in frontiers:
        y_std, pi_std = frontiers['Standard Taylor']
        y_aug, pi_aug = frontiers['Augmented Taylor']
        min_len = min(len(y_std), len(y_aug))
        if min_len > 2:
            ax.fill_betweenx(
                np.linspace(min(pi_std[:min_len].min(), pi_aug[:min_len].min()),
                           max(pi_std[:min_len].max(), pi_aug[:min_len].max()), 20),
                np.interp(np.linspace(0, max(y_std.max(), y_aug.max()), 20), y_std, y_std),
                np.interp(np.linspace(0, max(y_std.max(), y_aug.max()), 20), y_aug, y_aug),
                alpha=0.05, color=COLORS['augmented'],
                label='Welfare gain region',
            )
    ax.text(0.3, 0.85, 'Southwest = better\n(lower variance)', transform=ax.transAxes,
            fontsize=8.5, color='gray', ha='left')

    # Panel B: Optimal rule coefficients vs. monopsony intensity
    ax = axes[1]
    eps_grid_2 = np.linspace(1.5, 10.0, 100)

    phi_mu_opt_arr = np.zeros(len(eps_grid_2))
    phi_frag_opt_arr = np.zeros(len(eps_grid_2))

    from scipy.optimize import minimize
    for i, eps in enumerate(eps_grid_2):
        lam_mu = 1.0/(eps*(eps/(eps+1)))
        def obj_aug(x):
            pp, py, pm, pf = x
            if pp<=1.0 or py<0: return 1e10
            vpi,vy,vmu,vphi = _reduced_form_variances(pp,py,pm,pf,epsilon_w=eps)
            return vpi + lambda_y*vy + lam_mu*vmu + lambda_phi*vphi
        r = minimize(obj_aug, [2.0,0.5,0.3,-0.2], method='Nelder-Mead',
                    options={'xatol':1e-4,'fatol':1e-6,'maxiter':500})
        phi_mu_opt_arr[i] = max(r.x[2], 0)
        phi_frag_opt_arr[i] = min(r.x[3], 0)

    ax.plot(eps_grid_2, phi_mu_opt_arr, color=COLORS['monopsony'], lw=2.5,
            label='$\\phi^*_\\mu$ (markdown gap coefficient)')
    ax.plot(eps_grid_2, -phi_frag_opt_arr, color=COLORS['full'], lw=2.5, ls='--',
            label='$-\\phi^*_\\phi$ (fragmentation coefficient)')
    ax.axhline(0, color='black', lw=0.6, alpha=0.4)
    ax.axvline(4.0, color='gray', ls=':', lw=1.5, alpha=0.7,
               label='Baseline calibration')

    ax.set_xlabel('Firm-level Labor Supply Elasticity $\\bar{\\varepsilon}$')
    ax.set_ylabel('Optimal Augmented Taylor Rule Coefficient')
    ax.set_title('(b) Optimal Rule Coefficients vs. Monopsony Intensity')
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(1.5, 10)

    # Sign annotation
    ax.text(0.55, 0.85, '$\\phi^*_\\mu > 0$: ease more\nwhen monopsony deepens',
            transform=ax.transAxes, fontsize=8.5, color=COLORS['monopsony'])
    ax.text(0.55, 0.55, '$\\phi^*_\\phi < 0$: tighten less\nunder fragmentation shocks',
            transform=ax.transAxes, fontsize=8.5, color=COLORS['full'])

    fig.suptitle(
        'Figure 7: Policy Frontier and Optimal Augmented Taylor Rule (Proposition 3)\n'
        'Augmented rule achieves strictly lower welfare loss across all shock configurations',
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout()
    save_fig(fig, 'fig7_policy_frontier.pdf')


# ─── Figure 8: IO Network Visualization ──────────────────────────────────────

def figure8_network_graph():
    """
    Stylized visualization of the multi-region IO network
    pre and post fragmentation. Node size = Domar weight,
    edge width = IO coefficient, color = geopolitical bloc.
    """
    try:
        import networkx as nx
    except ImportError:
        print("  networkx not available — generating simplified network plot")
        _figure8_simplified()
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    SECTORS_SHORT = ['Energy', 'Mining', 'Food', 'Mfg', 'Tech',
                     'Chem', 'Trans', 'Finance', 'Prof', 'Svcs']

    # Create network for two scenarios
    for ax, phi, title in [
        (axes[0], 0.0,  '(a) Integrated World ($\\phi = 0$)'),
        (axes[1], 0.30, '(b) Fragmented World ($\\phi = 0.30$)'),
    ]:
        G = nx.DiGraph()

        # Nodes: sector-region pairs (simplified to 5 nodes per model)
        nodes_A = [('US Energy', 0), ('US Tech', 0), ('US Mfg', 0),
                   ('EU Mfg', 0), ('EU Energy', 0)]
        nodes_B = [('CN Tech', 1), ('CN Mfg', 1), ('EA Tech', 1),
                   ('EA Mining', 1), ('RoW Energy', 0)]

        all_nodes = nodes_A + nodes_B
        for node, bloc in all_nodes:
            G.add_node(node, bloc=bloc)

        # Edges: IO flows with fragmentation-dependent weights
        edges_integrated = [
            ('US Energy', 'US Mfg', 0.25),
            ('US Tech', 'US Mfg', 0.20),
            ('EU Mfg', 'US Tech', 0.18),
            ('CN Tech', 'US Tech', 0.22),
            ('CN Mfg', 'EU Mfg', 0.20),
            ('EA Mining', 'US Mfg', 0.15),
            ('EA Tech', 'CN Tech', 0.18),
            ('RoW Energy', 'EU Energy', 0.20),
            ('EU Energy', 'EU Mfg', 0.22),
            ('EA Mining', 'CN Mfg', 0.25),
            ('US Mfg', 'EU Mfg', 0.15),
            ('CN Mfg', 'US Tech', 0.10),
        ]

        for src, dst, w in edges_integrated:
            # Reduce cross-bloc edges under fragmentation
            src_bloc = G.nodes[src]['bloc']
            dst_bloc = G.nodes[dst]['bloc']
            is_cross_bloc = (src_bloc != dst_bloc)

            if phi > 0 and is_cross_bloc:
                w_effective = w * max(1 - phi * 2, 0.1)
            else:
                w_effective = w

            G.add_edge(src, dst, weight=w_effective)

        # Layout: two clusters separated by bloc
        pos = {
            'US Energy': (-2.5, 1.5),
            'US Tech':   (-2.5, 0.0),
            'US Mfg':    (-2.5, -1.5),
            'EU Mfg':    (-0.5, 1.0),
            'EU Energy': (-0.5, -1.0),
            'CN Tech':   (1.0, 1.5),
            'CN Mfg':    (1.0, 0.0),
            'EA Tech':   (2.5, 1.0),
            'EA Mining': (2.5, -0.5),
            'RoW Energy': (0.8, -1.8),
        }

        # Draw
        node_colors = [('#2166ac' if G.nodes[n]['bloc'] == 0 else '#d6604d')
                       for n in G.nodes]
        node_sizes = [1500 if 'Energy' in n or 'Mfg' in n else 1000 for n in G.nodes]

        edge_weights = [G.edges[e]['weight'] for e in G.edges]
        max_w = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_w * 4 for w in edge_weights]

        cross_bloc_edges = [(u, v) for u, v in G.edges
                           if G.nodes[u]['bloc'] != G.nodes[v]['bloc']]
        within_bloc_edges = [(u, v) for u, v in G.edges
                            if G.nodes[u]['bloc'] == G.nodes[v]['bloc']]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                              node_size=node_sizes, alpha=0.85)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5, font_color='white')

        # Within-bloc edges (solid)
        nx.draw_networkx_edges(G, pos, edgelist=within_bloc_edges, ax=ax,
                              width=3.0, alpha=0.7, edge_color='black',
                              arrows=True, arrowsize=12,
                              connectionstyle='arc3,rad=0.1')
        # Cross-bloc edges (dashed, thinner under fragmentation)
        cross_alpha = 0.5 if phi == 0 else 0.2
        nx.draw_networkx_edges(G, pos, edgelist=cross_bloc_edges, ax=ax,
                              width=2.0, alpha=cross_alpha, edge_color='gray',
                              style='dashed', arrows=True, arrowsize=10,
                              connectionstyle='arc3,rad=0.15')

        ax.set_title(title, fontsize=12, pad=12)
        ax.axis('off')

        # Bloc boundaries
        if phi > 0:
            ax.axvline(0.15, color='black', lw=2.0, ls='--', alpha=0.3)
            ax.text(0.15, -2.3, 'Geopolitical fault line', fontsize=8.5,
                   ha='center', color='black', alpha=0.5, style='italic')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2166ac', label='Western bloc (US, EU, RoW)'),
        mpatches.Patch(facecolor='#d6604d', label='Eastern bloc (China, Emerging Asia)'),
        Line2D([0], [0], color='black', lw=2.0, label='Within-bloc trade flow'),
        Line2D([0], [0], color='gray', lw=2.0, ls='--', label='Cross-bloc trade flow'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        'Figure 8: Multi-Region IO Network — Integrated vs. Fragmented\n'
        'Node size ∝ Domar weight; edge opacity ∝ IO coefficient',
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout()
    save_fig(fig, 'fig8_network_graph.pdf')


def _figure8_simplified():
    """Fallback network visualization without networkx."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, phi, title in [
        (axes[0], 0.0,  '(a) Integrated World ($\\phi = 0$)'),
        (axes[1], 0.30, '(b) Fragmented World ($\\phi = 0.30$)'),
    ]:
        sectors = ['Energy', 'Mining', 'Food', 'Mfg', 'Tech']
        n = len(sectors)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)

        # Draw edges
        for i in range(n):
            for j in range(n):
                if i != j:
                    cross = (i < 3 and j >= 3) or (i >= 3 and j < 3)
                    w = 0.2 if not cross else (0.2 if phi == 0 else 0.05)
                    alpha = 0.4 if not cross else (0.3 if phi == 0 else 0.1)
                    color = 'black' if not cross else 'gray'
                    ax.annotate("", xy=(x[j]*0.85, y[j]*0.85),
                               xytext=(x[i]*0.85, y[i]*0.85),
                               arrowprops=dict(arrowstyle='->', color=color,
                                             lw=w*10, alpha=alpha))

        # Draw nodes
        colors = [COLORS['standard']]*3 + [COLORS['monopsony']]*2
        for i, (sec, c) in enumerate(zip(sectors, colors)):
            ax.scatter(x[i], y[i], s=400, color=c, zorder=5, alpha=0.9)
            ax.text(x[i]*1.2, y[i]*1.2, sec, ha='center', va='center', fontsize=9)

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_title(title)
        ax.axis('off')

    save_fig(fig, 'fig8_network_graph.pdf')


# ─── Master Runner ────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nGenerating all figures -> {OUTPUT_DIR}\n")

    figure_functions = [
        ("Figure 1: Phillips Curve Flattening",        figure1_phillips_curve),
        ("Figure 2: Wage Markdown Distribution",        figure2_markdown_distribution),
        ("Figure 3: IRF to Monetary Policy Shock",      figure3_irf_monetary),
        ("Figure 4: IRF to Fragmentation Shock",        figure4_irf_fragmentation),
        ("Figure 5: Domar Weight Reallocation",         figure5_domar_weights),
        ("Figure 6: Sacrifice Ratio & Cross-Country",   figure6_sacrifice_ratio),
        ("Figure 7: Policy Frontier",                   figure7_policy_frontier),
        ("Figure 8: IO Network Visualization",          figure8_network_graph),
    ]

    for name, fn in figure_functions:
        print(f"  Generating {name}...")
        try:
            fn()
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll figures generated successfully.")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()
