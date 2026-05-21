# Monetary Policy in Fractured Markets
## A Unified HANK-IO Framework with Structural Monopsony and Geoeconomic Fragmentation

**Author:** Brendan Loalt  
**Status:** Working Paper  
**Target:** *American Economic Review* / *Journal of Political Economy*

---

## Abstract

We build a quantitative general equilibrium model integrating three features absent from standard frameworks: pervasive firm-level monopsony power in labor markets, heterogeneous households facing incomplete insurance markets, and multi-bloc trade networks subject to time-varying geopolitical frictions. Solving via sequence-space Jacobians, we establish that monopsony endogenously flattens the aggregate Phillips curve, supply-side shocks propagate with greater amplitude in fragmented IO networks, and the standard Taylor rule is welfare-dominated by an augmented rule conditioning on labor market concentration and trade-bloc network entropy. The framework rationalizes the post-2021 inflation surge, unemployment resilience to rate hikes, and asymmetric cross-country shock transmission.

---

## Project Structure

```
Monops-Eq/
├── README.md
├── paper/
│   ├── main.tex              # Master LaTeX file
│   ├── references.bib        # Full bibliography (50+ references)
│   ├── main.pdf              # Compiled manuscript
│   ├── sections/             # Individual section .tex files
│   │   ├── abstract.tex
│   │   ├── 01_introduction.tex
│   │   ├── 02_model.tex
│   │   ├── 03_equilibrium.tex
│   │   ├── 04_solution_method.tex
│   │   ├── 05_analytical_results.tex
│   │   ├── 06_quantitative.tex
│   │   ├── 07_policy.tex
│   │   ├── 08_conclusion.tex
│   │   └── appendix.tex
│   └── figures/              # All manuscript figures (PDF format)
│       ├── fig1_phillips_curve.pdf
│       ├── fig2_markdown_cyclical.pdf
│       ├── fig3_irf_monetary.pdf
│       ├── fig4_irf_fragmentation.pdf
│       ├── fig5_domar_weights.pdf
│       ├── fig6_sacrifice_ratio.pdf
│       ├── fig7_policy_frontier.pdf
│       └── fig8_network_graph.pdf
├── code/
│   ├── steady_state/         # Steady-state computation
│   │   ├── household.py      # EGM/VFI household problem
│   │   ├── firm.py           # Monopsonistic firm problem
│   │   └── io_network.py     # Multi-region IO equilibrium
│   ├── sequence_space/       # Sequence-space Jacobian machinery
│   │   ├── jacobians.py      # Core Jacobian computation
│   │   └── equilibrium.py    # System Jacobian assembly and solution
│   ├── quantitative/         # Quantitative exercises
│   │   ├── calibration.py    # SMM calibration
│   │   ├── exercises.py      # IRF exercises and counterfactuals
│   │   └── welfare.py        # Welfare analysis and optimal rule
│   └── figures/
│       └── generate_all_figures.py  # Master figure generation script
└── data/
    └── calibration_targets.json     # Empirical calibration targets
```

## Requirements

### Python
```
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
networkx>=3.1
```

Install via:
```bash
pip install numpy scipy matplotlib seaborn networkx
```

### LaTeX
- MiKTeX or TeX Live with standard packages
- Compile with: `cd paper && latexmk -pdf main.tex`

## Running the Code

### Generate All Figures
```bash
cd code/figures
python generate_all_figures.py
```

### Reproduce Full Analysis
```bash
cd code
python -m quantitative.calibration    # Calibrate model
python -m quantitative.exercises      # Run IRF exercises  
python -m quantitative.welfare        # Compute welfare analysis
```

### Compile Manuscript
```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

## Key Results

1. **Proposition 1 (Phillips Curve Flattening):** The monopsony-adjusted NKPC slope satisfies κ^mono < κ^comp, with the gap increasing in countercyclical markdown intensity γ.

2. **Proposition 2 (Fragmentation Amplification):** The GDP-fragmentation semi-elasticity is negative and increasing in magnitude in the network centrality of cross-bloc linkages.

3. **Proposition 3 (Augmented Taylor Rule):** The welfare-optimal monetary rule augments the standard Taylor rule with terms in the wage markdown gap and fragmentation premium, with strictly positive optimal coefficients.

## Citation

```bibtex
@unpublished{loalt2025monetary,
  title   = {Monetary Policy in Fractured Markets: A Unified {HANK-IO} Framework 
             with Structural Monopsony and Geoeconomic Fragmentation},
  author  = {Loalt, Brendan},
  year    = {2025},
  note    = {Working Paper}
}
```
