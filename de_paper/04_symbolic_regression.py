"""
04_symbolic_regression.py
─────────────────────────────────────────────────────────────
Symbolic regression on the GP-reconstructed Om(z) curve
using PySR (Cranmer et al. 2020).

Scientific question:
  Does the data independently suggest Om(z) = z^l/(1+z)^m ?

Input  : GP Om(z) mean and std from results/OHD_PP_SH0ES_DESI/
Output :
  results/symreg_equations.csv   — all discovered equations
  results/symreg_best.npy        — best expression metadata
  figures/master/Fig_symreg.pdf  — pareto front + best fit
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════════
# CHECK PYSR AVAILABLE
# ════════════════════════════════════════════════════════════
try:
    from pysr import PySRRegressor
    PYSR_OK = True
except ImportError:
    PYSR_OK = False
    print("[symreg] PySR not installed.")
    print("[symreg] Install with: pip install pysr")
    print("[symreg] Skipping symbolic regression.")

from models.cosmo_model import Om_parametric

os.makedirs("results",       exist_ok=True)
os.makedirs("figures/master", exist_ok=True)


# ════════════════════════════════════════════════════════════
# LOAD GP RESULTS
# ════════════════════════════════════════════════════════════

# Use OHD+PP+SH0ES+DESI combo — most constrained
COMBO = "OHD_PP_SH0ES_DESI"

def _load_gp(combo):
    z    = np.load(f"results/{combo}/gp_z.npy")
    Om_m = np.load(f"results/{combo}/gp_Om_mean.npy")
    Om_s = np.load(f"results/{combo}/gp_Om_std.npy")
    return z, Om_m, Om_s

try:
    z_full, Om_full, Om_std = _load_gp(COMBO)
    print(f"[symreg] Loaded GP results from {COMBO}")
except FileNotFoundError:
    # Fall back to OHD
    try:
        z_full, Om_full, Om_std = _load_gp("OHD")
        COMBO = "OHD"
        print(f"[symreg] Loaded GP results from OHD")
    except FileNotFoundError:
        print("[symreg] No GP results found. "
              "Run 02_gp_reconstruction.py first.")
        sys.exit(0)


# ════════════════════════════════════════════════════════════
# PREPARE DATA FOR PYSR
# ════════════════════════════════════════════════════════════

# Use z > 0.05 to avoid near-zero issues
mask   = (z_full > 0.05) & np.isfinite(Om_full)
z_sr   = z_full[mask]
Om_sr  = Om_full[mask]
std_sr = Om_std[mask]
std_sr = np.maximum(std_sr, 1e-4)   # prevent zero weights

# Weights inversely proportional to uncertainty
weights = 1.0 / std_sr

print(f"[symreg] Data points for SR: {len(z_sr)}")
print(f"[symreg] z range: [{z_sr.min():.3f}, "
      f"{z_sr.max():.3f}]")


# ════════════════════════════════════════════════════════════
# RUN PYSR
# ════════════════════════════════════════════════════════════

if PYSR_OK:
    print("\n[symreg] Starting PySR ...")
    print("[symreg] This takes 5–15 minutes.")

    model_sr = PySRRegressor(
        niterations      = 50,
        binary_operators = ["+", "-", "*", "/", "^"],
        unary_operators  = ["log", "exp", "sqrt",
                             "abs"],
        populations      = 30,
        population_size  = 50,
        maxsize          = 20,
        parsimony        = 0.001,
        weight_optimize  = 0.001,
        turbo            = True,
        verbosity        = 1,
        random_state     = 42,
        deterministic    = True,
        procs            = 0,      # single process
        multithreading   = False,
        temp_equation_file = True,
        output_jax_format  = False,
        equation_file    = "results/symreg_equations.csv",
    )

    X = z_sr.reshape(-1, 1)
    model_sr.fit(X, Om_sr,
                  weights=weights,
                  variable_names=["z"])

    # ── Results ───────────────────────────────────────────
    print("\n[symreg] Top equations (Pareto front):")
    print(model_sr)

    best_eq = model_sr.get_best()
    print(f"\n[symreg] Best equation: "
          f"{best_eq['equation']}")
    print(f"[symreg] Complexity: {best_eq['complexity']}")
    print(f"[symreg] Loss: {best_eq['loss']:.6f}")

    np.save("results/symreg_best.npy",
            {"equation": best_eq["equation"],
             "complexity": best_eq["complexity"],
             "loss": best_eq["loss"]})

    Om_sr_pred = model_sr.predict(X)

else:
    # Fallback: show what PySR would compare against
    print("[symreg] Using reference parametric curve "
          "as placeholder.")
    # Load MCMC params for best-fit comparison
    try:
        bf = np.load(
            f"results/{COMBO}/best_fit.npy")
        l_ref, m_ref = bf[1], bf[2]
    except FileNotFoundError:
        l_ref, m_ref = 0.45, 1.81
    Om_sr_pred = Om_parametric(z_sr, l_ref, m_ref)
    best_eq = {"equation": f"z^{l_ref:.3f}/(1+z)^{m_ref:.3f}",
               "complexity": 5, "loss": np.nan}


# ════════════════════════════════════════════════════════════
# LOAD MCMC BEST-FIT FOR COMPARISON
# ════════════════════════════════════════════════════════════

try:
    bf    = np.load(f"results/{COMBO}/best_fit.npy")
    H0_p, l_p, m_p = bf[0], bf[1], bf[2]
except FileNotFoundError:
    H0_p, l_p, m_p = 73.0, 0.45, 1.81

Om_param = Om_parametric(z_sr, l_p, m_p)


# ════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Left: GP data + SR fit + parametric ──────────────────
ax = axes[0]
ax.fill_between(z_sr,
                Om_sr - std_sr,
                Om_sr + std_sr,
                color='forestgreen', alpha=0.25,
                label='GP 1σ')
ax.plot(z_sr, Om_sr, 'forestgreen', lw=2,
        label='GP mean')
ax.plot(z_sr, Om_sr_pred, 'tomato', lw=2, ls='--',
        label=f'SR best fit:\n'
              f'$\\hat{{e}}={best_eq["equation"]}$')
ax.plot(z_sr, Om_param, 'k:', lw=1.8,
        label=f'Parametric: '
              f'$z^{{{l_p:.3f}}}/(1+z)^{{{m_p:.3f}}}$')
ax.axhline(0.30, color='gray', ls=':', lw=1.0,
           alpha=0.6)
ax.set_xlabel('Redshift $z$', fontsize=12)
ax.set_ylabel(r'$Om(z)$', fontsize=12)
ax.set_title('SR vs GP reconstruction', fontsize=13)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 2.5)
ax.set_ylim(-0.20, 0.70)

# ── Right: Pareto front (complexity vs loss) ─────────────
ax2 = axes[1]
if PYSR_OK:
    eqs = model_sr.equations_
    if eqs is not None and len(eqs) > 0:
        compl = eqs["complexity"].values
        loss  = eqs["loss"].values
        ax2.scatter(compl, np.log10(loss+1e-12),
                    c='steelblue', s=60,
                    edgecolors='black', lw=0.5,
                    zorder=5)
        # Highlight best
        bc = best_eq["complexity"]
        bl = best_eq["loss"]
        ax2.scatter([bc], [np.log10(bl+1e-12)],
                    c='red', s=120, zorder=6,
                    label='Best equation')
        ax2.legend(fontsize=9)
    ax2.set_xlabel('Complexity', fontsize=12)
    ax2.set_ylabel('$\\log_{10}$ Loss', fontsize=12)
    ax2.set_title('Pareto Front', fontsize=13)
    ax2.grid(True, alpha=0.3)
else:
    # Residuals plot instead
    resid_sr    = Om_sr - Om_sr_pred
    resid_param = Om_sr - Om_param
    ax2.plot(z_sr, resid_sr,    'tomato',
             lw=2, label='SR residuals')
    ax2.plot(z_sr, resid_param, 'k--',
             lw=1.5, label='Parametric residuals')
    ax2.axhline(0, color='gray', lw=1.0)
    ax2.fill_between(z_sr, -std_sr, std_sr,
                     color='forestgreen', alpha=0.15,
                     label='GP 1σ band')
    ax2.set_xlabel('$z$', fontsize=12)
    ax2.set_ylabel(r'Residuals $\Delta Om(z)$',
                   fontsize=12)
    ax2.set_title('Residuals vs GP', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

fig.suptitle(
    'Symbolic Regression on GP-Reconstructed $Om(z)$',
    fontsize=14)
fig.tight_layout()
fig.savefig("figures/master/Fig_symreg.pdf",
            bbox_inches='tight', dpi=200)
plt.close()
print("\n[symreg] Figure saved → "
      "figures/master/Fig_symreg.pdf")
print("[04_symbolic_regression.py] Complete.")
