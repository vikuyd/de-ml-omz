"""
02_gp_reconstruction.py
─────────────────────────────────────────────────────────────
Gaussian Process reconstruction of H(z) and Om(z) for all
four dataset combinations.

The GP is trained on OHD H(z) in all cases.
For OHD+PP, OHD+PP+SH0ES, and +DESI:
  - The MCMC best-fit parameters serve as reference curves
  - The GP H0 is evaluated at z→0
  - Pantheon and DESI enter only through the MCMC comparison;
    the GP itself is a pure OHD reconstruction
  (This is clearly stated in the paper methodology section)

Kernel: Matern-5/2 (primary), with sensitivity test vs
        Matern-3/2 and RBF.
MC samples: N=5000 for Om(z) uncertainty propagation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import george
from george import kernels

from data.ohd_data      import get_ohd_data
from models.cosmo_model import (Om_parametric, Om_from_H,
                                  H_parametric, H_LCDM,
                                  find_transition_z)

np.random.seed(42)

# ════════════════════════════════════════════════════════════
# DATA AND SETTINGS
# ════════════════════════════════════════════════════════════

z_ohd, H_ohd, sH_ohd = get_ohd_data()
H0_NORM = 100.0
Z_PRED  = np.linspace(0.02, 2.5, 600)
N_MC    = 5000

COMBOS = ["OHD", "OHD_PP", "OHD_PP_SH0ES",
          "OHD_PP_SH0ES_DESI"]
LABELS = {
    "OHD":               "OHD",
    "OHD_PP":            "OHD+PP",
    "OHD_PP_SH0ES":      "OHD+PP+SH0ES",
    "OHD_PP_SH0ES_DESI": "OHD+PP+SH0ES+DESI",
}

# Default fallback params if MCMC not yet run
DEFAULTS = {
    "OHD":               (72.12, 0.728, 1.823),
    "OHD_PP":            (72.73, 0.728, 1.823),
    "OHD_PP_SH0ES":      (73.01, 0.451, 1.810),
    "OHD_PP_SH0ES_DESI": (73.10, 0.440, 1.790),
}


# ════════════════════════════════════════════════════════════
# GP UTILITIES
# ════════════════════════════════════════════════════════════

def build_gp(kernel_name, z_data, H_norm, He_norm):
    """Build and optimise a GP with the given kernel."""
    var = float(np.var(H_norm))

    if kernel_name == "matern52":
        kern = var * kernels.Matern52Kernel(
            metric=0.5, ndim=1)
    elif kernel_name == "matern32":
        kern = var * kernels.Matern32Kernel(
            metric=0.5, ndim=1)
    elif kernel_name == "rbf":
        kern = var * kernels.ExpSquaredKernel(
            metric=0.5, ndim=1)
    else:
        raise ValueError(kernel_name)

    gp = george.GP(kern,
                   mean=float(np.mean(H_norm)),
                   fit_mean=True)
    gp.compute(z_data, He_norm)

    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(H_norm, quiet=True)
        return -ll if np.isfinite(ll) else 1e10

    def gnll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(H_norm, quiet=True)

    best = None
    for _ in range(6):
        p0 = (gp.get_parameter_vector()
              + 0.5*np.random.randn(
                  len(gp.get_parameter_vector())))
        r  = minimize(nll, p0, jac=gnll,
                      method="L-BFGS-B",
                      options={"maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r

    gp.set_parameter_vector(best.x)
    return gp, -best.fun


def reconstruct(gp, H_norm, z_pred, N_mc=5000):
    """
    Predict H(z) from GP and propagate into Om(z).
    Returns dict of arrays.
    """
    mu_n, cov_n = gp.predict(H_norm, z_pred,
                               return_cov=True)
    # Ensure positive definite
    cov_n += 1e-10 * np.eye(len(z_pred))
    var_n  = np.diag(cov_n)

    H_mean = mu_n * H0_NORM
    H_std  = np.sqrt(np.abs(var_n)) * H0_NORM

    # MC draws from posterior
    try:
        draws = np.random.multivariate_normal(
            mu_n, cov_n, size=N_mc)
    except np.linalg.LinAlgError:
        draws = np.random.normal(
            mu_n, np.sqrt(np.abs(var_n)),
            size=(N_mc, len(z_pred)))

    H_draws = draws * H0_NORM

    # Om(z) for each draw
    Om_list, H0_list = [], []
    for Hs in H_draws:
        H0s = float(Hs[0])
        if H0s <= 30:
            continue
        H0_list.append(H0s)
        Om_list.append(Om_from_H(z_pred, Hs, H0s))

    Om_arr = np.array(Om_list)
    valid  = np.all(np.isfinite(Om_arr), axis=1)
    Om_arr = Om_arr[valid]
    H0_arr = np.array(H0_list)[valid]

    return dict(
        H_mean  = H_mean,
        H_std   = H_std,
        Om_mean = Om_arr.mean(axis=0),
        Om_std  = Om_arr.std(axis=0),
        Om_samples = Om_arr[:500],
        H_draws    = H_draws[:500],
        H0_mean = float(H0_arr.mean()),
        H0_std  = float(H0_arr.std()),
    )


def bootstrap_zt(Om_samples, z_pred):
    """Bootstrap distribution of transition redshift."""
    zt_list = []
    for Om_s in Om_samples:
        zts = find_transition_z(z_pred, Om_s)
        if len(zts) > 0 and 0.05 < zts[0] < 3.0:
            zt_list.append(float(zts[0]))
    if len(zt_list) < 10:
        return np.nan, np.nan
    return float(np.median(zt_list)), float(np.std(zt_list))


# ════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════

gp_results = {}
H_norm  = H_ohd  / H0_NORM
He_norm = sH_ohd / H0_NORM

for combo in COMBOS:
    print(f"\n{'═'*60}")
    print(f"  GP: {LABELS[combo]}")
    print(f"{'═'*60}")

    os.makedirs(f"results/{combo}", exist_ok=True)
    os.makedirs(f"figures/{combo}", exist_ok=True)

    # Load best-fit params
    try:
        bf = np.load(f"results/{combo}/best_fit.npy")
        H0_p, l_p, m_p = bf[0], bf[1], bf[2]
    except FileNotFoundError:
        H0_p, l_p, m_p = DEFAULTS[combo]
        print(f"  [warn] Using default params.")

    # ── Primary GP (Matern52) ─────────────────────────────
    print("  Optimising GP (Matern-5/2) ...")
    gp_m52, ll_m52 = build_gp(
        "matern52", z_ohd, H_norm, He_norm)
    print(f"  log-likelihood = {ll_m52:.4f}")

    res = reconstruct(gp_m52, H_norm, Z_PRED, N_mc=N_MC)
    zt_val, zt_err = bootstrap_zt(
        res["Om_samples"], Z_PRED)

    print(f"  H0(GP) = {res['H0_mean']:.2f}"
          f" ± {res['H0_std']:.2f}")
    print(f"  z_t(GP) = {zt_val:.3f} ± {zt_err:.3f}")

    # ── Kernel sensitivity ────────────────────────────────
    kern_res = {}
    for kname in ["matern32", "matern52", "rbf"]:
        gp_k, ll_k = build_gp(
            kname, z_ohd, H_norm, He_norm)
        mu_k, var_k = gp_k.predict(
            H_norm, Z_PRED, return_var=True)
        H_k  = mu_k * H0_NORM
        H0_k = max(float(H_k[0]), 30.0)
        Om_k = Om_from_H(Z_PRED, H_k, H0_k)
        zts_k = find_transition_z(Z_PRED, Om_k)
        zt_k  = (float(zts_k[0])
                 if len(zts_k)>0 and 0.05<zts_k[0]<3.0
                 else np.nan)
        kern_res[kname] = dict(
            Om=Om_k, H=H_k, zt=zt_k, ll=ll_k)
        print(f"    {kname:10s}: "
              f"ll={ll_k:.3f}  z_t={zt_k:.3f}")

    # ── Save ──────────────────────────────────────────────
    np.save(f"results/{combo}/gp_z.npy",         Z_PRED)
    np.save(f"results/{combo}/gp_H_mean.npy",    res["H_mean"])
    np.save(f"results/{combo}/gp_H_std.npy",     res["H_std"])
    np.save(f"results/{combo}/gp_Om_mean.npy",   res["Om_mean"])
    np.save(f"results/{combo}/gp_Om_std.npy",    res["Om_std"])
    np.save(f"results/{combo}/gp_Om_samples.npy",res["Om_samples"])
    np.save(f"results/{combo}/gp_zt.npy",
            np.array([zt_val, zt_err]))
    np.save(f"results/{combo}/gp_H0.npy",
            np.array([res["H0_mean"], res["H0_std"]]))

    gp_results[combo] = dict(
        label   = LABELS[combo],
        H_mean  = res["H_mean"],
        H_std   = res["H_std"],
        Om_mean = res["Om_mean"],
        Om_std  = res["Om_std"],
        Om_samples = res["Om_samples"],
        H_draws    = res["H_draws"],
        zt      = zt_val,
        zt_err  = zt_err,
        H0      = res["H0_mean"],
        H0_std  = res["H0_std"],
        kern    = kern_res,
    )

    # ── Figures ───────────────────────────────────────────
    z_fine   = np.linspace(0.01, 3.0, 600)
    Om_paper = Om_parametric(z_fine, l_p, m_p)

    # Fig A: H(z)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.errorbar(z_ohd, H_ohd, yerr=sH_ohd,
                fmt='o', ms=5, color='royalblue',
                capsize=3, label='OHD', zorder=6)
    ax.plot(Z_PRED, res["H_mean"],
            'forestgreen', lw=2, label='GP mean')
    ax.fill_between(Z_PRED,
        res["H_mean"]-res["H_std"],
        res["H_mean"]+res["H_std"],
        color='forestgreen', alpha=0.22, label='GP 1σ')
    ax.fill_between(Z_PRED,
        res["H_mean"]-2*res["H_std"],
        res["H_mean"]+2*res["H_std"],
        color='forestgreen', alpha=0.08)
    ax.plot(z_fine,
            H_parametric(z_fine, H0_p, l_p, m_p),
            'r--', lw=1.8, label='Parametric')
    ax.plot(z_fine, H_LCDM(z_fine, H0_p),
            'k:', lw=1.5, label=r'$\Lambda$CDM')
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel(r'$H(z)$ [km/s/Mpc]', fontsize=12)
    ax.set_title(f'GP $H(z)$ — {LABELS[combo]}',
                  fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.5)
    fig.tight_layout()
    fig.savefig(f"figures/{combo}/GP_Hz_{combo}.pdf",
                dpi=150)
    plt.close()

    # Fig B: Om(z)
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.axhspan(0.29, 0.31, color='lightgray', alpha=0.4)
    ax2.axhline(0.30, color='gray', ls=':', lw=1.3,
                label=r'$\Lambda$CDM ($\Omega_m=0.30$)')
    ax2.plot(z_fine, Om_paper, 'r--', lw=2,
             label='Parametric model')
    ax2.plot(Z_PRED, res["Om_mean"],
             'forestgreen', lw=2, label='GP mean')
    ax2.fill_between(Z_PRED,
        res["Om_mean"]-res["Om_std"],
        res["Om_mean"]+res["Om_std"],
        color='forestgreen', alpha=0.28, label='GP 1σ')
    ax2.fill_between(Z_PRED,
        res["Om_mean"]-2*res["Om_std"],
        res["Om_mean"]+2*res["Om_std"],
        color='forestgreen', alpha=0.10, label='GP 2σ')
    if not np.isnan(zt_val):
        ax2.axvline(zt_val, color='forestgreen',
                    ls='--', lw=1.5,
                    label=f'$z_t^{{\\rm GP}}='
                          f'{zt_val:.2f}\\pm{zt_err:.2f}$')
    ax2.annotate('Phantom\n($w<-1$)',
                  xy=(0.15, 0.55), fontsize=9,
                  color='crimson', alpha=0.8)
    ax2.annotate('Quintessence\n($w>-1$)',
                  xy=(2.0, 0.05), fontsize=9,
                  color='navy', alpha=0.8)
    ax2.set_xlabel('Redshift $z$', fontsize=12)
    ax2.set_ylabel(r'$Om(z)$', fontsize=12)
    ax2.set_title(
        f'$Om(z)$ Diagnostic — GP — {LABELS[combo]}',
        fontsize=13)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 3.0)
    ax2.set_ylim(-0.20, 0.70)
    fig2.tight_layout()
    fig2.savefig(f"figures/{combo}/GP_Omz_{combo}.pdf",
                 dpi=150)
    plt.close()

    # Fig C: kernel sensitivity
    fig3, ax3 = plt.subplots(figsize=(8,5))
    cols = {"matern32":"steelblue",
            "matern52":"forestgreen",
            "rbf":"darkorchid"}
    for kn, kr in kern_res.items():
        lbl = (f'{kn} ($z_t={kr["zt"]:.2f}$)'
               if not np.isnan(kr["zt"]) else kn)
        ax3.plot(Z_PRED, kr["Om"],
                 color=cols[kn], lw=2, label=lbl)
    ax3.plot(z_fine, Om_paper, 'r--', lw=2,
             label='Parametric')
    ax3.axhline(0.30, color='gray', ls=':', lw=1.2)
    ax3.set_xlabel('$z$', fontsize=12)
    ax3.set_ylabel(r'$Om(z)$', fontsize=12)
    ax3.set_title(
        f'Kernel Sensitivity — {LABELS[combo]}',
        fontsize=13)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 3); ax3.set_ylim(-0.20, 0.70)
    fig3.tight_layout()
    fig3.savefig(f"figures/{combo}/GP_kernels_{combo}.pdf",
                 dpi=150)
    plt.close()

    print(f"  Figures saved → figures/{combo}/")

np.save("results/gp_results.npy", gp_results)
print("\n[02_gp_reconstruction.py] Complete.")
