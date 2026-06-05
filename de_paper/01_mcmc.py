"""
01_mcmc.py  —  FAST VERSION
─────────────────────────────────────────────────────────────
Optimised for speed:
  - 32 walkers  (was 80)
  - 3000 steps  (was 10000)
  - 500 burn-in (was 2000)
  - Fast Pantheon chi2 via grid interpolation (~100x faster)
  - Multiprocessing (all CPU cores)

Expected runtime:
  OHD only              ~5  min
  OHD + PP              ~20 min
  OHD + PP + SH0ES      ~25 min
  OHD + PP+SH0ES+DESI   ~30 min
  Total                 ~80 min
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import emcee
import corner
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import time

from data.ohd_data      import get_ohd_data
from data.pantheon_data import get_pantheon_data
from data.desi_data     import chi2_desi, RD_PLANCK
from models.cosmo_model import (H_parametric, H_LCDM,
                                  chi2_ohd, AIC, BIC,
                                  age_Gyr)

np.random.seed(42)

# ── MCMC settings ────────────────────────────────────────────
NWALKERS = 32
NSTEPS   = 3000
NBURN    = 500
NTHIN    = 10

# ── Data ─────────────────────────────────────────────────────
z_ohd, H_ohd, sH_ohd = get_ohd_data()
z_pp, mu_pp, cov_pp, se_pp, _ = get_pantheon_data("PP")
z_sh, mu_sh, cov_sh, se_sh, _ = get_pantheon_data("PP_SH0ES")

print(f"OHD:{len(z_ohd)}  PP:{len(z_pp)}  "
      f"PP+SH:{len(z_sh)}  DESI:12")

# ── Fast distance modulus via grid interpolation ──────────────
C_LIGHT = 2.998e5
_ZG = np.linspace(0.0, 2.6, 3000)

def mu_fast(z_sn, H0, l, m):
    Hz  = np.maximum(H_parametric(_ZG, H0, l, m), 1.0)
    dz  = np.diff(_ZG)
    iv  = C_LIGHT / Hz
    cum = np.concatenate([[0.0],
          np.cumsum(0.5*(iv[:-1]+iv[1:])*dz)])
    dc  = np.interp(z_sn, _ZG, cum)
    dL  = np.maximum((1.0+z_sn)*dc, 1e-5)
    return 5.0*np.log10(dL) + 25.0

def chi2_pp_fast(H0, l, m, z_sn, mu_obs, cov_inv):
    d = mu_obs - mu_fast(z_sn, H0, l, m)
    return float(d @ cov_inv @ d)

# ── Dataset configs ───────────────────────────────────────────
DATASETS = {
    "OHD": dict(
        label="OHD", use_pp=False, use_desi=False,
        z_sn=None, mu_obs=None, cov_inv=None,
        ndim=3, bounds=[(60,80),(0,3),(0,10)],
        N=len(z_ohd)),
    "OHD_PP": dict(
        label="OHD+PP", use_pp=True, use_desi=False,
        z_sn=z_pp, mu_obs=mu_pp, cov_inv=cov_pp,
        ndim=3, bounds=[(60,80),(0,3),(0,10)],
        N=len(z_ohd)+len(z_pp)),
    "OHD_PP_SH0ES": dict(
        label="OHD+PP+SH0ES", use_pp=True, use_desi=False,
        z_sn=z_sh, mu_obs=mu_sh, cov_inv=cov_sh,
        ndim=3, bounds=[(60,80),(0,3),(0,10)],
        N=len(z_ohd)+len(z_sh)),
    "OHD_PP_SH0ES_DESI": dict(
        label="OHD+PP+SH0ES+DESI", use_pp=True, use_desi=True,
        z_sn=z_sh, mu_obs=mu_sh, cov_inv=cov_sh,
        ndim=4, bounds=[(60,80),(0,3),(0,10),(143,151)],
        N=len(z_ohd)+len(z_sh)+12),
}

# ── Log posterior factory ─────────────────────────────────────
def make_lp(cfg):
    bounds   = cfg["bounds"]
    use_pp   = cfg["use_pp"]
    use_desi = cfg["use_desi"]
    z_sn     = cfg["z_sn"]
    mu_obs   = cfg["mu_obs"]
    cov_inv  = cfg["cov_inv"]

    def lp(params):
        for p,(lo,hi) in zip(params, bounds):
            if not (lo < p < hi):
                return -np.inf
        H0 = params[0]; l = params[1]; m = params[2]
        rd = params[3] if len(params)==4 else RD_PLANCK

        c2 = chi2_ohd((H0,l,m), z_ohd, H_ohd, sH_ohd)
        if use_pp:
            c2 += chi2_pp_fast(H0,l,m,z_sn,mu_obs,cov_inv)
        if use_desi:
            c2 += chi2_desi(
                lambda z: float(H_parametric(z,H0,l,m)),
                rd=rd)
        return -0.5*c2
    return lp

# ── LCDM reference chi2 ───────────────────────────────────────
def lcdm_chi2(H0, cfg):
    c2 = chi2_ohd((H0,0.3,0), z_ohd, H_ohd, sH_ohd)
    if cfg["use_pp"]:
        Hz  = np.maximum(H_LCDM(_ZG, H0, 0.315), 1.0)
        dz  = np.diff(_ZG)
        iv  = C_LIGHT/Hz
        cum = np.concatenate([[0.0],
              np.cumsum(0.5*(iv[:-1]+iv[1:])*dz)])
        dc  = np.interp(cfg["z_sn"], _ZG, cum)
        dL  = np.maximum((1+cfg["z_sn"])*dc, 1e-5)
        d   = cfg["mu_obs"] - (5*np.log10(dL)+25)
        c2 += float(d @ cfg["cov_inv"] @ d)
    if cfg["use_desi"]:
        c2 += chi2_desi(lambda z: float(H_LCDM(z,H0,0.315)))
    return c2

# ════════════════════════════════════════════════════════════
summary = {}

for combo, cfg in DATASETS.items():
    print(f"\n{'═'*55}")
    print(f"  {cfg['label']}")
    print(f"{'═'*55}")

    os.makedirs(f"results/{combo}", exist_ok=True)
    os.makedirs(f"figures/{combo}", exist_ok=True)

    lp   = make_lp(cfg)
    ndim = cfg["ndim"]

    # MAP
    print("  Finding MAP ...")
    de = differential_evolution(
        lambda p: -lp(p), bounds=cfg["bounds"],
        seed=42, maxiter=400, tol=1e-8,
        workers=1, polish=True)
    p0 = de.x
    print("  MAP: " + "  ".join(f"{v:.4f}" for v in p0))

    # MCMC
    print(f"  MCMC ({NWALKERS}w x {NSTEPS}s) ...")
    t0_ = time.time()
    p0s = p0 + 1e-3*np.random.randn(NWALKERS, ndim)

    try:
        from multiprocessing import Pool, cpu_count
        ncpu = min(cpu_count(), 8)
        with Pool(ncpu) as pool:
            sampler = emcee.EnsembleSampler(
                NWALKERS, ndim, lp, pool=pool)
            sampler.run_mcmc(p0s, NSTEPS, progress=True)
    except Exception:
        sampler = emcee.EnsembleSampler(NWALKERS, ndim, lp)
        sampler.run_mcmc(p0s, NSTEPS, progress=True)

    print(f"  Done: {(time.time()-t0_)/60:.1f} min")

    flat = sampler.get_chain(
        discard=NBURN, thin=NTHIN, flat=True)
    p16,p50,p84 = [np.percentile(flat,q,axis=0)
                   for q in [16,50,84]]

    H0_bf,l_bf,m_bf = p50[0],p50[1],p50[2]
    rd_bf = p50[3] if ndim==4 else RD_PLANCK

    pnames = ['H0','l','m','rd'][:ndim]
    for i,pn in enumerate(pnames):
        print(f"    {pn} = {p50[i]:.4f} "
              f"+{p84[i]-p50[i]:.4f} "
              f"-{p50[i]-p16[i]:.4f}")

    # Chi2, AIC, BIC
    c2 = chi2_ohd((H0_bf,l_bf,m_bf),z_ohd,H_ohd,sH_ohd)
    c2_ohd_val = float(c2)
    if cfg["use_pp"]:
        c2 += chi2_pp_fast(H0_bf,l_bf,m_bf,
                            cfg["z_sn"],cfg["mu_obs"],
                            cfg["cov_inv"])
    if cfg["use_desi"]:
        c2 += chi2_desi(
            lambda z: float(H_parametric(z,H0_bf,l_bf,m_bf)),
            rd=rd_bf)

    aic = AIC(c2,ndim); bic = BIC(c2,ndim,cfg["N"])
    c2_lc = lcdm_chi2(H0_bf, cfg)
    aic_lc = AIC(c2_lc,1); bic_lc = BIC(c2_lc,1,cfg["N"])
    t0_gyr = age_Gyr(H0_bf,l_bf,m_bf)
    tension = abs(H0_bf-73.27)/np.sqrt(
        (p84[0]-p50[0])**2+1.04**2)

    print(f"  chi2={c2:.2f}  "
          f"ΔAIC={aic-aic_lc:+.2f}  "
          f"ΔBIC={bic-bic_lc:+.2f}  "
          f"Age={t0_gyr:.3f}Gyr  "
          f"T={tension:.2f}σ")

    # Save
    np.save(f"results/{combo}/mcmc_samples.npy", flat)
    np.save(f"results/{combo}/best_fit.npy",     p50)
    np.save(f"results/{combo}/percentiles.npy",
            np.vstack([p16,p50,p84]))
    np.save(f"results/{combo}/chi2_min.npy",
            np.array([c2, c2_ohd_val]))

    summary[combo] = dict(
        label=cfg["label"],
        H0=p50[0], H0_lo=p50[0]-p16[0], H0_hi=p84[0]-p50[0],
        l=p50[1],  l_lo=p50[1]-p16[1],  l_hi=p84[1]-p50[1],
        m=p50[2],  m_lo=p50[2]-p16[2],  m_hi=p84[2]-p50[2],
        chi2=c2, chi2_ohd=c2_ohd_val,
        AIC=aic, BIC=bic,
        dAIC=aic-aic_lc, dBIC=bic-bic_lc,
        age=t0_gyr, tension=tension)

    # Corner plot
    fig = corner.corner(flat,
        labels=[r'$H_0$',r'$l$',r'$m$',r'$r_d$'][:ndim],
        quantiles=[0.16,0.50,0.84],
        show_titles=True, title_fmt='.3f',
        smooth=1.0, color='steelblue')
    fig.suptitle(cfg["label"], fontsize=12, y=1.01)
    fig.savefig(f"figures/{combo}/corner_{combo}.pdf",
                bbox_inches='tight', dpi=150)
    plt.close()

    # H(z) plot
    z_pl = np.linspace(0.01,2.5,500)
    idx  = np.random.choice(len(flat),
                             size=min(200,len(flat)),
                             replace=False)
    H_band = np.array([H_parametric(z_pl,*s[:3])
                        for s in flat[idx]])
    fig2,ax = plt.subplots(figsize=(8,5))
    ax.errorbar(z_ohd,H_ohd,yerr=sH_ohd,
                fmt='o',ms=5,color='royalblue',
                capsize=3,label='OHD',zorder=6)
    ax.plot(z_pl,H_parametric(z_pl,H0_bf,l_bf,m_bf),
            'r-',lw=2,
            label=f'Model $H_0={H0_bf:.1f}$ '
                  f'$l={l_bf:.3f}$ $m={m_bf:.3f}$')
    ax.fill_between(z_pl,
        np.percentile(H_band,16,axis=0),
        np.percentile(H_band,84,axis=0),
        color='red',alpha=0.20,label='1σ')
    ax.plot(z_pl,H_LCDM(z_pl,H0_bf),'k--',
            lw=1.5,label=r'$\Lambda$CDM')
    ax.set_xlabel('$z$',fontsize=12)
    ax.set_ylabel(r'$H(z)$ [km/s/Mpc]',fontsize=12)
    ax.set_title(cfg["label"],fontsize=13)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"figures/{combo}/Hz_mcmc_{combo}.pdf",
                 dpi=150)
    plt.close()

np.save("results/mcmc_summary.npy", summary)

# ── Final summary table ───────────────────────────────────────
print("\n"+"═"*95)
print(f"{'Dataset':<24} {'H0':>16} {'l':>12} "
      f"{'m':>10} {'chi2':>8} "
      f"{'ΔAIC':>7} {'ΔBIC':>7} "
      f"{'Age(Gyr)':>10} {'T_SH0ES':>9}")
print("─"*95)
for k,v in summary.items():
    print(f"{v['label']:<24} "
          f"{v['H0']:.2f}+{v['H0_hi']:.2f}-{v['H0_lo']:.2f}  "
          f"{v['l']:.3f}+{v['l_hi']:.3f}-{v['l_lo']:.3f}  "
          f"{v['m']:.3f}+{v['m_hi']:.3f}-{v['m_lo']:.3f}  "
          f"{v['chi2']:>8.2f}  "
          f"{v['dAIC']:>+7.2f}  "
          f"{v['dBIC']:>+7.2f}  "
          f"{v['age']:>10.3f}  "
          f"{v['tension']:>7.2f}σ")
print("═"*95)
print("\n[01_mcmc.py] Complete.")
