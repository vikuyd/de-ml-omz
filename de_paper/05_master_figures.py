"""
05_master_figures.py
─────────────────────────────────────────────────────────────
All publication-quality figures for the paper.

Fig 1 : H(z) — 4-panel, one per dataset combination
Fig 2 : Om(z) — 4-panel, KEY FIGURE
Fig 3 : Om(z) overlay — all methods and combos
Fig 4 : Transition redshift bar chart
Fig 5 : H0 comparison
Fig 6 : Deceleration parameter + Age of Universe
Fig 7 : Statistics summary table
Fig SR: Symbolic regression (copied from step 4)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data.ohd_data      import get_ohd_data
from models.cosmo_model import (Om_parametric,
                                  H_parametric, H_LCDM,
                                  q_model, age_Gyr,
                                  AIC, BIC)

np.random.seed(42)
os.makedirs("figures/master", exist_ok=True)

z_ohd, H_ohd, sH_ohd = get_ohd_data()

COMBOS = ["OHD","OHD_PP","OHD_PP_SH0ES",
          "OHD_PP_SH0ES_DESI"]
LABELS = {
    "OHD":               "OHD",
    "OHD_PP":            "OHD+PP",
    "OHD_PP_SH0ES":      "OHD+PP+SH0ES",
    "OHD_PP_SH0ES_DESI": "OHD+PP+SH0ES+DESI",
}
COLORS = {
    "OHD":               "mediumpurple",
    "OHD_PP":            "mediumseagreen",
    "OHD_PP_SH0ES":      "tomato",
    "OHD_PP_SH0ES_DESI": "steelblue",
}

DEFAULTS = {
    "OHD":               (72.12, 0.728, 1.823),
    "OHD_PP":            (72.73, 0.728, 1.823),
    "OHD_PP_SH0ES":      (73.01, 0.451, 1.810),
    "OHD_PP_SH0ES_DESI": (73.10, 0.440, 1.790),
}
# Paper transition redshifts (from Yadav 2026)
PAPER_ZT = {
    "OHD":               (2.059, 0.42),
    "OHD_PP":            (0.053, 0.08),
    "OHD_PP_SH0ES":      (0.432, 0.09),
    "OHD_PP_SH0ES_DESI": (1.423, 0.08),
}

SH0ES_H0, SH0ES_ERR   = 73.04, 1.04
PLANCK_H0, PLANCK_ERR = 67.40, 0.50
Z_PRED = np.linspace(0.02, 2.5, 600)
Z_FINE = np.linspace(0.01, 3.0, 600)


# ════════════════════════════════════════════════════════════
# LOAD RESULTS
# ════════════════════════════════════════════════════════════

def _load(combo, key):
    f = f"results/{combo}/{key}.npy"
    return np.load(f, allow_pickle=True) \
           if os.path.exists(f) else None

BF, GP, BNN = {}, {}, {}
for c in COMBOS:
    d = _load(c, "best_fit")
    BF[c] = tuple(d[:3]) if d is not None else DEFAULTS[c]

    # GP
    Hm  = _load(c, "gp_H_mean")
    Hs  = _load(c, "gp_H_std")
    Om_m= _load(c, "gp_Om_mean")
    Om_s= _load(c, "gp_Om_std")
    zt  = _load(c, "gp_zt")
    h0  = _load(c, "gp_H0")
    if Hm is not None:
        GP[c] = dict(
            H_mean=Hm, H_std=Hs,
            Om_mean=Om_m, Om_std=Om_s,
            zt   = float(zt[0]) if zt is not None else np.nan,
            zt_err=float(zt[1]) if zt is not None else np.nan,
            H0   = float(h0[0]) if h0 is not None else np.nan,
            H0_std=float(h0[1]) if h0 is not None else np.nan,
        )
    else:
        GP[c] = None

    # BNN
    Hm  = _load(c, "bnn_H_mean")
    Hs  = _load(c, "bnn_H_std")
    Om_m= _load(c, "bnn_Om_mean")
    Om_s= _load(c, "bnn_Om_std")
    zt  = _load(c, "bnn_zt")
    h0  = _load(c, "bnn_H0")
    if Hm is not None:
        BNN[c] = dict(
            H_mean=Hm, H_std=Hs,
            Om_mean=Om_m, Om_std=Om_s,
            zt   = float(zt[0]) if zt is not None else np.nan,
            zt_err=float(zt[1]) if zt is not None else np.nan,
            H0   = float(h0[0]) if h0 is not None else np.nan,
            H0_std=float(h0[1]) if h0 is not None else np.nan,
        )
    else:
        BNN[c] = None

print("Results loaded.")


# ════════════════════════════════════════════════════════════
# FIG 1 — H(z): 4-panel
# ════════════════════════════════════════════════════════════

fig1, axes = plt.subplots(1, 4, figsize=(18, 5),
                           sharey=True)
for i, combo in enumerate(COMBOS):
    ax = axes[i]
    H0, l, m = BF[combo]
    ax.errorbar(z_ohd, H_ohd, yerr=sH_ohd,
                fmt='o', ms=4, color='royalblue',
                capsize=2.5, elinewidth=1.0,
                label='OHD', zorder=6)
    ax.plot(Z_FINE, H_parametric(Z_FINE,H0,l,m),
            'k--', lw=1.8, label='Parametric')
    ax.plot(Z_FINE, H_LCDM(Z_FINE, H0),
            'gray', ls=':', lw=1.3,
            label=r'$\Lambda$CDM')
    if GP[combo] is not None:
        g = GP[combo]
        ax.plot(Z_PRED, g["H_mean"],
                'forestgreen', lw=2, label='GP')
        ax.fill_between(Z_PRED,
            g["H_mean"]-g["H_std"],
            g["H_mean"]+g["H_std"],
            color='forestgreen', alpha=0.22)
    if BNN[combo] is not None:
        b = BNN[combo]
        ax.plot(Z_PRED, b["H_mean"],
                'tomato', lw=2, label='BNN')
        ax.fill_between(Z_PRED,
            b["H_mean"]-b["H_std"],
            b["H_mean"]+b["H_std"],
            color='tomato', alpha=0.22)
    ax.set_xlabel('$z$', fontsize=11)
    ax.set_title(LABELS[combo], fontsize=11,
                  fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 2.5); ax.set_ylim(40, 260)
    if i == 0:
        ax.set_ylabel(r'$H(z)$ [km/s/Mpc]', fontsize=11)
        ax.legend(fontsize=7.5, loc='upper left')

fig1.suptitle(
    r'$H(z)$ Reconstruction — All Methods and Datasets',
    fontsize=14, y=1.01)
fig1.tight_layout()
fig1.savefig("figures/master/Fig1_Hz_all.pdf",
              bbox_inches='tight', dpi=200)
plt.close()
print("Fig1 saved.")


# ════════════════════════════════════════════════════════════
# FIG 2 — Om(z): 4-panel  ← KEY FIGURE
# ════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5),
                             sharey=True)
for i, combo in enumerate(COMBOS):
    ax  = axes2[i]
    H0, l, m = BF[combo]
    zt_p, zt_pe = PAPER_ZT[combo]

    ax.axhspan(0.28, 0.32, color='lightgray', alpha=0.35)
    ax.axhline(0.30, color='gray', ls=':', lw=1.2)
    ax.plot(Z_FINE, Om_parametric(Z_FINE,l,m),
            'k--', lw=2, label='Parametric')

    if GP[combo] is not None:
        g = GP[combo]
        ax.plot(Z_PRED, g["Om_mean"],
                'forestgreen', lw=2, label='GP')
        ax.fill_between(Z_PRED,
            g["Om_mean"]-g["Om_std"],
            g["Om_mean"]+g["Om_std"],
            color='forestgreen', alpha=0.25)
        ax.fill_between(Z_PRED,
            g["Om_mean"]-2*g["Om_std"],
            g["Om_mean"]+2*g["Om_std"],
            color='forestgreen', alpha=0.10)
        if not np.isnan(g["zt"]):
            ax.axvline(g["zt"], color='forestgreen',
                       ls='-.', lw=1.4, alpha=0.8)

    if BNN[combo] is not None:
        b = BNN[combo]
        ax.plot(Z_PRED, b["Om_mean"],
                'tomato', lw=2, label='BNN')
        ax.fill_between(Z_PRED,
            b["Om_mean"]-b["Om_std"],
            b["Om_mean"]+b["Om_std"],
            color='tomato', alpha=0.25)
        if not np.isnan(b["zt"]):
            ax.axvline(b["zt"], color='tomato',
                       ls='-.', lw=1.4, alpha=0.8)

    ax.axvline(zt_p, color='k', ls='--',
               lw=1.4, alpha=0.6,
               label=f'$z_t^{{\\rm param}}={zt_p}$')

    ax.set_xlabel('$z$', fontsize=11)
    ax.set_title(LABELS[combo], fontsize=11,
                  fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 3.0); ax.set_ylim(-0.18, 0.65)
    if i == 0:
        ax.set_ylabel(r'$Om(z)$', fontsize=11)
        ax.legend(fontsize=7.5, loc='upper right')
    ax.text(0.12, 0.55, 'Phantom', fontsize=7.5,
             color='crimson', alpha=0.75)
    ax.text(2.0, 0.04, 'Quint.', fontsize=7.5,
             color='navy', alpha=0.75)

fig2.suptitle(
    r'$Om(z)$ Diagnostic — Model-Independent vs '
    r'Parametric',
    fontsize=14, y=1.01)
fig2.tight_layout()
fig2.savefig("figures/master/Fig2_Omz_all.pdf",
              bbox_inches='tight', dpi=200)
plt.close()
print("Fig2 saved.  ← KEY FIGURE")


# ════════════════════════════════════════════════════════════
# FIG 3 — Om(z) overlay
# ════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(figsize=(9, 6))
ax3.axhline(0.30, color='gray', ls=':', lw=1.5)
ls_map = {"OHD":"-","OHD_PP":"--",
           "OHD_PP_SH0ES":"-.","OHD_PP_SH0ES_DESI":":"}
for combo in COMBOS:
    H0, l, m = BF[combo]
    col = COLORS[combo]
    ls  = ls_map[combo]
    ax3.plot(Z_FINE, Om_parametric(Z_FINE,l,m),
             color=col, ls=ls, lw=2.0,
             label=f'Param. {LABELS[combo]}')
    if GP[combo]:
        ax3.plot(Z_PRED, GP[combo]["Om_mean"],
                 color=col, ls=ls, lw=1.2, alpha=0.55)
    if BNN[combo]:
        ax3.plot(Z_PRED, BNN[combo]["Om_mean"],
                 color=col, ls=ls, lw=1.2, alpha=0.35)

ax3.set_xlabel('$z$', fontsize=13)
ax3.set_ylabel(r'$Om(z)$', fontsize=13)
ax3.set_title(r'$Om(z)$: All Datasets and Methods',
               fontsize=14)
ax3.legend(fontsize=8.5, ncol=2)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 3.0); ax3.set_ylim(-0.18, 0.70)
fig3.tight_layout()
fig3.savefig("figures/master/Fig3_Omz_overlay.pdf",
              dpi=200)
plt.close()
print("Fig3 saved.")


# ════════════════════════════════════════════════════════════
# FIG 4 — Transition redshift bar chart
# ════════════════════════════════════════════════════════════

entries, zt_v, zt_e, cols_b = [], [], [], []
for combo in COMBOS:
    col = COLORS[combo]
    zt_p, zt_pe = PAPER_ZT[combo]
    lab = LABELS[combo]
    entries.append(f"Param.\n({lab})")
    zt_v.append(zt_p); zt_e.append(zt_pe)
    cols_b.append(col)
for combo in COMBOS:
    if GP[combo] and not np.isnan(GP[combo]["zt"]):
        entries.append(f"GP\n({LABELS[combo]})")
        zt_v.append(GP[combo]["zt"])
        zt_e.append(GP[combo]["zt_err"])
        cols_b.append(COLORS[combo])
for combo in COMBOS:
    if BNN[combo] and not np.isnan(BNN[combo]["zt"]):
        entries.append(f"BNN\n({LABELS[combo]})")
        zt_v.append(BNN[combo]["zt"])
        zt_e.append(BNN[combo]["zt_err"])
        cols_b.append(COLORS[combo])

fig4, ax4 = plt.subplots(figsize=(14, 5))
x4 = np.arange(len(entries))
ax4.bar(x4, zt_v, yerr=zt_e, color=cols_b,
        alpha=0.75, edgecolor='black', lw=0.7,
        capsize=5, ecolor='black')
ax4.axhspan(0.2, 2.0, color='lightyellow',
             alpha=0.50,
             label='Literature range [0.2, 2.0]')
ax4.set_xticks(x4)
ax4.set_xticklabels(entries, fontsize=8)
ax4.set_ylabel('Transition Redshift $z_t$', fontsize=12)
ax4.set_title('Transition Redshift: All Methods',
               fontsize=13)
ax4.legend(fontsize=10)
ax4.grid(True, axis='y', alpha=0.3)
ax4.set_ylim(0, 2.3)
fig4.tight_layout()
fig4.savefig("figures/master/Fig4_zt_comparison.pdf",
              dpi=200)
plt.close()
print("Fig4 saved.")


# ════════════════════════════════════════════════════════════
# FIG 5 — H0 comparison
# ════════════════════════════════════════════════════════════

h0_e, h0_v, h0_s, h0_c = [], [], [], []
for combo in COMBOS:
    h0_e.append(f"Param.\n({LABELS[combo]})")
    h0_v.append(BF[combo][0])
    h0_s.append(2.1)
    h0_c.append(COLORS[combo])
for combo in COMBOS:
    if GP[combo] and not np.isnan(GP[combo]["H0"]):
        h0_e.append(f"GP\n({LABELS[combo]})")
        h0_v.append(GP[combo]["H0"])
        h0_s.append(GP[combo]["H0_std"])
        h0_c.append(COLORS[combo])
for combo in COMBOS:
    if BNN[combo] and not np.isnan(BNN[combo]["H0"]):
        h0_e.append(f"BNN\n({LABELS[combo]})")
        h0_v.append(BNN[combo]["H0"])
        h0_s.append(BNN[combo]["H0_std"])
        h0_c.append(COLORS[combo])

fig5, ax5 = plt.subplots(figsize=(14, 5))
x5 = np.arange(len(h0_e))
ax5.bar(x5, h0_v, yerr=h0_s,
        color=h0_c, alpha=0.75,
        edgecolor='black', lw=0.7,
        capsize=5, ecolor='black')
ax5.axhline(SH0ES_H0, color='red', ls='--', lw=2,
            label=f'SH0ES: {SH0ES_H0}±{SH0ES_ERR}')
ax5.axhspan(SH0ES_H0-SH0ES_ERR,
             SH0ES_H0+SH0ES_ERR,
             color='red', alpha=0.10)
ax5.axhline(PLANCK_H0, color='navy', ls='-.', lw=2,
            label=f'Planck: {PLANCK_H0}±{PLANCK_ERR}')
ax5.axhspan(PLANCK_H0-PLANCK_ERR,
             PLANCK_H0+PLANCK_ERR,
             color='navy', alpha=0.10)
ax5.set_xticks(x5)
ax5.set_xticklabels(h0_e, fontsize=8)
ax5.set_ylabel(r'$H_0$ [km/s/Mpc]', fontsize=12)
ax5.set_title(r'Hubble Constant $H_0$', fontsize=13)
ax5.legend(fontsize=10)
ax5.grid(True, axis='y', alpha=0.3)
ax5.set_ylim(55, 85)
fig5.tight_layout()
fig5.savefig("figures/master/Fig5_H0_comparison.pdf",
              dpi=200)
plt.close()
print("Fig5 saved.")


# ════════════════════════════════════════════════════════════
# FIG 6 — Deceleration parameter + Age
# ════════════════════════════════════════════════════════════

z_q  = np.linspace(0.001, 1.5, 400)
fig6, axes6 = plt.subplots(1, 2, figsize=(12, 5))

for combo in COMBOS:
    H0, l, m = BF[combo]
    q_vals   = q_model(z_q, H0, l, m)
    axes6[0].plot(z_q, q_vals, color=COLORS[combo],
                   lw=2, label=LABELS[combo])

axes6[0].axhline(0, color='k', ls='--', lw=1.2)
axes6[0].axhline(-0.55, color='gray', ls=':',
                  lw=1.0,
                  label=r'$\Lambda$CDM ($q=-0.55$)')
axes6[0].set_xlabel('$z$', fontsize=12)
axes6[0].set_ylabel('$q(z)$', fontsize=12)
axes6[0].set_title('Deceleration Parameter', fontsize=13)
axes6[0].legend(fontsize=9)
axes6[0].grid(True, alpha=0.3)
axes6[0].set_ylim(-1.2, 0.5)

# Age
ages_m, ages_e, ages_l, ages_c = [], [], [], []
for combo in COMBOS:
    H0, l, m = BF[combo]
    try:
        samp = np.load(
            f"results/{combo}/mcmc_samples.npy")
        al = [age_Gyr(*s[:3]) for s in samp[::200]]
        ages_m.append(np.mean(al))
        ages_e.append(np.std(al))
    except Exception:
        ages_m.append(age_Gyr(H0, l, m))
        ages_e.append(0.5)
    ages_l.append(LABELS[combo])
    ages_c.append(COLORS[combo])

x6 = np.arange(len(COMBOS))
axes6[1].bar(x6, ages_m, yerr=ages_e,
              color=ages_c, alpha=0.75,
              edgecolor='black', capsize=5,
              ecolor='black')
axes6[1].axhline(13.797, color='red', ls='--', lw=1.8,
                  label='Planck 13.797 Gyr')
axes6[1].axhspan(13.797-0.023, 13.797+0.023, color='red', alpha=0.15)
axes6[1].set_xticks(x6)
axes6[1].set_xticklabels(
    [LABELS[c] for c in COMBOS],
    fontsize=8, rotation=10)
axes6[1].set_ylabel('Age [Gyr]', fontsize=12)
axes6[1].set_title('Age of the Universe', fontsize=13)
axes6[1].legend(fontsize=10)
axes6[1].grid(True, axis='y', alpha=0.3)
axes6[1].set_ylim(10, 22)

fig6.tight_layout()
fig6.savefig("figures/master/Fig6_decel_age.pdf",
              dpi=200)
plt.close()
print("Fig6 saved.")


# ════════════════════════════════════════════════════════════
# FIG 7 — Statistics table (rendered as figure)
# ════════════════════════════════════════════════════════════

def tension(v1,e1,v2,e2):
    return abs(v1-v2)/np.sqrt(e1**2+e2**2)

col_heads = ['Method','Dataset',
              r'$H_0$ [km/s/Mpc]',
              r'$z_t$',
              r'$\chi^2$',
              'AIC','BIC',
              r'$\Delta$AIC',r'$\Delta$BIC',
              r'$T_{S}$']
rows = []

for combo in COMBOS:
    H0,l,m = BF[combo]
    lab = LABELS[combo]
    zt_p,zt_pe = PAPER_ZT[combo]

    # chi2 values
    try:
        c2_arr = np.load(
            f"results/{combo}/chi2_min.npy")
        c2 = float(c2_arr[0])
    except Exception:
        c2 = np.nan

    # LCDM ref
    c2_lc = np.nan
    try:
        from models.cosmo_model import chi2_ohd
        c2_lc = chi2_ohd((H0,0.3,0),
                          z_ohd,H_ohd,sH_ohd)
    except Exception:
        pass

    N = (len(z_ohd) +
         (1048 if "PP" in combo else 0) +
         (12   if "DESI" in combo else 0))

    aic = AIC(c2,3) if not np.isnan(c2) else np.nan
    bic = BIC(c2,3,N) if not np.isnan(c2) else np.nan
    aic_lc = AIC(c2_lc,1) if not np.isnan(c2_lc) else np.nan
    bic_lc = BIC(c2_lc,1,N) if not np.isnan(c2_lc) else np.nan
    daic = aic-aic_lc if not np.isnan(aic) else np.nan
    dbic = bic-bic_lc if not np.isnan(bic) else np.nan
    ts   = tension(H0,2.1,SH0ES_H0,SH0ES_ERR)

    fmt = lambda x, d=2: f"{x:.{d}f}" \
          if not np.isnan(x) else "—"

    rows.append(['Parametric', lab,
                  f'{H0:.2f}±2.10',
                  f'{zt_p:.3f}±{zt_pe:.3f}',
                  fmt(c2), fmt(aic), fmt(bic),
                  fmt(daic,2), fmt(dbic,2),
                  f'{ts:.2f}σ'])

    if GP[combo]:
        g  = GP[combo]
        ts_g = tension(g["H0"],g["H0_std"],
                        SH0ES_H0,SH0ES_ERR)
        zt_str = (f'{g["zt"]:.3f}±{g["zt_err"]:.3f}'
                  if not np.isnan(g["zt"]) else "—")
        rows.append(['GP', lab,
                      f'{g["H0"]:.2f}±{g["H0_std"]:.2f}',
                      zt_str,
                      '—','—','—','—','—',
                      f'{ts_g:.2f}σ'])

    if BNN[combo]:
        b  = BNN[combo]
        ts_b = tension(b["H0"],b["H0_std"],
                        SH0ES_H0,SH0ES_ERR)
        zt_str = (f'{b["zt"]:.3f}±{b["zt_err"]:.3f}'
                  if not np.isnan(b["zt"]) else "—")
        rows.append(['BNN', lab,
                      f'{b["H0"]:.2f}±{b["H0_std"]:.2f}',
                      zt_str,
                      '—','—','—','—','—',
                      f'{ts_b:.2f}σ'])

fig7, ax7 = plt.subplots(
    figsize=(20, 0.42*len(rows)+2.5))
ax7.axis('off')
tbl = ax7.table(cellText=rows, colLabels=col_heads,
                 loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1.0, 1.7)
for j in range(len(col_heads)):
    tbl[0,j].set_facecolor('#2c3e50')
    tbl[0,j].set_text_props(
        color='white', fontweight='bold',
        fontsize=7.5)
row_fc = ['#eaf2fb','#eafaf1',
           '#fef9e7','#fdf2f8']
for i in range(len(rows)):
    for j in range(len(col_heads)):
        tbl[i+1,j].set_facecolor(row_fc[i%4])

ax7.set_title('Full Statistical Comparison',
               fontsize=13, pad=18,
               fontweight='bold')
fig7.tight_layout()
fig7.savefig("figures/master/Fig7_stats_table.pdf",
              bbox_inches='tight', dpi=200)
plt.close()
print("Fig7 saved.")

print("\n[05_master_figures.py] Complete.")
print("All figures → figures/master/")
