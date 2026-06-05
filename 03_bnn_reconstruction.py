"""
03_bnn_reconstruction.py
─────────────────────────────────────────────────────────────
Bayesian Neural Network reconstruction of H(z) and Om(z).

Architecture : 1 → 128 → 128 → 64 → 32 → 1
Activations  : Tanh hidden, Softplus output (H > 0)
Training     : Adam + cosine annealing + early stopping
Uncertainty  : MC Dropout (2000 forward passes)

Joint loss for multi-dataset combos:
  L = chi2_OHD + lambda_pp * chi2_PP + lambda_desi * chi2_DESI

H0 fix: we add a Gaussian prior term on H0 predicted by the
network (penalises solutions far from SH0ES range [70,76])
to prevent the known low-z extrapolation bias.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from data.ohd_data      import get_ohd_data
from data.pantheon_data import get_pantheon_data
from data.desi_data     import chi2_desi
from models.cosmo_model import (Om_parametric, Om_from_H,
                                  H_parametric, H_LCDM,
                                  find_transition_z)

np.random.seed(42)
torch.manual_seed(42)

# ════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════

z_ohd, H_ohd, sH_ohd            = get_ohd_data()
z_pp,  mu_pp,  cov_pp,  se_pp,  _ = get_pantheon_data("PP")
z_sh,  mu_sh,  cov_sh,  se_sh,  _ = get_pantheon_data("PP_SH0ES")

H0_NORM = 100.0
Z_PRED  = np.linspace(0.02, 2.5, 600)
N_MC    = 2000

COMBOS = ["OHD", "OHD_PP", "OHD_PP_SH0ES",
          "OHD_PP_SH0ES_DESI"]
LABELS = {
    "OHD":               "OHD",
    "OHD_PP":            "OHD+PP",
    "OHD_PP_SH0ES":      "OHD+PP+SH0ES",
    "OHD_PP_SH0ES_DESI": "OHD+PP+SH0ES+DESI",
}

DEFAULTS = {
    "OHD":               (72.12, 0.728, 1.823),
    "OHD_PP":            (72.73, 0.728, 1.823),
    "OHD_PP_SH0ES":      (73.01, 0.451, 1.810),
    "OHD_PP_SH0ES_DESI": (73.10, 0.440, 1.790),
}


# ════════════════════════════════════════════════════════════
# BNN ARCHITECTURE
# ════════════════════════════════════════════════════════════

class CosmoBNN(nn.Module):
    def __init__(self, hidden=(128,128,64,32),
                 dropout=0.05):
        super().__init__()
        layers, d_in = [], 1
        for d_h in hidden:
            layers += [nn.Linear(d_in, d_h),
                       nn.Tanh(),
                       nn.Dropout(p=dropout)]
            d_in = d_h
        layers += [nn.Linear(d_in, 1), nn.Softplus()]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


# ════════════════════════════════════════════════════════════
# LOSSES
# ════════════════════════════════════════════════════════════

def chi2_ohd_t(H_pred, H_obs_t, H_err_t):
    return torch.sum(((H_pred - H_obs_t) / H_err_t)**2)


def pp_chi2_np(H_pred_np, z_sn, mu_obs, cov_inv):
    """Pantheon chi2 evaluated in numpy (no grad)."""
    from scipy.integrate import cumulative_trapezoid
    H_pred_phys = H_pred_np * H0_NORM
    H0_est = max(float(H_pred_phys[0]), 30.0)
    z_grid = np.linspace(0.001, max(z_sn)+0.01, 800)
    H_grid = np.interp(z_grid, z_ohd, H_pred_phys)
    H_grid = np.maximum(H_grid, 1.0)
    integ  = cumulative_trapezoid(
        1.0/H_grid, z_grid, initial=0.0)
    dL     = (1+z_grid)*2.998e5*integ
    dL     = np.maximum(dL, 1e-5)
    mu_th  = np.interp(z_sn, z_grid,
                        5*np.log10(dL)+25)
    delta  = mu_obs - mu_th
    return min(float(delta @ cov_inv @ delta), 1e8)


def desi_chi2_np(H_pred_np):
    """DESI chi2 evaluated in numpy (no grad)."""
    H_pred_phys = H_pred_np * H0_NORM
    def Hz_func(z):
        return float(np.interp(z, z_ohd, H_pred_phys))
    try:
        return min(chi2_desi(Hz_func), 1e6)
    except Exception:
        return 0.0


def h0_prior_loss(H_pred_t, H0_center=73.0, H0_sigma=5.0):
    """
    Gaussian prior on H0 = H(z_min).
    Prevents BNN from collapsing to low H0 on sparse data.
    """
    H0_pred = H_pred_t[0, 0] * H0_NORM
    return ((H0_pred - H0_center) / H0_sigma)**2


# ════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════

def train(model, z_t, H_t, He_t,
          use_pp, z_sn, mu_obs, cov_inv,
          use_desi,
          n_epochs=8000, lr=5e-4, wd=1e-4,
          patience=1200,
          lam_pp=0.005, lam_desi=0.01,
          lam_h0=0.5,
          verbose=True):

    opt   = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=1e-6)

    losses = []
    best   = np.inf
    best_w = None
    no_imp = 0

    for ep in range(1, n_epochs+1):
        model.train()
        opt.zero_grad()

        H_pred = model(z_t)
        loss   = chi2_ohd_t(H_pred, H_t, He_t)

        # H0 prior
        loss = loss + lam_h0 * h0_prior_loss(H_pred)

        H_np = H_pred.detach().numpy().flatten()

        if use_pp:
            c2_pp = pp_chi2_np(H_np, z_sn,
                                mu_obs, cov_inv)
            loss = loss + lam_pp * torch.tensor(
                c2_pp, dtype=torch.float32)

        if use_desi:
            c2_de = desi_chi2_np(H_np)
            loss = loss + lam_desi * torch.tensor(
                c2_de, dtype=torch.float32)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        lv = loss.item()
        losses.append(lv)

        if lv < best:
            best   = lv
            best_w = {k: v.clone()
                      for k, v in
                      model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= patience:
            if verbose:
                print(f"    Early stop @ epoch {ep}  "
                      f"best={best:.4f}")
            break

        if verbose and ep % 1000 == 0:
            lr_now = sched.get_last_lr()[0]
            print(f"    ep {ep:5d}  "
                  f"loss={lv:.4f}  "
                  f"lr={lr_now:.2e}")

    if best_w:
        model.load_state_dict(best_w)
    return losses, best


# ════════════════════════════════════════════════════════════
# MC DROPOUT INFERENCE
# ════════════════════════════════════════════════════════════

def mc_predict(model, z_arr, n_samples=2000):
    model.train()  # keep dropout ON
    z_t = torch.tensor(
        z_arr, dtype=torch.float32).unsqueeze(1)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(
                model(z_t).numpy().flatten())
    preds = np.array(preds) * H0_NORM
    return preds.mean(0), preds.std(0), preds


# ════════════════════════════════════════════════════════════
# DATASET CONFIGS
# ════════════════════════════════════════════════════════════

CFG = {
    "OHD": dict(
        use_pp=False, use_desi=False,
        z_sn=None, mu_obs=None, cov_inv=None),
    "OHD_PP": dict(
        use_pp=True, use_desi=False,
        z_sn=z_pp, mu_obs=mu_pp, cov_inv=cov_pp),
    "OHD_PP_SH0ES": dict(
        use_pp=True, use_desi=False,
        z_sn=z_sh, mu_obs=mu_sh, cov_inv=cov_sh),
    "OHD_PP_SH0ES_DESI": dict(
        use_pp=True, use_desi=True,
        z_sn=z_sh, mu_obs=mu_sh, cov_inv=cov_sh),
}

# OHD tensors
z_t_ohd  = torch.tensor(
    z_ohd, dtype=torch.float32).unsqueeze(1)
H_t_ohd  = torch.tensor(
    H_ohd/H0_NORM, dtype=torch.float32).unsqueeze(1)
He_t_ohd = torch.tensor(
    sH_ohd/H0_NORM, dtype=torch.float32).unsqueeze(1)


# ════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════

bnn_results = {}

for combo in COMBOS:
    print(f"\n{'═'*60}")
    print(f"  BNN: {LABELS[combo]}")
    print(f"{'═'*60}")

    os.makedirs(f"results/{combo}", exist_ok=True)
    os.makedirs(f"figures/{combo}", exist_ok=True)

    cfg_c = CFG[combo]

    # Load paper params for comparison
    try:
        bf = np.load(f"results/{combo}/best_fit.npy")
        H0_p, l_p, m_p = bf[0], bf[1], bf[2]
    except FileNotFoundError:
        H0_p, l_p, m_p = DEFAULTS[combo]

    # ── Train ─────────────────────────────────────────────
    model = CosmoBNN().to("cpu")
    losses, best_loss = train(
        model, z_t_ohd, H_t_ohd, He_t_ohd,
        use_pp   = cfg_c["use_pp"],
        z_sn     = cfg_c["z_sn"],
        mu_obs   = cfg_c["mu_obs"],
        cov_inv  = cfg_c["cov_inv"],
        use_desi = cfg_c["use_desi"],
        n_epochs = 8000,
        verbose  = True)

    torch.save(model.state_dict(),
               f"results/{combo}/bnn_weights.pt")

    # ── MC inference ──────────────────────────────────────
    H_mean, H_std, H_all = mc_predict(
        model, Z_PRED, n_samples=N_MC)

    H0_bnn  = float(H_mean[0])
    H0_bstd = float(H_std[0])

    # ── Om(z) ─────────────────────────────────────────────
    Om_list, H0_list = [], []
    for Hs in H_all:
        H0s = float(Hs[0])
        if H0s < 30:
            continue
        H0_list.append(H0s)
        Om_list.append(Om_from_H(Z_PRED, Hs, H0s))

    Om_arr = np.array(Om_list)
    valid  = np.all(np.isfinite(Om_arr), axis=1)
    Om_arr = Om_arr[valid]
    Om_mean = Om_arr.mean(axis=0)
    Om_std  = Om_arr.std(axis=0)

    # ── Transition redshift ────────────────────────────────
    zt_list = []
    for Om_s in Om_arr[::5]:
        zts = find_transition_z(Z_PRED, Om_s)
        if len(zts) > 0 and 0.05 < zts[0] < 3.0:
            zt_list.append(float(zts[0]))
    zt_val = (float(np.median(zt_list))
              if len(zt_list) >= 5 else np.nan)
    zt_err = (float(np.std(zt_list))
              if len(zt_list) >= 5 else np.nan)

    print(f"  H0(BNN) = {H0_bnn:.2f} ± {H0_bstd:.2f}")
    print(f"  z_t(BNN) = {zt_val:.3f} ± {zt_err:.3f}")

    # ── Save ──────────────────────────────────────────────
    np.save(f"results/{combo}/bnn_H_mean.npy",    H_mean)
    np.save(f"results/{combo}/bnn_H_std.npy",     H_std)
    np.save(f"results/{combo}/bnn_Om_mean.npy",   Om_mean)
    np.save(f"results/{combo}/bnn_Om_std.npy",    Om_std)
    np.save(f"results/{combo}/bnn_Om_samples.npy",Om_arr[:500])
    np.save(f"results/{combo}/bnn_zt.npy",
            np.array([zt_val, zt_err]))
    np.save(f"results/{combo}/bnn_H0.npy",
            np.array([H0_bnn, H0_bstd]))

    bnn_results[combo] = dict(
        label   = LABELS[combo],
        H_mean  = H_mean, H_std  = H_std,
        Om_mean = Om_mean, Om_std = Om_std,
        zt=zt_val, zt_err=zt_err,
        H0=H0_bnn, H0_std=H0_bstd,
        losses=losses,
    )

    # ── Figures ───────────────────────────────────────────
    z_fine   = np.linspace(0.01, 3.0, 600)
    Om_paper = Om_parametric(z_fine, l_p, m_p)

    # Training loss
    fig0, ax0 = plt.subplots(figsize=(7,4))
    ax0.semilogy(losses, color='steelblue', lw=1.5)
    ax0.set_xlabel('Epoch', fontsize=12)
    ax0.set_ylabel(r'Loss ($\chi^2$)', fontsize=12)
    ax0.set_title(
        f'BNN Training — {LABELS[combo]}', fontsize=13)
    ax0.grid(True, alpha=0.3)
    fig0.tight_layout()
    fig0.savefig(f"figures/{combo}/BNN_loss_{combo}.pdf",
                 dpi=150)
    plt.close()

    # H(z)
    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.errorbar(z_ohd, H_ohd, yerr=sH_ohd,
                 fmt='o', ms=5, color='royalblue',
                 capsize=3, label='OHD', zorder=6)
    ax1.plot(Z_PRED, H_mean, 'tomato', lw=2,
             label='BNN mean')
    ax1.fill_between(Z_PRED,
                      H_mean-H_std, H_mean+H_std,
                      color='tomato', alpha=0.25,
                      label='BNN 1σ')
    ax1.fill_between(Z_PRED,
                      H_mean-2*H_std, H_mean+2*H_std,
                      color='tomato', alpha=0.08)
    ax1.plot(z_fine,
             H_parametric(z_fine, H0_p, l_p, m_p),
             'k--', lw=1.8, label='Parametric')
    ax1.plot(z_fine, H_LCDM(z_fine, H0_p),
             'gray', ls=':', lw=1.5,
             label=r'$\Lambda$CDM')
    ax1.set_xlabel('$z$', fontsize=12)
    ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]', fontsize=12)
    ax1.set_title(
        f'BNN $H(z)$ — {LABELS[combo]}', fontsize=13)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2.5)
    fig1.tight_layout()
    fig1.savefig(f"figures/{combo}/BNN_Hz_{combo}.pdf",
                 dpi=150)
    plt.close()

    # Om(z)
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.axhspan(0.29, 0.31, color='lightgray', alpha=0.4)
    ax2.axhline(0.30, color='gray', ls=':', lw=1.3,
                label=r'$\Lambda$CDM')
    ax2.plot(z_fine, Om_paper, 'k--', lw=2,
             label='Parametric')
    ax2.plot(Z_PRED, Om_mean, 'tomato', lw=2,
             label='BNN mean')
    ax2.fill_between(Z_PRED,
                      Om_mean-Om_std, Om_mean+Om_std,
                      color='tomato', alpha=0.28,
                      label='BNN 1σ')
    ax2.fill_between(Z_PRED,
                      Om_mean-2*Om_std, Om_mean+2*Om_std,
                      color='tomato', alpha=0.10)
    if not np.isnan(zt_val):
        ax2.axvline(zt_val, color='tomato', ls='--',
                    lw=1.5,
                    label=f'$z_t^{{\\rm BNN}}='
                          f'{zt_val:.2f}\\pm{zt_err:.2f}$')
    ax2.set_xlabel('$z$', fontsize=12)
    ax2.set_ylabel(r'$Om(z)$', fontsize=12)
    ax2.set_title(
        f'$Om(z)$ — BNN — {LABELS[combo]}', fontsize=13)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 3.0); ax2.set_ylim(-0.20, 0.70)
    fig2.tight_layout()
    fig2.savefig(f"figures/{combo}/BNN_Omz_{combo}.pdf",
                 dpi=150)
    plt.close()

    print(f"  Figures saved → figures/{combo}/")

np.save("results/bnn_results.npy", bnn_results)
print("\n[03_bnn_reconstruction.py] Complete.")
