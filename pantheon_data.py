"""
data/pantheon_data.py
─────────────────────────────────────────────────────────────
Pantheon+ and SH0ES supernova data loader.

REAL DATA (download once and place in data/pantheon_plus/):
  https://github.com/PantheonPlusSH0ES/DataRelease
  Files needed:
    Pantheon+SH0ES.dat
    Pantheon+SH0ES_STAT+SYS.cov

If files are absent a realistic mock is used automatically
so the pipeline runs end-to-end for testing.
"""

import os
import numpy as np
from scipy.integrate import quad

# ── Paths ────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_HERE, "pantheon_plus")
DAT_FILE  = os.path.join(DATA_DIR, "Pantheon+SH0ES.dat")
COV_FILE  = os.path.join(DATA_DIR, "Pantheon+SH0ES_STAT+SYS.cov")
C_LIGHT   = 2.998e5          # km/s


# ════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════

def get_pantheon_data(dataset="PP_SH0ES"):
    """
    Parameters
    ----------
    dataset : "PP" | "PP_SH0ES"

    Returns
    -------
    z_sn    : (N,) redshifts
    mu_obs  : (N,) observed distance moduli
    cov_inv : (N,N) inverse covariance
    mu_err  : (N,) diagonal sqrt of covariance
    is_mock : bool
    """
    if os.path.exists(DAT_FILE) and os.path.exists(COV_FILE):
        return _load_real(dataset)
    print(f"[pantheon] Real files not found in {DATA_DIR}/")
    print(f"[pantheon] Using MOCK data.")
    return _make_mock(dataset)


# ════════════════════════════════════════════════════════════
# Real data loader
# ════════════════════════════════════════════════════════════

def _load_real(dataset):
    import pandas as pd

    df = pd.read_csv(DAT_FILE, sep=r'\s+', comment='#')
    print(f"[pantheon] Columns: {list(df.columns)}")

    # ── redshift ─────────────────────────────────────────────
    for c in ['zHD', 'zhd', 'z_HD', 'zCMB', 'zcmb', 'z']:
        if c in df.columns:
            z_col = c; break
    else:
        raise KeyError(f"No redshift column. Have: {list(df.columns)}")

    # ── distance modulus ─────────────────────────────────────
    if dataset == "PP_SH0ES":
        mu_cands = ['MU_SH0ES', 'mu_sh0es', 'MU', 'mu',
                    'm_b_corr', 'mB']
    else:
        mu_cands = ['MU', 'mu', 'MU_SH0ES',
                    'm_b_corr', 'mB']
    for c in mu_cands:
        if c in df.columns:
            mu_col = c; break
    else:
        raise KeyError(
            f"No mu column. Tried {mu_cands}. "
            f"Have: {list(df.columns)}")

    # ── diagonal error ────────────────────────────────────────
    err_cands = ['MU_SH0ES_ERR_DIAG', 'MUERR_FINAL',
                 'MU_ERR', 'mu_err', 'muerr',
                 'mBERR', 'x1ERR', 'MUERR']
    for c in err_cands:
        if c in df.columns:
            me_col = c; break
    else:
        print("[pantheon] No error column found; using 5% of mu.")
        df['_sig'] = df[mu_col].abs() * 0.05
        me_col = '_sig'

    print(f"[pantheon] z={z_col}  mu={mu_col}  err={me_col}")

    # ── filter ────────────────────────────────────────────────
    if dataset == "PP":
        if 'IS_CALIBRATOR' in df.columns:
            df = df[df['IS_CALIBRATOR'] == 0]
        df = df.head(1048)

    z_sn   = df[z_col].values.astype(float)
    mu_obs = df[mu_col].values.astype(float)
    mu_err = df[me_col].values.astype(float)

    # ── covariance ────────────────────────────────────────────
    N = len(z_sn)
    try:
        raw = np.loadtxt(COV_FILE)
        if raw.size == N * N:
            cov = raw.reshape(N, N)
        elif raw.size == (N + 1) * N:
            cov = raw[1:].reshape(N, N)
        else:
            # First line may be the size integer
            cov = raw.reshape(-1)[-N*N:].reshape(N, N)
        cov += np.diag(mu_err ** 2)
    except Exception as exc:
        print(f"[pantheon] Cov load failed ({exc}); "
              f"using diagonal.")
        cov = np.diag(mu_err ** 2)

    cov_inv = np.linalg.inv(cov)
    print(f"[pantheon] {dataset}: N={N}  "
          f"z=[{z_sn.min():.3f}, {z_sn.max():.3f}]")
    return z_sn, mu_obs, cov_inv, mu_err, False


# ════════════════════════════════════════════════════════════
# Mock generator
# ════════════════════════════════════════════════════════════

def _lcdm_mu(z, H0=73.04, Om=0.334):
    out = []
    for zi in np.atleast_1d(z):
        if zi < 1e-4:
            out.append(np.nan); continue
        I, _ = quad(lambda zp: 1/np.sqrt(
            Om*(1+zp)**3+(1-Om)), 0, zi, limit=200)
        dL = (1+zi)*(C_LIGHT/H0)*I
        out.append(5*np.log10(dL)+25)
    return np.array(out)


def _make_mock(dataset):
    rng = np.random.default_rng(42)
    N   = 1701 if dataset == "PP_SH0ES" else 1048
    z1  = rng.uniform(0.001, 0.10, int(0.15*N))
    z2  = rng.uniform(0.10,  0.80, int(0.65*N))
    z3  = rng.uniform(0.80,  2.26, N-len(z1)-len(z2))
    z_sn = np.sort(np.clip(
        np.concatenate([z1,z2,z3]), 1e-4, 2.26))
    mu_t = _lcdm_mu(z_sn)
    ok   = np.isfinite(mu_t)
    z_sn, mu_t = z_sn[ok], mu_t[ok]
    N    = len(z_sn)
    sig  = 0.12 + 0.04*(z_sn>0.5) + 0.03*(z_sn>1.0)
    mu_obs  = mu_t + rng.normal(0, sig, N)
    cov_inv = np.diag(1.0/sig**2)
    print(f"[pantheon] MOCK {dataset}: N={N}")
    return z_sn, mu_obs, cov_inv, sig, True


# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    for ds in ["PP", "PP_SH0ES"]:
        z, mu, ci, se, mock = get_pantheon_data(ds)
        tag = "MOCK" if mock else "REAL"
        print(f"{ds:12s} [{tag}]  N={len(z)}  "
              f"z=[{z.min():.3f},{z.max():.3f}]")
