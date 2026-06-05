"""
models/cosmo_model.py
─────────────────────────────────────────────────────────────
All cosmological functions for the paper.
"""
import numpy as np
from scipy.integrate import quad

C_LIGHT = 2.998e5   # km/s


# ════════════════════════════════════════════════════════════
# Yadav (2026) parametrisation
# ════════════════════════════════════════════════════════════

def Om_parametric(z, l, m):
    """Om(z) = z^l / (1+z)^m.  Safe at z=0 (returns 0)."""
    z = np.atleast_1d(np.float64(z))
    with np.errstate(divide='ignore', invalid='ignore'):
        val = np.where(z == 0.0, 0.0,
                       z**l / (1.0+z)**m)
    return float(val[0]) if val.size == 1 else val


def H_parametric(z, H0, l, m):
    """H(z) from the Yadav Om(z) parametrisation."""
    z  = np.atleast_1d(np.float64(z))
    Om = Om_parametric(z, l, m)
    v  = H0 * np.sqrt(1.0 + Om*((1.0+z)**3 - 1.0))
    return float(v[0]) if v.size == 1 else v


def H_LCDM(z, H0, Omega_m=0.315):
    """Flat LCDM Hubble parameter."""
    z = np.atleast_1d(np.float64(z))
    v = H0 * np.sqrt(Omega_m*(1+z)**3 + (1-Omega_m))
    return float(v[0]) if v.size == 1 else v


# ════════════════════════════════════════════════════════════
# Om(z) from reconstructed H(z)
# ════════════════════════════════════════════════════════════

def Om_from_H(z, H, H0):
    """
    Om(z) = ( (H/H0)^2 - 1 ) / ( (1+z)^3 - 1 )
    Returns NaN at z=0.
    """
    z = np.atleast_1d(np.float64(z))
    H = np.atleast_1d(np.float64(H))
    E2    = (H / H0)**2
    denom = (1.0+z)**3 - 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        Om = np.where(np.abs(denom) < 1e-9,
                      np.nan, (E2-1.0)/denom)
    return Om


# ════════════════════════════════════════════════════════════
# Distance modulus
# ════════════════════════════════════════════════════════════

def _dL(z, Hz_func):
    if z <= 0:
        return 1e-10
    I, _ = quad(lambda zp: C_LIGHT/Hz_func(zp),
                0.0, z, limit=300, epsrel=1e-7)
    return (1.0+z) * I


def mu_model(z_arr, H0, l, m):
    """Distance modulus array for the parametric model."""
    Hz = lambda z: H_parametric(z, H0, l, m)
    return np.array([5*np.log10(_dL(zi, Hz))+25
                     for zi in np.asarray(z_arr)])


# ════════════════════════════════════════════════════════════
# Deceleration parameter
# ════════════════════════════════════════════════════════════

def q_model(z, H0, l, m):
    """q(z) = -1 + (1+z)/H * dH/dz (numerical)."""
    z   = np.atleast_1d(np.float64(z))
    dz  = 1e-5
    Hz  = H_parametric(z,    H0, l, m)
    Hzp = H_parametric(z+dz, H0, l, m)
    return -1.0 + (1.0+z)*(Hzp-Hz)/(dz*Hz)


# ════════════════════════════════════════════════════════════
# Age of universe
# ════════════════════════════════════════════════════════════

def age_Gyr(H0, l, m):
    """
    t0 in Gyr.
    Conversion: 1 Mpc/(km/s) = 977.8 Gyr
    """
    integrand = lambda z: 1.0 / (
        (1.0+z) * H_parametric(z, H0, l, m))
    val, _ = quad(integrand, 0.0, 1100.0,
                  limit=1000, epsrel=1e-7)
    return val * 977.8


# ════════════════════════════════════════════════════════════
# Chi-squared
# ════════════════════════════════════════════════════════════

def chi2_ohd(params, z, H_obs, H_err):
    H0, l, m = params
    H_th = H_parametric(z, H0, l, m)
    return float(np.sum(((H_obs-H_th)/H_err)**2))


def chi2_pp(params, z_sn, mu_obs, cov_inv):
    H0, l, m = params
    mu_th = mu_model(z_sn, H0, l, m)
    d = mu_obs - mu_th
    return float(d @ cov_inv @ d)


def chi2_joint(params,
               z_ohd, H_ohd, sH_ohd,
               z_sn=None, mu_obs=None, cov_inv=None,
               Hz_func_for_desi=None,
               use_pp=False, use_desi=False,
               rd=147.09):
    """
    Full joint chi2.
    Hz_func_for_desi: callable z->H(z) built from params
    """
    from data.desi_data import chi2_desi
    c2 = chi2_ohd(params, z_ohd, H_ohd, sH_ohd)
    if use_pp and z_sn is not None:
        c2 += chi2_pp(params, z_sn, mu_obs, cov_inv)
    if use_desi and Hz_func_for_desi is not None:
        c2 += chi2_desi(Hz_func_for_desi, rd=rd)
    return c2


# ════════════════════════════════════════════════════════════
# Information criteria
# ════════════════════════════════════════════════════════════

def AIC(chi2_min, n_params):
    return chi2_min + 2*n_params

def BIC(chi2_min, n_params, n_data):
    return chi2_min + n_params*np.log(n_data)


# ════════════════════════════════════════════════════════════
# Transition redshift
# ════════════════════════════════════════════════════════════

def find_transition_z(z_arr, Om_arr):
    """
    Find redshifts where d(Om)/dz changes sign.
    Returns array of crossing redshifts (may be empty).
    """
    dOm = np.gradient(Om_arr, z_arr)
    idx = np.where(np.diff(np.sign(dOm)) != 0)[0]
    return z_arr[idx]


# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    z = np.array([0.1, 0.5, 1.0, 2.0])
    p = (72.12, 0.728, 1.823)
    print("H(z):", H_parametric(z, *p))
    print("q(z):", q_model(z, *p))
    print("t0  :", age_Gyr(*p), "Gyr")
