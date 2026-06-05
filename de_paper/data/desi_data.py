"""
data/desi_data.py
─────────────────────────────────────────────────────────────
DESI BAO DR2 (2024) measurements and chi2 computation.

Reference: DESI Collaboration, arXiv:2404.03002 (2024)

Observables:
  DM/rd  — comoving angular diameter distance / sound horizon
  DH/rd  — Hubble distance c/H(z) / sound horizon
  DV/rd  — spherically averaged distance / sound horizon

The sound horizon rd is a derived quantity; we marginalise
over it analytically following the standard approach.
"""
import numpy as np
from scipy.integrate import quad

C_LIGHT = 2.998e5   # km/s

# ════════════════════════════════════════════════════════════
# DESI DR2 BAO measurements
# Table 1 of DESI 2024 (arXiv:2404.03002)
# Each row: (z_eff, type, value, sigma)
# type: 'DM_rd' | 'DH_rd' | 'DV_rd'
# ════════════════════════════════════════════════════════════

DESI_DR2 = [
    # BGS
    (0.295, 'DV_rd',  7.93,  0.15),
    # LRG1
    (0.510, 'DM_rd', 13.62,  0.25),
    (0.510, 'DH_rd', 20.98,  0.61),
    # LRG2
    (0.706, 'DM_rd', 16.85,  0.32),
    (0.706, 'DH_rd', 20.08,  0.60),
    # LRG3+ELG1
    (0.930, 'DM_rd', 21.71,  0.28),
    (0.930, 'DH_rd', 17.88,  0.35),
    # ELG2
    (1.317, 'DM_rd', 27.79,  0.69),
    (1.317, 'DH_rd', 13.82,  0.42),
    # QSO
    (1.491, 'DV_rd', 26.07,  0.67),
    # Lya QSO
    (2.330, 'DM_rd', 39.71,  0.94),
    (2.330, 'DH_rd',  8.52,  0.17),
]

# Correlation coefficients between DM/rd and DH/rd
# for the same redshift bin (from DESI 2024 Table 1)
DESI_CORR = {
    0.510: -0.445,
    0.706: -0.431,
    0.930: -0.389,
    1.317: -0.467,
    2.330: -0.437,
}

# Standard sound horizon (Planck 2018 best-fit, used as prior)
RD_PLANCK = 147.09   # Mpc
RD_SIGMA  = 0.26     # Mpc (Planck uncertainty)


def get_desi_data():
    """Return the DESI DR2 BAO measurements as arrays."""
    rows = np.array([(z, v, s)
                     for z, t, v, s in DESI_DR2],
                    dtype=float)
    types = [t for z, t, v, s in DESI_DR2]
    zeffs = [z for z, t, v, s in DESI_DR2]
    return zeffs, types, rows[:, 1], rows[:, 2]


# ════════════════════════════════════════════════════════════
# Distance calculations
# ════════════════════════════════════════════════════════════

def comoving_distance(z, Hz_func):
    """DC(z) = c * integral_0^z dz'/H(z') in Mpc."""
    I, _ = quad(lambda zp: C_LIGHT / Hz_func(zp),
                0.0, z, limit=300, epsrel=1e-7)
    return I


def DM(z, Hz_func):
    """Comoving angular diameter distance (flat)."""
    return comoving_distance(z, Hz_func)


def DH(z, Hz_func):
    """Hubble distance c/H(z)."""
    return C_LIGHT / Hz_func(z)


def DV(z, Hz_func):
    """Spherically-averaged distance."""
    dm = DM(z, Hz_func)
    dh = DH(z, Hz_func)
    return (z * dm**2 * dh)**(1.0/3.0)


# ════════════════════════════════════════════════════════════
# Chi2 for DESI BAO (Option B — full chi2 with rd)
# ════════════════════════════════════════════════════════════

def chi2_desi(Hz_func, rd=RD_PLANCK):
    """
    Full BAO chi2 using DM/rd, DH/rd, DV/rd predictions.

    Hz_func : callable, takes scalar z, returns H(z) in km/s/Mpc
    rd      : sound horizon in Mpc (default: Planck value)

    Returns scalar chi2.
    """
    c2 = 0.0
    # Group by redshift for correlated pairs
    processed = set()

    for z_eff, obs_type, obs_val, obs_sig in DESI_DR2:
        if z_eff in processed:
            continue

        # Find all measurements at this z_eff
        pts = [(ot, ov, os_) for ze, ot, ov, os_
               in DESI_DR2 if ze == z_eff]

        if len(pts) == 1:
            # Single measurement (DV/rd)
            ot, ov, os_ = pts[0]
            if ot == 'DV_rd':
                pred = DV(z_eff, Hz_func) / rd
            elif ot == 'DM_rd':
                pred = DM(z_eff, Hz_func) / rd
            else:
                pred = DH(z_eff, Hz_func) / rd
            c2 += ((ov - pred) / os_) ** 2

        elif len(pts) == 2:
            # Correlated DM/rd and DH/rd pair
            dm_obs = next(v for t,v,s in pts if t=='DM_rd')
            dh_obs = next(v for t,v,s in pts if t=='DH_rd')
            dm_sig = next(s for t,v,s in pts if t=='DM_rd')
            dh_sig = next(s for t,v,s in pts if t=='DH_rd')
            rho    = DESI_CORR.get(z_eff, 0.0)

            dm_pred = DM(z_eff, Hz_func) / rd
            dh_pred = DH(z_eff, Hz_func) / rd

            ddm = dm_obs - dm_pred
            ddh = dh_obs - dh_pred

            det = 1.0 - rho**2
            c2 += (1.0/det) * (
                (ddm/dm_sig)**2
                - 2*rho*(ddm/dm_sig)*(ddh/dh_sig)
                + (ddh/dh_sig)**2
            )

        processed.add(z_eff)

    # Add Gaussian prior on rd
    c2 += ((rd - RD_PLANCK) / RD_SIGMA) ** 2

    return float(c2)


# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Quick test with LCDM
    def H_lcdm(z, H0=67.4, Om=0.315):
        return H0 * np.sqrt(Om*(1+z)**3 + (1-Om))

    c2 = chi2_desi(lambda z: H_lcdm(z))
    print(f"DESI DR2: {len(DESI_DR2)} data points "
          f"across {len(set(z for z,*_ in DESI_DR2))} "
          f"redshift bins")
    print(f"chi2 (Planck LCDM): {c2:.3f}")
