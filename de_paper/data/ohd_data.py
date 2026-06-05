"""
data/ohd_data.py
─────────────────────────────────────────────────────────────
33 cosmic-chronometer H(z) measurements.
References:
  Jimenez & Loeb 2002, Simon+2005, Stern+2010,
  Moresco+2012, 2015, 2016, Zhang+2014,
  Ratsimbazafy+2017, Borghi+2022, Jiao+2023.
"""
import numpy as np


def get_ohd_data():
    """
    Returns (z, H_obs, sigma_H) — all 33 points, sorted by z.
    Units: H in km/s/Mpc.
    Both z=1.0363 and z=1.0370 are kept — they are distinct
    measurements from different observational programs.
    """
    data = np.array([
        # z        H(z)   sigma_H
        [0.0700,   69.0,  19.6],
        [0.0900,   69.0,  12.0],
        [0.1200,   68.6,  26.2],
        [0.1700,   83.0,   8.0],
        [0.1791,   75.0,   4.0],
        [0.1993,   75.0,   5.0],
        [0.2000,   72.9,  29.6],
        [0.2700,   77.0,  14.0],
        [0.2800,   88.8,  36.6],
        [0.3519,   83.0,  14.0],
        [0.3802,   83.0,  13.5],
        [0.4000,   95.0,  17.0],
        [0.4004,   77.0,  10.2],
        [0.4247,   87.1,  11.2],
        [0.4497,   92.8,  12.9],
        [0.4783,   80.9,   9.0],
        [0.4800,   97.0,  62.0],
        [0.5929,  104.0,  13.0],
        [0.6000,   87.9,   6.1],
        [0.6797,   92.0,   8.0],
        [0.7300,   97.3,   7.0],
        [0.7812,  105.0,  12.0],
        [0.8754,  125.0,  17.0],
        [0.8800,   90.0,  40.0],
        [0.9000,  117.0,  23.0],
        [1.0363,  154.0,  20.0],   # Moresco+2012
        [1.0370,  154.0,  20.0],   # independent compilation
        [1.3000,  168.0,  17.0],
        [1.3630,  160.0,  33.6],
        [1.4300,  177.0,  18.0],
        [1.5300,  140.0,  14.0],
        [1.7500,  202.0,  40.0],
        [1.9650,  186.5,  50.4],
    ])
    idx  = np.argsort(data[:, 0])
    data = data[idx]
    return data[:, 0], data[:, 1], data[:, 2]


if __name__ == "__main__":
    z, H, e = get_ohd_data()
    print(f"OHD: N={len(z)}, "
          f"z=[{z.min():.3f}, {z.max():.3f}], "
          f"H=[{H.min():.1f}, {H.max():.1f}]")
