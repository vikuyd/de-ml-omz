# Model-Independent Validation of the Phantom-to-Quintessence Dark Energy Transition

**Vikrant Yadav** — Department of Mathematics, School of CS & AI, SR University, Warangal, Telangana, India

This repository contains all analysis code for the paper:

> *Model-Independent Validation of the Phantom-to-Quintessence Dark Energy Transition Using Gaussian Processes, Bayesian Neural Networks, and Symbolic Regression: A Multi-Dataset Analysis with OHD, Pantheon+, SH0ES, and DESI BAO DR2*

---

## Repository structure

```
de-ml-omz/
├── run_all.py                     ← master pipeline runner
├── 01_mcmc.py                     ← MCMC parameter estimation
├── 02_gp_reconstruction.py        ← Gaussian Process reconstruction
├── 03_bnn_reconstruction.py       ← Bayesian Neural Network
├── 04_symbolic_regression.py      ← Symbolic Regression (PySR)
├── 05_master_figures.py           ← all publication figures
│
├── data/
│   ├── ohd_data.py                ← 33 cosmic chronometer H(z) measurements
│   ├── pantheon_data.py           ← Pantheon+ data loader
│   ├── desi_data.py               ← DESI DR2 BAO chi-squared
│   └── pantheon_plus/             ← place Pantheon+ data files here (see below)
│       ├── Pantheon+SH0ES.dat
│       └── Pantheon+SH0ES_STAT+SYS.cov
│
├── models/
│   └── cosmo_model.py             ← H(z), Om(z), chi2, AIC/BIC, age functions
│
├── results/                       ← created automatically by pipeline
└── figures/                       ← created automatically by pipeline
```

---

## Requirements

Python 3.8 or higher. Install all dependencies with:

```bash
pip install -r requirements.txt
```

Or individually:

```bash
pip install numpy scipy matplotlib torch emcee george corner pandas tqdm astropy pysr
```

**PySR** (Symbolic Regression) requires Julia to be installed on your system.
Install instructions: https://astroautomata.com/PySR/

If PySR/Julia is not available, Steps 1–3 and all figures will still run.
Step 4 (symbolic regression) will be skipped automatically with a warning.

---

## Data

### OHD (cosmic chronometers)
The 33 OHD measurements are hard-coded in `data/ohd_data.py` — no download needed.

### Pantheon+ and SH0ES
Download two files from the official PantheonPlusSH0ES data release:

```
https://github.com/PantheonPlusSH0ES/DataRelease
```

Navigate to `Pantheon+SH0ES_Data/4_DISTANCES_AND_COVAR/` and download:

- `Pantheon+SH0ES.dat` (~1.2 MB)
- `Pantheon+SH0ES_STAT+SYS.cov` (~23 MB)

Place both files in `data/pantheon_plus/`.

**Verify the data is found:**
```bash
python3 data/pantheon_data.py
```
You should see `[REAL]` next to each dataset, not `[MOCK]`.

### DESI BAO DR2
The 12 DESI DR2 BAO measurements and their covariance are hard-coded in
`data/desi_data.py` — no download needed.
Source: DESI Collaboration (2025), Phys. Rev. D 112, 083515.
DOI: 10.1103/tr6y-kpc6

---

## Running the pipeline

```bash
# Full pipeline (~90 min on a laptop)
python3 run_all.py

# Skip MCMC (use saved results), run GP + BNN + figures only
python3 run_all.py --skip-mcmc

# Run a single step
python3 run_all.py --step 1   # MCMC only
python3 run_all.py --step 2   # GP only
python3 run_all.py --step 3   # BNN only
python3 run_all.py --step 4   # Symbolic Regression only
python3 run_all.py --step 5   # Figures only

# Quick test with default parameters (no MCMC)
python3 run_all.py --test
```

### Expected runtimes (standard laptop, 8 cores)

| Step | Description | Time |
|------|-------------|------|
| 1 | MCMC (32 walkers × 3000 steps × 4 combinations) | ~80 min |
| 2 | GP reconstruction (Matérn-5/2, 5000 MC draws) | ~10 min |
| 3 | BNN training (4 combinations, early stopping) | ~25 min |
| 4 | Symbolic Regression (PySR, 50 iterations) | ~10 min |
| 5 | Publication figures | ~3 min |

---

## Key results

After running the pipeline, the summary table is printed to the terminal and
all results are saved to `results/`. Publication figures go to `figures/master/`.

The key scientific result — symbolic regression independently recovering
`Om(z) = z^1.212 / (1+z)^2.064` from the GP-reconstructed curve — is produced
by Step 4 and visualised in `figures/master/Fig_symreg.pdf`.

---

## Citation

If you use this code, please cite:

```
Yadav, V. (2026). Model-Independent Validation of the Phantom-to-Quintessence
Dark Energy Transition Using Gaussian Processes, Bayesian Neural Networks, and
Symbolic Regression. [journal details upon acceptance]
```

The parent paper introducing the Om(z) parametrisation:

```
Yadav, M., Dixit, A., Pradhan, A., & Barak, M. S. (2026).
Dynamical Dark Energy Signatures from a New Transition Om(z) Parametrization
in Flat FLRW Cosmology. Phys. Lett. B, 140238.
DOI: 10.1016/j.physletb.2026.140238
```

---

## License

MIT License. See `LICENSE` for details.
