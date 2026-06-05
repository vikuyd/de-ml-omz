"""
run_all.py — Master pipeline runner
─────────────────────────────────────────────────────────────
Usage:
  python3 run_all.py                  full pipeline
  python3 run_all.py --skip-mcmc      skip MCMC (use saved)
  python3 run_all.py --step 2         run only step 2
  python3 run_all.py --test           quick test (no MCMC)

Steps:
  1. MCMC         (~90 min total across 4 combos)
  2. GP           (~10 min)
  3. BNN          (~25 min)
  4. Symbolic reg (~10 min, needs PySR)
  5. Figures      (~3 min)
"""

import sys, os, time, argparse, importlib.util
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--skip-mcmc', action='store_true',
    help='Skip MCMC, use saved or default parameters')
parser.add_argument('--step', type=int, default=0,
    help='Run only this step number (1-5)')
parser.add_argument('--test', action='store_true',
    help='Quick test: inject defaults, skip MCMC')
args = parser.parse_args()

# ── Create all directories ────────────────────────────────
for combo in ["OHD","OHD_PP","OHD_PP_SH0ES",
              "OHD_PP_SH0ES_DESI"]:
    os.makedirs(f"results/{combo}", exist_ok=True)
    os.makedirs(f"figures/{combo}", exist_ok=True)
os.makedirs("figures/master", exist_ok=True)

# ── Inject defaults if skipping MCMC ─────────────────────
if args.skip_mcmc or args.test:
    DEFAULTS = {
        "OHD":               [72.12, 0.728, 1.823],
        "OHD_PP":            [72.73, 0.728, 1.823],
        "OHD_PP_SH0ES":      [73.01, 0.451, 1.810],
        "OHD_PP_SH0ES_DESI": [73.10, 0.440, 1.790],
    }
    for combo, p in DEFAULTS.items():
        np.save(f"results/{combo}/best_fit.npy",
                np.array(p))
    print("Default parameters injected.")


# ── Step runner ───────────────────────────────────────────
def run(name, script, skip=False):
    if skip:
        print(f"\n{'─'*50}")
        print(f"  SKIP: {name}")
        print(f"{'─'*50}")
        return True
    print(f"\n{'═'*50}")
    print(f"  STEP: {name}")
    print(f"{'═'*50}")
    t0   = time.time()
    spec = importlib.util.spec_from_file_location(
        name, script)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
        elapsed = time.time()-t0
        print(f"\n  ✓ {name}  ({elapsed:.1f}s)")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  ✗ {name} FAILED: {e}")
        return False


STEPS = {
    1: ("MCMC",          "01_mcmc.py",
        args.skip_mcmc or args.test),
    2: ("GP",            "02_gp_reconstruction.py", False),
    3: ("BNN",           "03_bnn_reconstruction.py", False),
    4: ("SymbolicReg",   "04_symbolic_regression.py",False),
    5: ("MasterFigures", "05_master_figures.py",     False),
}

t_start  = time.time()
run_ids  = [args.step] if args.step else list(STEPS.keys())
statuses = {}

for sid in run_ids:
    name, script, skip = STEPS[sid]
    ok = run(name, script, skip)
    statuses[sid] = ok
    if not ok:
        print(f"\nPipeline halted at step {sid}.")
        break

# ── Final summary ─────────────────────────────────────────
total = time.time()-t_start
print(f"\n{'═'*55}")
print(f"  PIPELINE COMPLETE  ({total:.1f}s total)")
print(f"{'═'*55}")
for sid, ok in statuses.items():
    mark = '✓' if ok else '✗'
    print(f"  {mark}  Step {sid}: {STEPS[sid][0]}")

print("\nKey output files:")
for combo in ["OHD","OHD_PP","OHD_PP_SH0ES",
              "OHD_PP_SH0ES_DESI"]:
    for fn in ["best_fit.npy","gp_zt.npy","bnn_zt.npy"]:
        fp = f"results/{combo}/{fn}"
        if os.path.exists(fp):
            sz = os.path.getsize(fp)/1024
            print(f"  {fp:<52} {sz:5.1f} KB")

print("\nPublication figures:")
for fn in sorted(os.listdir("figures/master")):
    fp = f"figures/master/{fn}"
    sz = os.path.getsize(fp)/1024
    print(f"  {fp:<52} {sz:5.1f} KB")
