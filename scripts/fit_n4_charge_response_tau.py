from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PY = Path(r"C:\Users\lmf\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe")
SIM = ROOT / "scripts" / "simulate_multigrid_teng_calibrated.py"
OUTBASE = ROOT / "results_multigrid_calibrated_a4_smoothed"
TARGET_ISC_PP_UA = 11.33357946976


def run_tau(tau: float, outdir: Path) -> tuple[float, float]:
    subprocess.run(
        [str(PY), str(SIM), "--outdir", str(outdir), "--load-for-waveforms", "1G", "--charge-response-tau", str(tau)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    csv_path = outdir / "teng_N4_1G_timeseries.csv"
    current = []
    charge = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            current.append(float(row["source_current_A"]) * 1e6)
            charge.append(float(row["source_charge_C"]) * 1e9)
    current = np.array(current)
    charge = np.array(charge)
    return float(current.max() - current.min()), float(charge.max() - charge.min())


def main() -> None:
    OUTBASE.mkdir(parents=True, exist_ok=True)
    lo, hi = 0.0, 0.08
    # Ensure upper bound reduces peak sufficiently.
    for _ in range(8):
        pp, _ = run_tau(hi, OUTBASE / "_fit_tmp")
        if pp < TARGET_ISC_PP_UA:
            break
        hi *= 1.8
    best_tau = hi
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        pp, _ = run_tau(mid, OUTBASE / "_fit_tmp")
        if pp > TARGET_ISC_PP_UA:
            lo = mid
        else:
            hi = mid
        best_tau = hi
    final_pp, final_qpp = run_tau(best_tau, OUTBASE)
    (OUTBASE / "charge_response_tau_fit.txt").write_text(
        f"target_isc_pp_uA={TARGET_ISC_PP_UA:.12e}\n"
        f"fitted_tau_s={best_tau:.12e}\n"
        f"simulated_isc_pp_uA={final_pp:.12e}\n"
        f"simulated_q_pp_nC={final_qpp:.12e}\n",
        encoding="utf-8",
    )
    print(f"tau={best_tau:.8f} s")
    print(f"Isc_pp={final_pp:.4f} uA")
    print(f"Qpp={final_qpp:.4f} nC")


if __name__ == "__main__":
    main()
