from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SIM_CSV = ROOT / "results_multigrid_calibrated" / "short_circuit_N4_full_cycle.csv"
EXP_TXT = Path(r"C:\Users\lmf\Desktop\20260507\4ELE\2-TENG\a-5-Q-650nC.txt")
OUTDIR = ROOT / "results_multigrid_calibrated" / "n4_charge_experiment_compare"


def load_sim() -> tuple[np.ndarray, np.ndarray]:
    t, q = [], []
    with SIM_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["time_s"]))
            q.append(float(row["integrated_transferred_charge_nC"]))
    return np.array(t), np.array(q)


def load_lvm_charge(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    in_data = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("X_Value"):
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"No numeric data parsed from {path}")
    data = np.array(rows)
    t = data[:, 0]
    q = data[:, 1] * 1e9
    return t, q


def detrend_to_zero(q: np.ndarray) -> np.ndarray:
    n = max(5, len(q) // 100)
    return q - float(np.mean(q[:n]))


def resample_to_common(te: np.ndarray, qe: np.ndarray, ts: np.ndarray, qs: np.ndarray, n: int = 2000):
    xe = (te - te[0]) / (te[-1] - te[0])
    xs = (ts - ts[0]) / (ts[-1] - ts[0])
    x = np.linspace(0.0, 1.0, n)
    qe_i = np.interp(x, xe, qe)
    qs_i = np.interp(x, xs, qs)
    return x, qe_i, qs_i


def pick_best_cycle(t: np.ndarray, q: np.ndarray, ts: np.ndarray, qs: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    sim_duration = float(ts[-1] - ts[0])
    dt = float(np.median(np.diff(t)))
    window = max(10, int(round(sim_duration / dt)))
    step = max(1, window // 30)
    qs0 = detrend_to_zero(qs)
    xs = (ts - ts[0]) / sim_duration
    x_common = np.linspace(0.0, 1.0, 1600)
    qs_i = np.interp(x_common, xs, qs0)
    qs_i = qs_i - np.mean(qs_i)
    qs_std = float(np.std(qs_i))
    best: tuple[float, int, int, float, float] | None = None
    for start in range(0, len(t) - window, step):
        end = start + window
        tw = t[start:end] - t[start]
        qw = detrend_to_zero(q[start:end])
        amp = float(np.max(qw) - np.min(qw))
        if amp < 100.0:
            continue
        xw = tw / tw[-1]
        qw_i = np.interp(x_common, xw, qw)
        qw_i = qw_i - np.mean(qw_i)
        qw_std = float(np.std(qw_i))
        if qw_std <= 0 or qs_std <= 0:
            continue
        corr = float(np.corrcoef(qw_i, qs_i)[0, 1])
        corr_abs = abs(corr)
        amp_score = math.exp(-((amp - 650.0) / 120.0) ** 2)
        score = corr_abs + 0.15 * amp_score
        if best is None or score > best[0]:
            best = (score, start, end, corr, amp)
    if best is None:
        raise RuntimeError("Could not find a suitable experimental cycle.")
    _, start, end, corr, amp = best
    te = t[start:end] - t[start]
    qe = detrend_to_zero(q[start:end])
    return te, qe, {
        "selected_start_s": float(t[start]),
        "selected_end_s": float(t[end - 1]),
        "selected_duration_s": float(te[-1] - te[0]),
        "window_correlation_before_sign": float(corr),
        "selected_charge_pp_nC": float(amp),
    }


def svg_plot(path: Path, title: str, xlabel: str, ylabel: str, series: list[tuple[np.ndarray, np.ndarray, str, str]]) -> None:
    width, height = 840, 460
    left, right, top, bottom = 92, 28, 34, 66
    xmin = min(float(np.min(x)) for x, _, _, _ in series)
    xmax = max(float(np.max(x)) for x, _, _, _ in series)
    ymin = min(float(np.min(y)) for _, y, _, _ in series)
    ymax = max(float(np.max(y)) for _, y, _, _ in series)
    pad = 0.10 * (ymax - ymin if ymax != ymin else 1.0)
    ymin -= pad
    ymax += pad

    def pts(x: np.ndarray, y: np.ndarray) -> str:
        px = left + (x - xmin) / (xmax - xmin) * (width - left - right)
        py = top + (ymax - y) / (ymax - ymin) * (height - top - bottom)
        return " ".join(f"{float(a):.2f},{float(b):.2f}" for a, b in zip(px, py))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="23" text-anchor="middle" font-family="Arial" font-size="15">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="12">{xlabel}</text>',
        f'<text x="16" y="{height/2}" text-anchor="middle" transform="rotate(-90 16 {height/2})" font-family="Arial" font-size="12">{ylabel}</text>',
    ]
    for frac in np.linspace(0, 1, 6):
        yy = top + frac * (height - top - bottom)
        value = ymax - frac * (ymax - ymin)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e1e5ea"/>')
        lines.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{value:.1f}</text>')
    for frac in np.linspace(0, 1, 6):
        xx = left + frac * (width - left - right)
        value = xmin + frac * (xmax - xmin)
        lines.append(f'<line x1="{xx:.1f}" y1="{height-bottom}" x2="{xx:.1f}" y2="{height-bottom+5}" stroke="#333"/>')
        lines.append(f'<text x="{xx:.1f}" y="{height-bottom+20}" text-anchor="middle" font-family="Arial" font-size="10">{value:.2f}</text>')
    if ymin < 0 < ymax:
        y0 = top + (ymax - 0) / (ymax - ymin) * (height - top - bottom)
        lines.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{width-right}" y2="{y0:.1f}" stroke="#999" stroke-dasharray="4 4"/>')
    for idx, (x, y, label, color) in enumerate(series):
        lines.append(f'<polyline points="{pts(x, y)}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{width-right-150}" y="{top+22+18*idx}" font-family="Arial" font-size="12" fill="{color}">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts, qs = load_sim()
    te_raw, qe_raw = load_lvm_charge(EXP_TXT)
    te, qe0, pick_info = pick_best_cycle(te_raw, qe_raw, ts, qs)
    qs0 = detrend_to_zero(qs)

    x, qe_i, qs_i = resample_to_common(te, qe0, ts, qs0)
    # Align sign convention to maximize positive correlation.
    corr_pos = float(np.corrcoef(qe_i, qs_i)[0, 1])
    corr_neg = float(np.corrcoef(qe_i, -qs_i)[0, 1])
    if corr_neg > corr_pos:
        qs_i = -qs_i
        qs0 = -qs0
    amp_exp = float(np.max(qe_i) - np.min(qe_i))
    amp_sim = float(np.max(qs_i) - np.min(qs_i))
    qs_scaled = qs_i * (amp_exp / amp_sim) if amp_sim else qs_i
    rmse = float(np.sqrt(np.mean((qe_i - qs_i) ** 2)))
    nrmse = rmse / amp_exp * 100.0 if amp_exp else float("nan")
    rmse_scaled = float(np.sqrt(np.mean((qe_i - qs_scaled) ** 2)))
    nrmse_scaled = rmse_scaled / amp_exp * 100.0 if amp_exp else float("nan")
    corr = float(np.corrcoef(qe_i, qs_i)[0, 1])

    with (OUTDIR / "n4_charge_comparison_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["experiment_points_raw", len(te_raw)])
        writer.writerow(["experiment_points_used", len(te)])
        for key, value in pick_info.items():
            writer.writerow([key, f"{value:.12e}"])
        writer.writerow(["experiment_duration_used_s", f"{te[-1] - te[0]:.12e}"])
        writer.writerow(["simulation_duration_s", f"{ts[-1] - ts[0]:.12e}"])
        writer.writerow(["experiment_charge_pp_nC", f"{amp_exp:.12e}"])
        writer.writerow(["simulation_charge_pp_nC", f"{amp_sim:.12e}"])
        writer.writerow(["charge_pp_error_percent", f"{(amp_sim - amp_exp) / amp_exp * 100.0:.12e}"])
        writer.writerow(["rmse_nC", f"{rmse:.12e}"])
        writer.writerow(["nrmse_percent_of_exp_pp", f"{nrmse:.12e}"])
        writer.writerow(["scaled_simulation_rmse_nC", f"{rmse_scaled:.12e}"])
        writer.writerow(["scaled_simulation_nrmse_percent_of_exp_pp", f"{nrmse_scaled:.12e}"])
        writer.writerow(["correlation", f"{corr:.12e}"])

    with (OUTDIR / "n4_charge_comparison_resampled.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["normalized_time", "experiment_charge_nC", "simulation_charge_nC", "difference_nC"])
        for row in zip(x, qe_i, qs_i, qs_i - qe_i):
            writer.writerow([f"{float(v):.12e}" for v in row])

    svg_plot(
        OUTDIR / "n4_charge_experiment_vs_sim_normalized.svg",
        "N=4 transferred charge: experiment vs simulation",
        "Normalized time",
        "Transferred charge (nC)",
        [(x, qe_i, "Experiment", "#c5221f"), (x, qs_i, "Simulation", "#2f6f9f")],
    )
    svg_plot(
        OUTDIR / "n4_charge_experiment_vs_sim_scaled.svg",
        "N=4 transferred charge: experiment vs amplitude-scaled simulation",
        "Normalized time",
        "Transferred charge (nC)",
        [(x, qe_i, "Experiment", "#c5221f"), (x, qs_scaled, "Simulation scaled", "#34a853")],
    )
    svg_plot(
        OUTDIR / "n4_charge_experiment_raw_cycle.svg",
        "N=4 measured transferred charge cycle",
        "Time (s)",
        "Transferred charge (nC)",
        [(te, qe0, "Experiment", "#c5221f")],
    )
    print(OUTDIR)
    print(f"experiment Qpp nC: {amp_exp:.3f}")
    print(f"simulation Qpp nC: {amp_sim:.3f}")
    print(f"NRMSE %: {nrmse:.2f}")
    print(f"scaled NRMSE %: {nrmse_scaled:.2f}")
    print(f"correlation: {corr:.4f}")


if __name__ == "__main__":
    main()
