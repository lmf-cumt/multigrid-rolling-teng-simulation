from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Params:
    electrode_width: float = 16e-3
    electrode_gap: float = 5e-3
    pcb_thickness: float = 1e-3
    rows: int = 16
    cols: int = 9
    stroke: float = 100e-3
    acceleration: float = 5.0
    dwell_start: float = 30e-3
    dwell_stop: float = 30e-3
    transferred_charge_total: float = 380e-9
    optimum_load: float = 1e9
    internal_capacitance: float = 150e-12
    samples: int = 5000

    @property
    def parallel_units(self) -> int:
        return self.rows * self.cols

    @property
    def unit_transferred_charge(self) -> float:
        return self.transferred_charge_total / self.parallel_units

    @property
    def pitch(self) -> float:
        return 2 * self.electrode_width + self.electrode_gap


def build_motion(p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_acc = math.sqrt(p.stroke / p.acceleration)
    t_motion = 2 * t_acc
    duration = p.dwell_start + t_motion + p.dwell_stop
    t = np.linspace(0.0, duration, p.samples)
    x = np.zeros_like(t)
    v = np.zeros_like(t)

    t0 = p.dwell_start
    t1 = t0 + t_acc
    t2 = t0 + t_motion
    v_peak = p.acceleration * t_acc
    half = 0.5 * p.stroke

    acc_mask = (t >= t0) & (t < t1)
    tau = t[acc_mask] - t0
    x[acc_mask] = 0.5 * p.acceleration * tau**2
    v[acc_mask] = p.acceleration * tau

    dec_mask = (t >= t1) & (t < t2)
    tau = t[dec_mask] - t1
    x[dec_mask] = half + v_peak * tau - 0.5 * p.acceleration * tau**2
    v[dec_mask] = v_peak - p.acceleration * tau

    stop_mask = t >= t2
    x[stop_mask] = p.stroke
    v[stop_mask] = 0.0
    return t, x, v


def source_charge(x: np.ndarray, p: Params) -> np.ndarray:
    # Smooth single-stroke transfer model. The scale is calibrated so that the
    # whole 100 mm stroke transfers the 6514 electrometer value, 380 nC.
    s = np.clip(x / p.stroke, 0.0, 1.0)
    return 0.5 * p.transferred_charge_total * (-np.cos(math.pi * s))


def solve_load_response(
    t: np.ndarray,
    qsrc: np.ndarray,
    resistance: float,
    p: Params,
    internal_capacitance: float | None = None,
) -> dict[str, np.ndarray | float]:
    isrc = np.gradient(qsrc, t)
    c_internal = internal_capacitance if internal_capacitance is not None else p.internal_capacitance

    v = np.zeros_like(t)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        alpha = math.exp(-dt / (resistance * c_internal))
        # Exact update for constant source current over the step.
        i_mid = 0.5 * (isrc[i] + isrc[i - 1])
        v[i] = v[i - 1] * alpha + resistance * i_mid * (1.0 - alpha)

    iload = v / resistance
    power = v * iload
    energy = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(t))])
    qload = np.concatenate([[0.0], np.cumsum(0.5 * (iload[1:] + iload[:-1]) * np.diff(t))])

    return {
        "resistance_ohm": resistance,
        "capacitance_F": c_internal,
        "source_current_A": isrc,
        "load_current_A": iload,
        "voltage_V": v,
        "power_W": power,
        "energy_J": energy,
        "load_charge_C": qload,
    }


def parse_resistance_label(label: str) -> float:
    label = label.strip().upper().replace("Ω", "").replace("OHM", "")
    multipliers = {"K": 1e3, "M": 1e6, "G": 1e9}
    if label[-1:] in multipliers:
        return float(label[:-1]) * multipliers[label[-1]]
    return float(label)


def default_loads() -> list[float]:
    labels = [
        "10K",
        "20K",
        "50K",
        "100K",
        "200K",
        "500K",
        "1M",
        "2M",
        "5M",
        "10M",
        "20M",
        "50M",
        "100M",
        "200M",
        "500M",
        "1G",
        "2G",
        "5G",
        "10G",
    ]
    return [parse_resistance_label(x) for x in labels]


def load_experimental_targets(path: Path) -> dict[float, dict[str, float]]:
    if not path.exists():
        return {}
    targets: dict[float, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                r = float(row["电阻值(Ω)"])
                i_peak = float(row["峰值电流(μA)"])
                p_mean = float(row["平均功率(mW)"])
            except (KeyError, ValueError):
                continue
            targets[r] = {"peak_current_uA": i_peak, "mean_power_mW": p_mean}
    return targets


def fit_internal_capacitance(t: np.ndarray, qsrc: np.ndarray, p: Params, targets: dict[float, dict[str, float]]) -> float:
    if not targets:
        return p.internal_capacitance
    candidate_caps = np.logspace(-12, -9, 121)
    best_cap = p.internal_capacitance
    best_error = float("inf")
    for cap in candidate_caps:
        err = 0.0
        count = 0
        for r, target in targets.items():
            measured_peak_uA = target["peak_current_uA"]
            response = solve_load_response(t, qsrc, r, p, cap)
            simulated_peak_uA = float(np.max(np.abs(response["load_current_A"])) * 1e6)
            if measured_peak_uA > 0 and simulated_peak_uA > 0:
                err += (math.log(simulated_peak_uA) - math.log(measured_peak_uA)) ** 2
                count += 1
        if count and err / count < best_error:
            best_error = err / count
            best_cap = float(cap)
    return best_cap


def fmt_resistance(r: float) -> str:
    if r >= 1e9:
        return f"{r / 1e9:g}G"
    if r >= 1e6:
        return f"{r / 1e6:g}M"
    if r >= 1e3:
        return f"{r / 1e3:g}K"
    return f"{r:g}"


def write_timeseries(path: Path, t: np.ndarray, x: np.ndarray, v_motion: np.ndarray, qsrc: np.ndarray, response: dict[str, np.ndarray | float]) -> None:
    fields = [
        "time_s",
        "position_m",
        "motion_velocity_m_per_s",
        "source_charge_C",
        "load_charge_C",
        "source_current_A",
        "load_current_A",
        "voltage_V",
        "power_W",
        "energy_J",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for values in zip(
            t,
            x,
            v_motion,
            qsrc,
            response["load_charge_C"],
            response["source_current_A"],
            response["load_current_A"],
            response["voltage_V"],
            response["power_W"],
            response["energy_J"],
        ):
            writer.writerow([f"{float(v):.12e}" for v in values])


def write_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = [
        "load",
        "resistance_ohm",
        "experimental_peak_current_uA",
        "load_current_peak_uA",
        "peak_current_error_percent",
        "load_current_mean_abs_uA",
        "load_current_rms_uA",
        "voltage_peak_V",
        "voltage_rms_V",
        "experimental_mean_power_mW",
        "power_peak_mW",
        "power_mean_mW",
        "mean_power_error_percent",
        "energy_uJ",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def svg_plot(path: Path, title: str, xlabel: str, ylabel: str, x: np.ndarray, series: list[tuple[str, np.ndarray, str]]) -> None:
    width, height = 780, 430
    left, right, top, bottom = 82, 24, 28, 58
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin = min(float(np.min(y)) for _, y, _ in series)
    ymax = max(float(np.max(y)) for _, y, _ in series)
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    def pts(y: np.ndarray) -> str:
        px = left + (x - xmin) / (xmax - xmin) * (width - left - right)
        py = top + (ymax - y) / (ymax - ymin) * (height - top - bottom)
        return " ".join(f"{float(a):.2f},{float(b):.2f}" for a, b in zip(px, py))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="20" text-anchor="middle" font-family="Arial" font-size="14">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-14}" text-anchor="middle" font-family="Arial" font-size="12">{xlabel}</text>',
        f'<text x="16" y="{height/2}" text-anchor="middle" transform="rotate(-90 16 {height/2})" font-family="Arial" font-size="12">{ylabel}</text>',
    ]
    for frac in np.linspace(0, 1, 5):
        yy = top + frac * (height - top - bottom)
        value = ymax - frac * (ymax - ymin)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{value:.2e}</text>')
    for i, (name, y, color) in enumerate(series):
        lines.append(f'<polyline points="{pts(y)}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        lines.append(f'<text x="{width-right-120}" y="{top+20+18*i}" font-family="Arial" font-size="12" fill="{color}">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_loglog(path: Path, title: str, ylabel: str, loads: np.ndarray, y: np.ndarray) -> None:
    x = np.log10(loads)
    svg_plot(path, title, "log10(R / ohm)", ylabel, x, [(ylabel, y, "#2f6f9f")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results_p2_calibrated")
    parser.add_argument("--loads", default="", help="Comma-separated load labels, e.g. 100M,1G,2G")
    parser.add_argument("--experimental-summary", default=r"C:\Users\lmf\Desktop\P-2-TENG\P-2-TENG_TENG_分析结果.csv")
    args = parser.parse_args()

    p = Params()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    loads = [parse_resistance_label(x) for x in args.loads.split(",") if x.strip()] if args.loads else default_loads()

    t, x, v_motion = build_motion(p)
    qsrc = source_charge(x, p)
    targets = load_experimental_targets(Path(args.experimental_summary))
    fitted_cap = fit_internal_capacitance(t, qsrc, p, targets)
    summary = []
    selected = {}

    for r in loads:
        response = solve_load_response(t, qsrc, r, p, fitted_cap)
        label = fmt_resistance(r)
        write_timeseries(outdir / f"p2_teng_{label}_timeseries.csv", t, x, v_motion, qsrc, response)
        iload = response["load_current_A"]
        voltage = response["voltage_V"]
        power = response["power_W"]
        energy = response["energy_J"]
        target = targets.get(r, {})
        sim_peak_uA = float(np.max(np.abs(iload))) * 1e6
        sim_mean_power_mW = float(np.mean(power)) * 1e3
        exp_peak_uA = target.get("peak_current_uA", float("nan"))
        exp_mean_power_mW = target.get("mean_power_mW", float("nan"))
        peak_error = (sim_peak_uA - exp_peak_uA) / exp_peak_uA * 100.0 if exp_peak_uA > 0 else float("nan")
        power_error = (
            (sim_mean_power_mW - exp_mean_power_mW) / exp_mean_power_mW * 100.0
            if exp_mean_power_mW > 0
            else float("nan")
        )
        summary.append(
            {
                "load": label,
                "resistance_ohm": f"{r:.12e}",
                "experimental_peak_current_uA": f"{exp_peak_uA:.12e}",
                "load_current_peak_uA": f"{sim_peak_uA:.12e}",
                "peak_current_error_percent": f"{peak_error:.12e}",
                "load_current_mean_abs_uA": f"{float(np.mean(np.abs(iload))) * 1e6:.12e}",
                "load_current_rms_uA": f"{float(np.sqrt(np.mean(iload**2))) * 1e6:.12e}",
                "voltage_peak_V": f"{float(np.max(np.abs(voltage))):.12e}",
                "voltage_rms_V": f"{float(np.sqrt(np.mean(voltage**2))):.12e}",
                "experimental_mean_power_mW": f"{exp_mean_power_mW:.12e}",
                "power_peak_mW": f"{float(np.max(power)) * 1e3:.12e}",
                "power_mean_mW": f"{sim_mean_power_mW:.12e}",
                "mean_power_error_percent": f"{power_error:.12e}",
                "energy_uJ": f"{float(energy[-1]) * 1e6:.12e}",
            }
        )
        if label in {"100M", "500M", "1G", "2G"}:
            selected[label] = response

    write_summary(outdir / "p2_teng_calibrated_summary.csv", summary)
    meta = {
        "structure": "16 x 9 parallel dual-electrode rolling-ball TENG array",
        "parallel_units": p.parallel_units,
        "pcb_thickness_m": p.pcb_thickness,
        "single_unit_electrode_width_m": p.electrode_width,
        "single_unit_electrode_gap_m": p.electrode_gap,
        "stroke_m": p.stroke,
        "acceleration_m_per_s2": p.acceleration,
        "dwell_start_s": p.dwell_start,
        "dwell_stop_s": p.dwell_stop,
        "measured_total_transferred_charge_C": p.transferred_charge_total,
        "equivalent_unit_transferred_charge_C": p.unit_transferred_charge,
        "optimum_load_reference_ohm": p.optimum_load,
        "fitted_internal_capacitance_F": fitted_cap,
        "experimental_summary_used": str(Path(args.experimental_summary)),
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    loads_arr = np.array([float(row["resistance_ohm"]) for row in summary])
    i_peak = np.array([float(row["load_current_peak_uA"]) for row in summary])
    p_mean = np.array([float(row["power_mean_mW"]) for row in summary])
    e_uJ = np.array([float(row["energy_uJ"]) for row in summary])
    svg_loglog(outdir / "current_peak_vs_resistance.svg", "Peak current vs load", "Peak current (uA)", loads_arr, i_peak)
    svg_loglog(outdir / "mean_power_vs_resistance.svg", "Mean power vs load", "Mean power (mW)", loads_arr, p_mean)
    svg_loglog(outdir / "energy_vs_resistance.svg", "Energy per stroke vs load", "Energy (uJ)", loads_arr, e_uJ)

    if selected:
        svg_plot(
            outdir / "selected_current_waveforms.svg",
            "Selected load current waveforms",
            "Time (s)",
            "Current (uA)",
            t,
            [(name, resp["load_current_A"] * 1e6, color) for (name, resp), color in zip(selected.items(), ["#2f6f9f", "#34a853", "#b26a00", "#c5221f"])],
        )
        svg_plot(
            outdir / "selected_power_waveforms.svg",
            "Selected load power waveforms",
            "Time (s)",
            "Power (mW)",
            t,
            [(name, resp["power_W"] * 1e3, color) for (name, resp), color in zip(selected.items(), ["#2f6f9f", "#34a853", "#b26a00", "#c5221f"])],
        )


if __name__ == "__main__":
    main()
