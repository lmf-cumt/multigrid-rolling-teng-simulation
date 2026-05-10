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
    acceleration: float = 4.0
    dwell_start: float = 30e-3
    dwell_stop: float = 30e-3
    transferred_charge_total_n2: float = 380e-9
    transferred_charge_total_n4: float = 650e-9
    internal_capacitance_n2: float = 112.20184543019652e-12
    short_circuit_current_peak: float = 8e-6
    charge_response_tau: float = 0.0
    samples: int = 12000

    @property
    def parallel_units(self) -> int:
        return self.rows * self.cols

    @property
    def unit_transferred_charge_n2(self) -> float:
        return self.transferred_charge_total_n2 / self.parallel_units


def default_loads() -> list[float]:
    return [1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9, 2e9, 5e9, 1e10]


def fmt_resistance(r: float) -> str:
    if r >= 1e9:
        return f"{r / 1e9:g}G"
    if r >= 1e6:
        return f"{r / 1e6:g}M"
    if r >= 1e3:
        return f"{r / 1e3:g}K"
    return f"{r:g}"


def build_motion(p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_acc = math.sqrt(p.stroke / p.acceleration)
    t_motion = 2.0 * t_acc
    duration = p.dwell_start + t_motion + p.dwell_stop + t_motion + p.dwell_start
    t = np.linspace(0.0, duration, p.samples)
    x = np.zeros_like(t)
    v = np.zeros_like(t)

    t0 = p.dwell_start
    t1 = t0 + t_acc
    t2 = t0 + t_motion
    t3 = t2 + p.dwell_stop
    t4 = t3 + t_acc
    t5 = t3 + t_motion
    v_peak = p.acceleration * t_acc

    acc = (t >= t0) & (t < t1)
    tau = t[acc] - t0
    x[acc] = 0.5 * p.acceleration * tau**2
    v[acc] = p.acceleration * tau

    dec = (t >= t1) & (t < t2)
    tau = t[dec] - t1
    x[dec] = 0.5 * p.stroke + v_peak * tau - 0.5 * p.acceleration * tau**2
    v[dec] = v_peak - p.acceleration * tau

    stop = t >= t2
    x[stop] = p.stroke

    ret_acc = (t >= t3) & (t < t4)
    tau = t[ret_acc] - t3
    x[ret_acc] = p.stroke - 0.5 * p.acceleration * tau**2
    v[ret_acc] = -p.acceleration * tau

    ret_dec = (t >= t4) & (t < t5)
    tau = t[ret_dec] - t4
    x[ret_dec] = 0.5 * p.stroke - v_peak * tau + 0.5 * p.acceleration * tau**2
    v[ret_dec] = -v_peak + p.acceleration * tau

    home = t >= t5
    x[home] = 0.0
    v[home] = 0.0
    return t, x, v


def structure_length(electrode_count: int, p: Params) -> float:
    return electrode_count * p.electrode_width + (electrode_count - 1) * p.electrode_gap


def source_waveforms(t: np.ndarray, electrode_count: int, p: Params) -> tuple[np.ndarray, np.ndarray]:
    # Multi-transfer model:
    # The PTFE surface charge density is fixed. More alternating electrodes
    # make the ball group undergo more effective transfer events during the
    # same stroke. N=4 measured charge gives the non-ideal transfer efficiency.
    event_count = transfer_event_count(electrode_count)
    q_total = transfer_charge_total(electrode_count, p)
    isrc = pulse_train_current(t, event_count, q_total, p)
    if p.charge_response_tau > 0:
        isrc = lowpass_current(t, isrc, p.charge_response_tau)
    qsrc = cumulative_trapezoid(t, isrc)
    qpp = float(np.max(qsrc) - np.min(qsrc))
    if qpp > 0:
        isrc = isrc * (q_total / qpp)
        qsrc = cumulative_trapezoid(t, isrc)
    return qsrc, isrc


def cumulative_trapezoid(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])


def lowpass_current(t: np.ndarray, current: np.ndarray, tau: float) -> np.ndarray:
    filtered = np.zeros_like(current)
    filtered[0] = current[0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        alpha = math.exp(-dt / tau)
        filtered[i] = alpha * filtered[i - 1] + (1.0 - alpha) * current[i]
    return filtered


def pulse_train_current(t: np.ndarray, event_count: int, q_per_stroke: float, p: Params) -> np.ndarray:
    t_acc = math.sqrt(p.stroke / p.acceleration)
    t_motion = 2.0 * t_acc
    forward_start = p.dwell_start
    forward_end = forward_start + t_motion
    reverse_start = forward_end + p.dwell_stop
    reverse_end = reverse_start + t_motion
    centers_forward = forward_start + (np.arange(event_count) + 0.5) / event_count * t_motion
    centers_reverse = reverse_start + (np.arange(event_count) + 0.5) / event_count * t_motion

    weights = pulse_weights(event_count, p)
    # Use the N=2 pulse width as the reference. The largest pulse is calibrated
    # to the measured short-circuit peak; later pulses are smaller, not equal.
    tau = fit_single_pulse_width(t, centers_forward[0], p.transferred_charge_total_n2, p.short_circuit_current_peak)
    forward = weighted_raw_pulses(t, centers_forward, weights, tau)
    reverse = weighted_raw_pulses(t, centers_reverse, weights, tau)
    area_per_unit = np.trapezoid(raw_pulses(t, np.array([centers_forward[0]]), tau), t)
    scale = p.short_circuit_current_peak
    # By construction, peak = scale and charge = scale * area_per_unit * sum(weights).
    isrc = scale * (forward - reverse)

    # Suppress numerical tails inside dwell intervals.
    isrc[(t < forward_start) | ((t > forward_end) & (t < reverse_start)) | (t > reverse_end)] *= 0.02
    return isrc


def raw_pulses(t: np.ndarray, centers: np.ndarray, tau: float) -> np.ndarray:
    z = (t[:, None] - centers[None, :]) / tau
    return np.sum(1.0 / np.cosh(np.clip(z, -30.0, 30.0)) ** 2, axis=1)


def weighted_raw_pulses(t: np.ndarray, centers: np.ndarray, weights: np.ndarray, tau: float) -> np.ndarray:
    z = (t[:, None] - centers[None, :]) / tau
    return np.sum(weights[None, :] / np.cosh(np.clip(z, -30.0, 30.0)) ** 2, axis=1)


def pulse_weights(event_count: int, p: Params) -> np.ndarray:
    if event_count <= 1:
        return np.array([1.0])
    # Four-electrode measurement: Q4/Q2 = 1 + r, so the second pulse is lower
    # than the first. Extend the same decay trend for the third pulse in N=6.
    r = p.transferred_charge_total_n4 / p.transferred_charge_total_n2 - 1.0
    return np.array([r**k for k in range(event_count)], dtype=float)


def fit_single_pulse_width(t: np.ndarray, center: float, q_single: float, target_peak: float) -> float:
    lo = 1e-4
    hi = max(0.5, float(t[-1] - t[0]))
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        shape = raw_pulses(t, np.array([center]), mid)
        area = np.trapezoid(shape, t)
        peak = q_single / area * float(np.max(shape)) if area > 0 else float("inf")
        if peak > target_peak:
            lo = mid
        else:
            hi = mid
    return hi


def transfer_event_count(electrode_count: int) -> int:
    return max(1, electrode_count // 2)


def transfer_efficiency(p: Params) -> float:
    return p.transferred_charge_total_n4 / (2.0 * p.transferred_charge_total_n2)


def charge_per_event(electrode_count: int, p: Params) -> float:
    if electrode_count == 2:
        return p.transferred_charge_total_n2
    return transfer_efficiency(p) * p.transferred_charge_total_n2


def transfer_charge_total(electrode_count: int, p: Params) -> float:
    if electrode_count == 2:
        return p.transferred_charge_total_n2
    weights = pulse_weights(transfer_event_count(electrode_count), p)
    return p.transferred_charge_total_n2 * float(np.sum(weights))


def internal_capacitance(electrode_count: int, p: Params) -> float:
    # Patent-comparison mode: keep the experimentally calibrated array-level
    # capacitance constant and isolate the effect of additional alternating
    # electrode boundaries.
    return p.internal_capacitance_n2


def solve_load_response(t: np.ndarray, isrc: np.ndarray, resistance: float, capacitance: float) -> dict[str, np.ndarray]:
    v = np.zeros_like(t)
    tau_rc = resistance * capacitance
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        alpha = math.exp(-dt / tau_rc) if tau_rc > 0 else 0.0
        i_mid = 0.5 * (isrc[i] + isrc[i - 1])
        v[i] = v[i - 1] * alpha + resistance * i_mid * (1.0 - alpha)
    iload = v / resistance
    power = v * iload
    energy = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(t))])
    qload = cumulative_trapezoid(t, iload)
    return {
        "source_current_A": isrc,
        "load_current_A": iload,
        "voltage_V": v,
        "power_W": power,
        "energy_J": energy,
        "load_charge_C": qload,
    }


def summarize(electrode_count: int, load: float, t: np.ndarray, qsrc: np.ndarray, response: dict[str, np.ndarray]) -> dict[str, str]:
    isrc = response["source_current_A"]
    iload = response["load_current_A"]
    voltage = response["voltage_V"]
    power = response["power_W"]
    energy = response["energy_J"]
    qpp = float(np.max(qsrc) - np.min(qsrc))
    return {
        "electrode_count": str(electrode_count),
        "load": fmt_resistance(load),
        "resistance_ohm": f"{load:.12e}",
        "event_count": str(transfer_event_count(electrode_count)),
        "source_charge_peak_to_peak_nC": f"{qpp * 1e9:.12e}",
        "modeled_total_transfer_charge_nC": f"{transfer_charge_total(electrode_count, Params()) * 1e9:.12e}",
        "source_current_peak_uA": f"{np.max(np.abs(isrc)) * 1e6:.12e}",
        "load_current_peak_uA": f"{np.max(np.abs(iload)) * 1e6:.12e}",
        "load_current_rms_uA": f"{np.sqrt(np.mean(iload**2)) * 1e6:.12e}",
        "voltage_peak_V": f"{np.max(np.abs(voltage)):.12e}",
        "voltage_rms_V": f"{np.sqrt(np.mean(voltage**2)):.12e}",
        "power_peak_mW": f"{np.max(power) * 1e3:.12e}",
        "power_mean_mW": f"{np.mean(power) * 1e3:.12e}",
        "energy_uJ": f"{energy[-1] * 1e6:.12e}",
        "duration_s": f"{(t[-1] - t[0]):.12e}",
    }


def write_timeseries(path: Path, t: np.ndarray, x: np.ndarray, velocity: np.ndarray, qsrc: np.ndarray, response: dict[str, np.ndarray]) -> None:
    fields = [
        "time_s",
        "position_m",
        "motion_velocity_m_per_s",
        "source_charge_C",
        "source_current_A",
        "load_charge_C",
        "load_current_A",
        "voltage_V",
        "power_W",
        "energy_J",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for row in zip(
            t,
            x,
            velocity,
            qsrc,
            response["source_current_A"],
            response["load_charge_C"],
            response["load_current_A"],
            response["voltage_V"],
            response["power_W"],
            response["energy_J"],
        ):
            writer.writerow([f"{float(v):.12e}" for v in row])


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def svg_plot(path: Path, title: str, xlabel: str, ylabel: str, x: np.ndarray, series: list[tuple[str, np.ndarray, str]]) -> None:
    width, height = 800, 430
    left, right, top, bottom = 86, 24, 30, 58
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


def make_plots(outdir: Path, t: np.ndarray, selected: dict[int, dict[str, np.ndarray]], summary_rows: list[dict[str, str]]) -> None:
    colors = {2: "#2f6f9f", 4: "#34a853", 6: "#b26a00"}
    for key, ylabel, filename, scale in [
        ("load_current_A", "Current (uA)", "current_comparison_1G.svg", 1e6),
        ("voltage_V", "Voltage (V)", "voltage_comparison_1G.svg", 1.0),
        ("power_W", "Power (mW)", "power_comparison_1G.svg", 1e3),
        ("energy_J", "Energy (uJ)", "energy_comparison_1G.svg", 1e6),
    ]:
        svg_plot(
            outdir / filename,
            f"{ylabel} at 1 Gohm",
            "Time (s)",
            ylabel,
            t,
            [(f"N={n}", resp[key] * scale, colors[n]) for n, resp in selected.items()],
        )

    rows_1g = [r for r in summary_rows if r["load"] == "1G"]
    x = np.array([int(r["electrode_count"]) for r in rows_1g], dtype=float)
    for metric, ylabel, filename in [
        ("load_current_peak_uA", "Peak current (uA)", "key_peak_current.svg"),
        ("power_mean_mW", "Mean power (mW)", "key_mean_power.svg"),
        ("energy_uJ", "Energy (uJ)", "key_energy.svg"),
    ]:
        y = np.array([float(r[metric]) for r in rows_1g])
        svg_plot(outdir / filename, ylabel + " at 1 Gohm", "Electrode count", ylabel, x, [(metric, y, "#2f6f9f")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results_multigrid_calibrated")
    parser.add_argument("--load-for-waveforms", default="1G")
    parser.add_argument("--charge-response-tau", type=float, default=0.0)
    args = parser.parse_args()

    p = Params(charge_response_tau=args.charge_response_tau)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t, x, velocity = build_motion(p)
    loads = default_loads()
    waveform_load = {"100M": 1e8, "500M": 5e8, "1G": 1e9, "2G": 2e9}.get(args.load_for_waveforms.upper(), 1e9)

    summary_rows: list[dict[str, str]] = []
    selected: dict[int, dict[str, np.ndarray]] = {}
    selected_qsrc: dict[int, np.ndarray] = {}
    for n in (2, 4, 6):
        qsrc, isrc = source_waveforms(t, n, p)
        cap = internal_capacitance(n, p)
        for load in loads:
            response = solve_load_response(t, isrc, load, cap)
            summary_rows.append(summarize(n, load, t, qsrc, response))
            if math.isclose(load, waveform_load):
                selected[n] = response
                selected_qsrc[n] = qsrc
                write_timeseries(outdir / f"teng_N{n}_{fmt_resistance(load)}_timeseries.csv", t, x, velocity, qsrc, response)

    write_csv(outdir / "multigrid_calibrated_summary.csv", summary_rows)
    rows_1g = [r for r in summary_rows if r["load"] == fmt_resistance(waveform_load)]
    base = next(r for r in rows_1g if r["electrode_count"] == "2")
    comparison = []
    for r in rows_1g:
        row = dict(r)
        row["peak_current_ratio_vs_N2"] = f"{float(r['load_current_peak_uA']) / float(base['load_current_peak_uA']):.12e}"
        row["mean_power_ratio_vs_N2"] = f"{float(r['power_mean_mW']) / float(base['power_mean_mW']):.12e}"
        row["energy_ratio_vs_N2"] = f"{float(r['energy_uJ']) / float(base['energy_uJ']):.12e}"
        comparison.append(row)
    write_csv(outdir / f"key_comparison_{fmt_resistance(waveform_load)}.csv", comparison)
    make_plots(outdir, t, selected, summary_rows)

    metadata = {
        "model": "experiment-calibrated equivalent circuit/kinematic simulation",
        "cycle_definition": "full reciprocating cycle: start dwell, forward acceleration/deceleration stroke, end dwell, reverse acceleration/deceleration stroke, home dwell",
        "electrode_counts": [2, 4, 6],
        "parallel_units": p.parallel_units,
        "pcb_thickness_m": p.pcb_thickness,
        "stroke_m": p.stroke,
        "acceleration_m_per_s2": p.acceleration,
        "dwell_start_s": p.dwell_start,
        "dwell_stop_s": p.dwell_stop,
        "N2_measured_total_transferred_charge_C": p.transferred_charge_total_n2,
        "N4_measured_total_transferred_charge_C": p.transferred_charge_total_n4,
        "fixed_PTFE_charge_assumption": "same PTFE ball surface charge density for N=2, N=4, and N=6; transfer charge changes because the electrode pattern creates more transfer events",
        "multi_transfer_model": "N=2 has 1 transfer event, N=4 has 2 non-equal events; only the highest short-circuit current peak is constrained to match the test. The N=4 charge gives the second-pulse weight r=Q4/Q2-1; N=6 extends the same decay trend.",
        "nonideal_transfer_efficiency_eta": transfer_efficiency(p),
        "N2_total_transfer_charge_C": transfer_charge_total(2, p),
        "N4_total_transfer_charge_C": transfer_charge_total(4, p),
        "predicted_N6_total_transfer_charge_C": transfer_charge_total(6, p),
        "pulse_weights_N2": pulse_weights(1, p).tolist(),
        "pulse_weights_N4": pulse_weights(2, p).tolist(),
        "pulse_weights_N6": pulse_weights(3, p).tolist(),
        "N2_unit_transferred_charge_C": p.unit_transferred_charge_n2,
        "N2_fitted_internal_capacitance_F": p.internal_capacitance_n2,
        "short_circuit_current_peak_A": p.short_circuit_current_peak,
        "capacitance_scaling": "array-level internal capacitance kept equal to the N=2 calibrated value to isolate electrode-boundary effect",
        "source_charge_scaling": "PTFE charge density is fixed; total transfer charge increases with the number of effective transfer events and is corrected by eta from the N=4 measurement",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
