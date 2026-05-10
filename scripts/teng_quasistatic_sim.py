from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS0 = 8.8541878128e-12


@dataclass(frozen=True)
class Params:
    electrode_width: float = 16e-3
    gap: float = 5e-3
    ball_radius: float = 2e-3
    ball_count: int = 4
    surface_charge_density: float = -50e-6
    speed: float = 0.1
    load_resistance: float = 100e6
    active_depth: float = 16e-3
    ball_height: float = 2.2e-3
    coupling_sigma: float = 2.5e-3
    samples_per_electrode: int = 100


def electrode_intervals(n: int, p: Params) -> list[tuple[float, float, int]]:
    intervals = []
    x = 0.0
    for i in range(n):
        intervals.append((x, x + p.electrode_width, 1 if i % 2 == 0 else -1))
        x += p.electrode_width + p.gap
    return intervals


def gaussian_interval_integral(x0: np.ndarray, a: float, b: float, sigma: float) -> np.ndarray:
    z1 = (b - x0) / (math.sqrt(2.0) * sigma)
    z0 = (a - x0) / (math.sqrt(2.0) * sigma)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (erf_vec(z1) - erf_vec(z0))


def simulate_structure(n: int, p: Params) -> dict[str, np.ndarray | float | int]:
    intervals = electrode_intervals(n, p)
    total_length = n * p.electrode_width + (n - 1) * p.gap
    scan_margin = p.electrode_width
    x_start = -scan_margin
    x_end = total_length + scan_margin
    samples = max(301, n * p.samples_per_electrode)
    x_ref = np.linspace(x_start, x_end, samples)
    time = (x_ref - x_ref[0]) / p.speed

    offsets = (np.arange(p.ball_count) - (p.ball_count - 1) / 2.0) * (2.0 * p.ball_radius)
    ball_area = math.pi * p.ball_radius**2
    q_scale = p.surface_charge_density * ball_area * p.ball_count

    signed_coupling = np.zeros_like(x_ref)
    abs_coupling = np.zeros_like(x_ref)
    for off in offsets:
        centers = x_ref + off
        for a, b, sign in intervals:
            c = gaussian_interval_integral(centers, a, b, p.coupling_sigma)
            signed_coupling += sign * c
            abs_coupling += c

    # Normalize by ball count so Q is bounded by the projected charged area.
    signed_coupling /= p.ball_count
    abs_coupling /= p.ball_count

    charge = q_scale * signed_coupling
    current = np.gradient(charge, time)
    voltage = current * p.load_resistance
    power = voltage * current
    # Numerical noise can make tiny negative values; a resistor dissipates non-negative power.
    power = np.maximum(power, 0.0)
    energy = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(time))])

    # Auxiliary open-circuit estimate for sanity checking only.
    eff_gap = p.ball_height
    cap = EPS0 * p.active_depth * total_length / eff_gap
    voc = charge / cap

    return {
        "n": n,
        "total_length_m": total_length,
        "time_s": time,
        "x_ball_m": x_ref,
        "charge_C": charge,
        "current_A": current,
        "voltage_V": voltage,
        "power_W": power,
        "energy_J": energy,
        "voc_estimate_V": voc,
        "signed_coupling": signed_coupling,
        "abs_coupling": abs_coupling,
    }


def device_capacitance(n: int, p: Params) -> float:
    total_length = n * p.electrode_width + (n - 1) * p.gap
    return EPS0 * p.active_depth * total_length / p.ball_height


def apply_rc_load(
    result: dict[str, np.ndarray | float | int],
    p: Params,
    capacitance: float | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Apply Norton current-source plus parallel device capacitance RC load dynamics."""
    n = int(result["n"])
    cap = device_capacitance(n, p) if capacitance is None else capacitance
    time = result["time_s"]
    source_charge = result["charge_C"]
    source_current = np.gradient(source_charge, time)
    voltage = np.zeros_like(time)

    for i in range(1, len(time)):
        dt = float(time[i] - time[i - 1])
        i_mid = 0.5 * float(source_current[i] + source_current[i - 1])
        voltage[i] = (voltage[i - 1] + dt * i_mid / cap) / (1.0 + dt / (p.load_resistance * cap))

    load_current = voltage / p.load_resistance
    power = voltage * load_current
    energy = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(time))])
    load_charge = np.concatenate([[0.0], np.cumsum(0.5 * (load_current[1:] + load_current[:-1]) * np.diff(time))])

    updated = dict(result)
    updated.update(
        {
            "current_A": load_current,
            "voltage_V": voltage,
            "power_W": power,
            "energy_J": energy,
            "load_charge_C": load_charge,
            "source_charge_C": source_charge,
            "source_current_A": source_current,
            "device_capacitance_F": cap,
            "rc_time_constant_s": p.load_resistance * cap,
        }
    )
    return updated


def write_csv(path: Path, result: dict[str, np.ndarray | float | int]) -> None:
    fields = [
        "time_s",
        "x_ball_m",
        "charge_C",
        "current_A",
        "voltage_V",
        "power_W",
        "energy_J",
        "voc_estimate_V",
        "signed_coupling",
        "abs_coupling",
    ]
    optional_fields = ["load_charge_C", "source_charge_C", "source_current_A"]
    fields += [name for name in optional_fields if name in result]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        columns = [result[name] for name in fields]
        for row in zip(*columns):
            writer.writerow([f"{float(v):.12e}" for v in row])


def write_summary(path: Path, results: list[dict[str, np.ndarray | float | int]]) -> None:
    fields = [
        "electrode_count",
        "scan_length_m",
        "duration_s",
        "charge_peak_to_peak_C",
        "cumulative_transfer_charge_C",
        "load_cumulative_transfer_charge_C",
        "current_peak_abs_A",
        "voltage_peak_abs_V",
        "power_peak_W",
        "power_mean_W",
        "energy_final_J",
        "energy_per_meter_J_per_m",
        "pulse_count_estimate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            time = r["time_s"]
            charge = r["charge_C"]
            cumulative_transfer = float(np.sum(np.abs(np.diff(charge))))
            current = r["current_A"]
            voltage = r["voltage_V"]
            power = r["power_W"]
            energy = r["energy_J"]
            x = r["x_ball_m"]
            load_charge = r.get("load_charge_C", charge)
            load_transfer = float(np.sum(np.abs(np.diff(load_charge))))
            writer.writerow(
                {
                    "electrode_count": r["n"],
                    "scan_length_m": f"{float(x[-1] - x[0]):.12e}",
                    "duration_s": f"{float(time[-1] - time[0]):.12e}",
                    "charge_peak_to_peak_C": f"{float(charge.max() - charge.min()):.12e}",
                    "cumulative_transfer_charge_C": f"{cumulative_transfer:.12e}",
                    "load_cumulative_transfer_charge_C": f"{load_transfer:.12e}",
                    "current_peak_abs_A": f"{float(np.max(np.abs(current))):.12e}",
                    "voltage_peak_abs_V": f"{float(np.max(np.abs(voltage))):.12e}",
                    "power_peak_W": f"{float(power.max()):.12e}",
                    "power_mean_W": f"{float(power.mean()):.12e}",
                    "energy_final_J": f"{float(energy[-1]):.12e}",
                    "energy_per_meter_J_per_m": f"{float(energy[-1] / (x[-1] - x[0])):.12e}",
                    "pulse_count_estimate": 2 * (int(r["n"]) - 1),
                }
            )


def _svg_polyline(x: np.ndarray, y: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float) -> str:
    w, h = 760, 400
    left, right, top, bottom = 70, 20, 20, 55
    px = left + (x - xmin) / (xmax - xmin) * (w - left - right)
    if abs(ymax - ymin) < 1e-300:
        py = np.full_like(y, top + 0.5 * (h - top - bottom))
    else:
        py = top + (ymax - y) / (ymax - ymin) * (h - top - bottom)
    return " ".join(f"{float(a):.2f},{float(b):.2f}" for a, b in zip(px, py))


def _write_svg_plot(
    path: Path,
    title: str,
    ylabel: str,
    results: list[dict[str, np.ndarray | float | int]],
    key: str,
) -> None:
    colors = {2: "#2f6f9f", 4: "#34a853", 6: "#b26a00"}
    x_min = min(float(np.min(r["time_s"])) for r in results)
    x_max = max(float(np.max(r["time_s"])) for r in results)
    y_min = min(float(np.min(r[key])) for r in results)
    y_max = max(float(np.max(r[key])) for r in results)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    w, h = 760, 400
    left, right, top, bottom = 70, 20, 20, 55
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="18" text-anchor="middle" font-family="Arial" font-size="14">{title}</text>',
        f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-bottom}" stroke="#333"/>',
        f'<text x="{w/2}" y="{h-12}" text-anchor="middle" font-family="Arial" font-size="12">Time (s)</text>',
        f'<text x="16" y="{h/2}" text-anchor="middle" transform="rotate(-90 16 {h/2})" font-family="Arial" font-size="12">{ylabel}</text>',
    ]
    for frac in np.linspace(0, 1, 5):
        y = top + frac * (h - top - bottom)
        value = y_max - frac * (y_max - y_min)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{value:.2e}</text>')
    for frac in np.linspace(0, 1, 5):
        x = left + frac * (w - left - right)
        value = x_min + frac * (x_max - x_min)
        lines.append(f'<line x1="{x:.1f}" y1="{h-bottom}" x2="{x:.1f}" y2="{h-bottom+5}" stroke="#333"/>')
        lines.append(f'<text x="{x:.1f}" y="{h-bottom+20}" text-anchor="middle" font-family="Arial" font-size="10">{value:.2f}</text>')
    for idx, r in enumerate(results):
        n = int(r["n"])
        pts = _svg_polyline(r["time_s"], r[key], x_min, x_max, y_min, y_max)
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{colors[n]}" stroke-width="1.8"/>')
        lines.append(
            f'<text x="{w-right-80}" y="{top+22+18*idx}" font-family="Arial" font-size="12" fill="{colors[n]}">N={n}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_svg_bar(path: Path, results: list[dict[str, np.ndarray | float | int]]) -> None:
    w, h = 640, 380
    left, right, top, bottom = 70, 30, 30, 55
    colors = {2: "#2f6f9f", 4: "#34a853", 6: "#b26a00"}
    energies = [float(r["energy_J"][-1]) for r in results]
    max_e = max(energies) if energies else 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="20" text-anchor="middle" font-family="Arial" font-size="14">Final energy comparison</text>',
        f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-bottom}" stroke="#333"/>',
        f'<text x="16" y="{h/2}" text-anchor="middle" transform="rotate(-90 16 {h/2})" font-family="Arial" font-size="12">Energy (J)</text>',
    ]
    bar_w = 80
    span = (w - left - right) / len(results)
    for i, r in enumerate(results):
        n = int(r["n"])
        e = float(r["energy_J"][-1])
        bh = 0 if max_e <= 0 else e / max_e * (h - top - bottom - 20)
        x = left + i * span + 0.5 * (span - bar_w)
        y = h - bottom - bh
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{colors[n]}"/>')
        lines.append(f'<text x="{x+bar_w/2:.1f}" y="{h-bottom+20}" text-anchor="middle" font-family="Arial" font-size="12">N={n}</text>')
        lines.append(f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{e:.2e}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_series(outdir: Path, results: list[dict[str, np.ndarray | float | int]]) -> None:
    series = [
        ("voltage_V", "Voltage (V)", "voltage_comparison.svg"),
        ("charge_C", "Charge (C)", "charge_comparison.svg"),
        ("current_A", "Current (A)", "current_comparison.svg"),
        ("power_W", "Power (W)", "power_comparison.svg"),
        ("energy_J", "Energy (J)", "energy_comparison.svg"),
    ]
    for key, ylabel, filename in series:
        _write_svg_plot(outdir / filename, ylabel, ylabel, results, key)
    _write_svg_bar(outdir / "energy_bar.svg", results)


def write_validation(path: Path, results: list[dict[str, np.ndarray | float | int]], p: Params) -> None:
    checks = {}
    for r in results:
        n = int(r["n"])
        current_from_q = np.gradient(r["charge_C"], r["time_s"])
        current_to_check = r.get("source_current_A", r["current_A"])
        checks[f"N{n}"] = {
            "power_nonnegative": bool(np.all(r["power_W"] >= -1e-30)),
            "energy_monotonic": bool(np.all(np.diff(r["energy_J"]) >= -1e-30)),
            "v_over_i_matches_load_when_i_nonzero": bool(
                np.allclose(
                    r["voltage_V"][np.abs(r["current_A"]) > 1e-18]
                    / r["current_A"][np.abs(r["current_A"]) > 1e-18],
                    p.load_resistance,
                    rtol=1e-9,
                    atol=1e-9,
                )
            ),
            "source_current_matches_dqdt": bool(np.allclose(current_to_check, current_from_q)),
        }
    path.write_text(json.dumps(checks, indent=2), encoding="utf-8")


def calibrate_charge_density(p: Params, target_n: int, target_transfer_charge: float) -> tuple[Params, dict[str, float | int]]:
    reference = simulate_structure(target_n, p)
    charge = reference["charge_C"]
    transfer_charge = float(np.sum(np.abs(np.diff(charge))))
    if transfer_charge <= 0.0:
        raise ValueError("Reference transfer charge is zero; cannot calibrate charge density.")

    scale = abs(target_transfer_charge) / transfer_charge
    calibrated_density = p.surface_charge_density * scale
    calibrated = Params(
        electrode_width=p.electrode_width,
        gap=p.gap,
        ball_radius=p.ball_radius,
        ball_count=p.ball_count,
        surface_charge_density=calibrated_density,
        speed=p.speed,
        load_resistance=p.load_resistance,
        active_depth=p.active_depth,
        ball_height=p.ball_height,
        coupling_sigma=p.coupling_sigma,
        samples_per_electrode=p.samples_per_electrode,
    )
    info = {
        "calibration_electrode_count": target_n,
        "target_transfer_charge_C": target_transfer_charge,
        "transfer_charge_definition": "sum(abs(diff(Q))) over one constant-speed scan",
        "uncalibrated_cumulative_transfer_charge_C": transfer_charge,
        "charge_density_scale_factor": scale,
        "calibrated_surface_charge_density_C_per_m2": calibrated_density,
    }
    return calibrated, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--charge-density", type=float, default=-50e-6)
    parser.add_argument("--calibrate-n", type=int, default=None)
    parser.add_argument("--target-transfer-charge", type=float, default=None)
    parser.add_argument("--rc-load", action="store_true")
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--load", type=float, default=100e6)
    args = parser.parse_args()

    p = Params(
        surface_charge_density=args.charge_density,
        speed=args.speed,
        load_resistance=args.load,
    )
    calibration = None
    if args.calibrate_n is not None or args.target_transfer_charge is not None:
        if args.calibrate_n is None or args.target_transfer_charge is None:
            raise ValueError("--calibrate-n and --target-transfer-charge must be used together.")
        p, calibration = calibrate_charge_density(p, args.calibrate_n, args.target_transfer_charge)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = [simulate_structure(n, p) for n in (2, 4, 6)]
    if args.rc_load:
        results = [apply_rc_load(r, p) for r in results]
    for r in results:
        write_csv(outdir / f"teng_N{r['n']}_timeseries.csv", r)

    write_summary(outdir / "summary.csv", results)
    plot_series(outdir, results)
    write_validation(outdir / "validation.json", results, p)

    meta = {
        "model": "2D quasi-static equivalent rolling-ball TENG postprocess",
        "electrode_counts": [2, 4, 6],
        "electrode_width_m": p.electrode_width,
        "gap_m": p.gap,
        "ball_radius_m": p.ball_radius,
        "ball_count": p.ball_count,
        "surface_charge_density_C_per_m2": p.surface_charge_density,
        "surface_charge_density_uC_per_m2": p.surface_charge_density * 1e6,
        "speed_m_per_s": p.speed,
        "load_resistance_ohm": p.load_resistance,
        "calibration": calibration,
        "load_model": "RC Norton equivalent" if args.rc_load else "instantaneous resistor",
        "note": "Q is an induced-charge proxy from Gaussian electrode coupling. With --rc-load, Q drives a Norton source in parallel with device capacitance and the reported V/I/P/E are load-side RC dynamic outputs. When calibration is enabled, all electrode counts use the same calibrated PTFE equivalent charge density.",
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
