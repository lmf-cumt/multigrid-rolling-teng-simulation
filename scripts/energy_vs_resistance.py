from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from teng_quasistatic_sim import Params, apply_rc_load, calibrate_charge_density, simulate_structure


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results"


def write_csv(path: Path, loads: np.ndarray, energies: dict[int, list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["resistance_ohm", "energy_N2_J", "energy_N4_J", "energy_N6_J"])
        for i, load in enumerate(loads):
            writer.writerow(
                [
                    f"{float(load):.12e}",
                    f"{energies[2][i]:.12e}",
                    f"{energies[4][i]:.12e}",
                    f"{energies[6][i]:.12e}",
                ]
            )


def write_peak_summary(path: Path, loads: np.ndarray, energies: dict[int, list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["electrode_count", "optimal_resistance_ohm", "max_energy_J"])
        for n in (2, 4, 6):
            arr = np.array(energies[n])
            idx = int(np.argmax(arr))
            writer.writerow([n, f"{float(loads[idx]):.12e}", f"{float(arr[idx]):.12e}"])


def log_svg_polyline(x: np.ndarray, y: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float) -> str:
    w, h = 820, 460
    left, right, top, bottom = 85, 25, 30, 60
    lx = np.log10(x)
    ly = np.log10(y)
    px = left + (lx - xmin) / (xmax - xmin) * (w - left - right)
    py = top + (ymax - ly) / (ymax - ymin) * (h - top - bottom)
    return " ".join(f"{float(a):.2f},{float(b):.2f}" for a, b in zip(px, py))


def write_svg(path: Path, loads: np.ndarray, energies: dict[int, list[float]]) -> None:
    colors = {2: "#2f6f9f", 4: "#34a853", 6: "#b26a00"}
    w, h = 820, 460
    left, right, top, bottom = 85, 25, 30, 60
    lx_min = math.floor(math.log10(float(loads.min())))
    lx_max = math.ceil(math.log10(float(loads.max())))
    all_e = np.array([v for values in energies.values() for v in values])
    ly_min = math.floor(math.log10(float(all_e.min())))
    ly_max = math.ceil(math.log10(float(all_e.max())))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="20" text-anchor="middle" font-family="Arial" font-size="15">RC-load output energy vs load resistance</text>',
        f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-bottom}" stroke="#333"/>',
        f'<text x="{w/2}" y="{h-15}" text-anchor="middle" font-family="Arial" font-size="12">Load resistance (Ohm)</text>',
        f'<text x="18" y="{h/2}" text-anchor="middle" transform="rotate(-90 18 {h/2})" font-family="Arial" font-size="12">Output energy per scan (J)</text>',
    ]

    for decade in range(lx_min, lx_max + 1):
        x = left + (decade - lx_min) / (lx_max - lx_min) * (w - left - right)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h-bottom}" stroke="#eee"/>')
        lines.append(f'<text x="{x:.1f}" y="{h-bottom+22}" text-anchor="middle" font-family="Arial" font-size="10">1e{decade}</text>')

    for decade in range(ly_min, ly_max + 1):
        y = top + (ly_max - decade) / (ly_max - ly_min) * (h - top - bottom)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" stroke="#eee"/>')
        lines.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">1e{decade}</text>')

    for idx, n in enumerate((2, 4, 6)):
        y = np.array(energies[n])
        pts = log_svg_polyline(loads, y, lx_min, lx_max, ly_min, ly_max)
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{colors[n]}" stroke-width="2.2"/>')
        lines.append(
            f'<text x="{w-right-85}" y="{top+24+20*idx}" font-family="Arial" font-size="12" fill="{colors[n]}">N={n}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = Params(surface_charge_density=-50e-6, speed=0.1, load_resistance=100e6)
    calibrated, _ = calibrate_charge_density(base, target_n=4, target_transfer_charge=650e-9)
    loads = np.logspace(3, 14, 221)

    energies: dict[int, list[float]] = {2: [], 4: [], 6: []}
    for load in loads:
        p = Params(
            electrode_width=calibrated.electrode_width,
            gap=calibrated.gap,
            ball_radius=calibrated.ball_radius,
            ball_count=calibrated.ball_count,
            surface_charge_density=calibrated.surface_charge_density,
            speed=calibrated.speed,
            load_resistance=float(load),
            active_depth=calibrated.active_depth,
            ball_height=calibrated.ball_height,
            coupling_sigma=calibrated.coupling_sigma,
            samples_per_electrode=calibrated.samples_per_electrode,
        )
        for n in (2, 4, 6):
            result = apply_rc_load(simulate_structure(n, p), p)
            energies[n].append(float(result["energy_J"][-1]))

    write_csv(OUTDIR / "energy_vs_resistance.csv", loads, energies)
    write_peak_summary(OUTDIR / "energy_vs_resistance_peaks.csv", loads, energies)
    write_svg(OUTDIR / "energy_vs_resistance.svg", loads, energies)


if __name__ == "__main__":
    main()
