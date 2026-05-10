from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelParams:
    rows: int = 16
    cols: int = 9
    pcb_thickness_m: float = 1e-3
    electrode_width_m: float = 16e-3
    electrode_gap_m: float = 5e-3
    stroke_m: float = 100e-3
    acceleration_m_s2: float = 4.0
    dwell_start_s: float = 30e-3
    dwell_stop_s: float = 30e-3
    load_ohm: float = 1e9
    q_transfer_n2_C: float = 380e-9
    q_transfer_n4_C: float = 650e-9
    current_peak_n4_A: float = 6e-6
    samples: int = 12000

    @property
    def parallel_units(self) -> int:
        return self.rows * self.cols


def event_count(electrode_count: int) -> int:
    # One event for dual electrodes, two for four electrodes, three for six.
    return max(1, electrode_count // 2)


def pulse_weights(electrode_count: int, p: ModelParams) -> np.ndarray:
    count = event_count(electrode_count)
    if count == 1:
        return np.array([1.0])
    # The N=4 measured charge fixes the relative contribution of the second
    # event: Q4/Q2 = 1 + r. Extend the same attenuation trend for N=6.
    r = p.q_transfer_n4_C / p.q_transfer_n2_C - 1.0
    return np.array([r**k for k in range(count)], dtype=float)


def transfer_charge(electrode_count: int, p: ModelParams) -> float:
    if electrode_count == 2:
        return p.q_transfer_n2_C
    return p.q_transfer_n2_C * float(np.sum(pulse_weights(electrode_count, p)))


def build_motion(p: ModelParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_acc = math.sqrt(p.stroke_m / p.acceleration_m_s2)
    t_motion = 2.0 * t_acc
    duration = p.dwell_start_s + t_motion + p.dwell_stop_s + t_motion + p.dwell_start_s
    t = np.linspace(0.0, duration, p.samples)
    x = np.zeros_like(t)
    v = np.zeros_like(t)

    t0 = p.dwell_start_s
    t1 = t0 + t_acc
    t2 = t0 + t_motion
    t3 = t2 + p.dwell_stop_s
    t4 = t3 + t_acc
    t5 = t3 + t_motion
    v_peak = p.acceleration_m_s2 * t_acc

    mask = (t >= t0) & (t < t1)
    tau = t[mask] - t0
    x[mask] = 0.5 * p.acceleration_m_s2 * tau**2
    v[mask] = p.acceleration_m_s2 * tau

    mask = (t >= t1) & (t < t2)
    tau = t[mask] - t1
    x[mask] = 0.5 * p.stroke_m + v_peak * tau - 0.5 * p.acceleration_m_s2 * tau**2
    v[mask] = v_peak - p.acceleration_m_s2 * tau

    x[t >= t2] = p.stroke_m

    mask = (t >= t3) & (t < t4)
    tau = t[mask] - t3
    x[mask] = p.stroke_m - 0.5 * p.acceleration_m_s2 * tau**2
    v[mask] = -p.acceleration_m_s2 * tau

    mask = (t >= t4) & (t < t5)
    tau = t[mask] - t4
    x[mask] = 0.5 * p.stroke_m - v_peak * tau + 0.5 * p.acceleration_m_s2 * tau**2
    v[mask] = -v_peak + p.acceleration_m_s2 * tau

    x[t >= t5] = 0.0
    v[t >= t5] = 0.0
    return t, x, v


def sech2_pulses(t: np.ndarray, centers: np.ndarray, weights: np.ndarray, width: float) -> np.ndarray:
    z = (t[:, None] - centers[None, :]) / width
    return np.sum(weights[None, :] / np.cosh(np.clip(z, -30.0, 30.0)) ** 2, axis=1)


def integrate(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])


def source_current_and_charge(t: np.ndarray, electrode_count: int, p: ModelParams) -> tuple[np.ndarray, np.ndarray]:
    count = event_count(electrode_count)
    weights = pulse_weights(electrode_count, p)
    q_total = transfer_charge(electrode_count, p)
    t_acc = math.sqrt(p.stroke_m / p.acceleration_m_s2)
    t_motion = 2.0 * t_acc
    forward_start = p.dwell_start_s
    forward_end = forward_start + t_motion
    reverse_start = forward_end + p.dwell_stop_s
    reverse_end = reverse_start + t_motion

    centers_f = forward_start + (np.arange(count) + 0.5) / count * t_motion
    centers_r = reverse_start + (np.arange(count) + 0.5) / count * t_motion
    # Width is calibrated from the four-electrode measured peak current while
    # preserving the measured N=4 transferred charge.
    width = fit_pulse_width_for_n4(t, p)
    raw = sech2_pulses(t, centers_f, weights, width) - sech2_pulses(t, centers_r, weights, width)
    q_raw = integrate(t, raw)
    qpp_raw = float(np.max(q_raw) - np.min(q_raw))
    current = raw * (q_total / qpp_raw)
    charge = integrate(t, current)
    current[(t < forward_start) | ((t > forward_end) & (t < reverse_start)) | (t > reverse_end)] *= 0.0
    charge = integrate(t, current)
    charge *= q_total / (float(np.max(charge) - np.min(charge)))
    current = np.gradient(charge, t)
    return current, charge


def fit_pulse_width_for_n4(t: np.ndarray, p: ModelParams) -> float:
    lo, hi = 1e-4, 0.2
    for _ in range(70):
        mid = math.sqrt(lo * hi)
        current, _ = source_current_and_charge_with_width(t, 4, p, mid)
        peak = float(np.max(np.abs(current)))
        if peak > p.current_peak_n4_A:
            lo = mid
        else:
            hi = mid
    return hi


def source_current_and_charge_with_width(
    t: np.ndarray, electrode_count: int, p: ModelParams, width: float
) -> tuple[np.ndarray, np.ndarray]:
    count = event_count(electrode_count)
    weights = pulse_weights(electrode_count, p)
    q_total = transfer_charge(electrode_count, p)
    t_acc = math.sqrt(p.stroke_m / p.acceleration_m_s2)
    t_motion = 2.0 * t_acc
    forward_start = p.dwell_start_s
    reverse_start = forward_start + t_motion + p.dwell_stop_s
    centers_f = forward_start + (np.arange(count) + 0.5) / count * t_motion
    centers_r = reverse_start + (np.arange(count) + 0.5) / count * t_motion
    raw = sech2_pulses(t, centers_f, weights, width) - sech2_pulses(t, centers_r, weights, width)
    q_raw = integrate(t, raw)
    raw *= q_total / (float(np.max(q_raw) - np.min(q_raw)))
    q = integrate(t, raw)
    return raw, q


def write_csv(path: Path, t: np.ndarray, x: np.ndarray, v: np.ndarray, current: np.ndarray, charge: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "position_m", "velocity_m_s", "short_circuit_current_A", "short_circuit_current_uA", "transferred_charge_C", "transferred_charge_nC"])
        for row in zip(t, x, v, current, current * 1e6, charge, charge * 1e9):
            writer.writerow([f"{float(value):.12e}" for value in row])


def svg_plot(path: Path, title: str, ylabel: str, t: np.ndarray, y: np.ndarray, color: str) -> None:
    width, height = 820, 440
    left, right, top, bottom = 92, 24, 32, 62
    xmin, xmax = float(t.min()), float(t.max())
    ymin, ymax = float(y.min()), float(y.max())
    pad = 0.1 * (ymax - ymin if ymax != ymin else 1.0)
    ymin -= pad
    ymax += pad

    px = left + (t - xmin) / (xmax - xmin) * (width - left - right)
    py = top + (ymax - y) / (ymax - ymin) * (height - top - bottom)
    points = " ".join(f"{float(a):.2f},{float(b):.2f}" for a, b in zip(px, py))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="Arial" font-size="15">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="12">Time (s)</text>',
        f'<text x="16" y="{height/2}" text-anchor="middle" transform="rotate(-90 16 {height/2})" font-family="Arial" font-size="12">{ylabel}</text>',
    ]
    for frac in np.linspace(0, 1, 6):
        yy = top + frac * (height - top - bottom)
        value = ymax - frac * (ymax - ymin)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e1e5ea"/>')
        lines.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{value:.2f}</text>')
    if ymin < 0 < ymax:
        y0 = top + (ymax - 0.0) / (ymax - ymin) * (height - top - bottom)
        lines.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{width-right}" y2="{y0:.1f}" stroke="#999" stroke-dasharray="4 4"/>')
    lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_multi_current_plot(path: Path, t: np.ndarray, series: dict[int, np.ndarray]) -> None:
    width, height = 900, 460
    left, right, top, bottom = 92, 118, 34, 64
    xmin, xmax = float(t.min()), float(t.max())
    all_y = np.concatenate([value for value in series.values()])
    ymin, ymax = float(all_y.min()), float(all_y.max())
    pad = 0.1 * (ymax - ymin if ymax != ymin else 1.0)
    ymin -= pad
    ymax += pad
    colors = {2: "#3367d6", 4: "#c5221f", 6: "#188038"}

    def xy_points(y: np.ndarray) -> str:
        px = left + (t - xmin) / (xmax - xmin) * (width - left - right)
        py = top + (ymax - y) / (ymax - ymin) * (height - top - bottom)
        return " ".join(f"{float(a):.2f},{float(b):.2f}" for a, b in zip(px, py))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="23" text-anchor="middle" font-family="Arial" font-size="15">Short-circuit current comparison</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{(width-right+left)/2}" y="{height-17}" text-anchor="middle" font-family="Arial" font-size="12">Time (s)</text>',
        f'<text x="16" y="{height/2}" text-anchor="middle" transform="rotate(-90 16 {height/2})" font-family="Arial" font-size="12">Current (uA)</text>',
    ]
    for frac in np.linspace(0, 1, 6):
        yy = top + frac * (height - top - bottom)
        value = ymax - frac * (ymax - ymin)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e1e5ea"/>')
        lines.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{value:.2f}</text>')
    if ymin < 0 < ymax:
        y0 = top + (ymax - 0.0) / (ymax - ymin) * (height - top - bottom)
        lines.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{width-right}" y2="{y0:.1f}" stroke="#999" stroke-dasharray="4 4"/>')
    for n, y in series.items():
        lines.append(f'<polyline points="{xy_points(y)}" fill="none" stroke="{colors[n]}" stroke-width="2"/>')
    legend_x = width - right + 20
    for i, n in enumerate(series):
        yy = top + 24 + i * 24
        lines.append(f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x+28}" y2="{yy}" stroke="{colors[n]}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+36}" y="{yy+4}" font-family="Arial" font-size="12">N={n}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = bytearray()
    stride = width * 3
    for y in range(height):
        rows.append(0)
        rows.extend(pixels[y * stride : (y + 1) * stride])
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(rows), level=9)))
    png.extend(chunk(b"IEND", b""))
    path.write_bytes(bytes(png))


def png_current_plot(path: Path, t: np.ndarray, series: dict[int, np.ndarray]) -> None:
    width, height = 1400, 720
    left, right, top, bottom = 120, 90, 70, 95
    pixels = bytearray([255] * width * height * 3)
    colors = {2: (51, 103, 214), 4: (197, 34, 31), 6: (24, 128, 56)}
    xmin, xmax = float(t.min()), float(t.max())
    all_y = np.concatenate([value for value in series.values()])
    ymin, ymax = float(all_y.min()), float(all_y.max())
    pad = 0.1 * (ymax - ymin if ymax != ymin else 1.0)
    ymin -= pad
    ymax += pad

    def put(x: int, y: int, rgb: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            idx = (y * width + x) * 3
            pixels[idx : idx + 3] = bytes(rgb)

    def line(x0: int, y0: int, x1: int, y1: int, rgb: tuple[int, int, int], thickness: int = 1) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in range(-thickness + 1, thickness):
                for oy in range(-thickness + 1, thickness):
                    put(x0 + ox, y0 + oy, rgb)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def map_xy(tt: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px = left + (tt - xmin) / (xmax - xmin) * (width - left - right)
        py = top + (ymax - yy) / (ymax - ymin) * (height - top - bottom)
        return px.astype(int), py.astype(int)

    # Grid and axes.
    for frac in np.linspace(0, 1, 6):
        y = int(top + frac * (height - top - bottom))
        line(left, y, width - right, y, (225, 229, 234), 1)
    line(left, height - bottom, width - right, height - bottom, (45, 45, 45), 2)
    line(left, top, left, height - bottom, (45, 45, 45), 2)
    if ymin < 0 < ymax:
        y0 = int(top + (ymax - 0.0) / (ymax - ymin) * (height - top - bottom))
        for x in range(left, width - right, 14):
            line(x, y0, min(x + 7, width - right), y0, (135, 135, 135), 1)

    for n, y in series.items():
        px, py = map_xy(t, y)
        step = max(1, len(px) // 4500)
        for i in range(step, len(px), step):
            line(int(px[i - step]), int(py[i - step]), int(px[i]), int(py[i]), colors[n], 2)

    # Minimal legend swatches.
    for i, n in enumerate(series):
        y = top + 24 + i * 28
        line(width - right - 20, y, width - right + 50, y, colors[n], 4)
    write_png(path, width, height, pixels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results_patent_reproduction")
    parser.add_argument("--acceleration", type=float, default=4.0)
    args = parser.parse_args()

    p = ModelParams(acceleration_m_s2=args.acceleration)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t, x, v = build_motion(p)

    summary = []
    current_series = {}
    for n in (2, 4, 6):
        current, charge = source_current_and_charge(t, n, p)
        current_series[n] = current * 1e6
        write_csv(outdir / f"N{n}_short_circuit_waveform.csv", t, x, v, current, charge)
        svg_plot(outdir / f"N{n}_short_circuit_current.svg", f"N={n} short-circuit current", "Current (uA)", t, current * 1e6, "#c5221f")
        svg_plot(outdir / f"N{n}_transferred_charge.svg", f"N={n} transferred charge", "Charge (nC)", t, charge * 1e9, "#2f6f9f")
        summary.append(
            {
                "electrode_count": n,
                "event_count": event_count(n),
                "charge_pp_nC": float(np.max(charge) - np.min(charge)) * 1e9,
                "current_peak_abs_uA": float(np.max(np.abs(current))) * 1e6,
                "current_pp_uA": float(np.max(current) - np.min(current)) * 1e6,
                "duration_s": float(t[-1] - t[0]),
            }
        )

    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    svg_multi_current_plot(outdir / "current_N2_N4_N6_comparison.svg", t, current_series)
    png_current_plot(outdir / "current_N2_N4_N6_comparison.png", t, current_series)
    png_current_plot(outdir / "N4_short_circuit_current.png", t, {4: current_series[4]})
    (outdir / "metadata.json").write_text(json.dumps(p.__dict__, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
