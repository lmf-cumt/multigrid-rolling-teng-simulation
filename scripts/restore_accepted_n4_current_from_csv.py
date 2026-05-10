from __future__ import annotations

import argparse
import csv
import struct
import zlib
from pathlib import Path

import numpy as np


def read_series(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    t: list[float] = []
    i: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["time_s"]))
            i.append(float(row["load_current_A"]) * 1e6)
    return np.array(t), np.array(i)


def write_svg(path: Path, t: np.ndarray, current_uA: np.ndarray) -> None:
    width, height = 860, 450
    left, right, top, bottom = 90, 30, 34, 62
    xmin, xmax = float(t.min()), float(t.max())
    ymin, ymax = float(current_uA.min()), float(current_uA.max())
    pad = 0.1 * (ymax - ymin if ymax != ymin else 1.0)
    ymin -= pad
    ymax += pad
    px = left + (t - xmin) / (xmax - xmin) * (width - left - right)
    py = top + (ymax - current_uA) / (ymax - ymin) * (height - top - bottom)
    points = " ".join(f"{float(x):.2f},{float(y):.2f}" for x, y in zip(px, py))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="Arial" font-size="15">Accepted N=4 current waveform from old CSV</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="12">Time (s)</text>',
        f'<text x="16" y="{height/2}" text-anchor="middle" transform="rotate(-90 16 {height/2})" font-family="Arial" font-size="12">Current (uA)</text>',
    ]
    for frac in np.linspace(0, 1, 6):
        yy = top + frac * (height - top - bottom)
        value = ymax - frac * (ymax - ymin)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e1e5ea"/>')
        lines.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{value:.2f}</text>')
    if ymin < 0 < ymax:
        y0 = top + (ymax - 0.0) / (ymax - ymin) * (height - top - bottom)
        lines.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{width-right}" y2="{y0:.1f}" stroke="#888" stroke-dasharray="4 4"/>')
    lines.append(f'<polyline points="{points}" fill="none" stroke="#c5221f" stroke-width="2"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_png(path: Path, t: np.ndarray, current_uA: np.ndarray) -> None:
    width, height = 1400, 720
    left, right, top, bottom = 120, 70, 70, 95
    pixels = bytearray([255] * width * height * 3)
    xmin, xmax = float(t.min()), float(t.max())
    ymin, ymax = float(current_uA.min()), float(current_uA.max())
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

    for frac in np.linspace(0, 1, 6):
        y = int(top + frac * (height - top - bottom))
        line(left, y, width - right, y, (225, 229, 234), 1)
    line(left, height - bottom, width - right, height - bottom, (45, 45, 45), 2)
    line(left, top, left, height - bottom, (45, 45, 45), 2)
    if ymin < 0 < ymax:
        y0 = int(top + (ymax - 0.0) / (ymax - ymin) * (height - top - bottom))
        for x in range(left, width - right, 14):
            line(x, y0, min(x + 7, width - right), y0, (130, 130, 130), 1)

    px = left + (t - xmin) / (xmax - xmin) * (width - left - right)
    py = top + (ymax - current_uA) / (ymax - ymin) * (height - top - bottom)
    step = max(1, len(px) // 4500)
    for idx in range(step, len(px), step):
        line(int(px[idx - step]), int(py[idx - step]), int(px[idx]), int(py[idx]), (197, 34, 31), 2)

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results_multigrid_calibrated/teng_N4_1G_timeseries.csv")
    parser.add_argument("--outdir", default="results_accepted_waveform")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t, current_uA = read_series(csv_path)
    write_svg(outdir / "accepted_N4_current_from_old_csv.svg", t, current_uA)
    write_png(outdir / "accepted_N4_current_from_old_csv.png", t, current_uA)
    with (outdir / "accepted_N4_current_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_csv", "current_field", "peak_positive_uA", "peak_negative_uA", "peak_to_peak_uA", "duration_s"])
        writer.writerow(
            [
                str(csv_path),
                "load_current_A",
                f"{float(np.max(current_uA)):.12e}",
                f"{float(np.min(current_uA)):.12e}",
                f"{float(np.max(current_uA) - np.min(current_uA)):.12e}",
                f"{float(t[-1] - t[0]):.12e}",
            ]
        )


if __name__ == "__main__":
    main()
