from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "results_multigrid_calibrated" / "n4_full_cycle_current"
PACKAGE_CSV = PACKAGE_DIR / "n4_full_cycle_current_1G.csv"
OUTPUT_SVG = PACKAGE_DIR / "n4_full_cycle_current_1G.svg"
DOC = PACKAGE_DIR / "四电极完整周期电流图生成说明.md"


# Frozen patent-figure parameters. Do not import the mutable comparison script.
STROKE_M = 100e-3
ACCELERATION = 5.0
DWELL_START_S = 30e-3
DWELL_STOP_S = 30e-3
LOAD_OHM = 1e9
INTERNAL_CAPACITANCE_F = 112.20184543019652e-12
N4_TRANSFER_CHARGE_C = 650e-9
EVENT_COUNT = 3  # A-B-A-B has three adjacent alternating A/B boundaries.
SAMPLES = 12000


def build_full_cycle_motion() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_acc = math.sqrt(STROKE_M / ACCELERATION)
    t_motion = 2.0 * t_acc
    duration = DWELL_START_S + t_motion + DWELL_STOP_S + t_motion + DWELL_START_S
    t = np.linspace(0.0, duration, SAMPLES)
    x = np.zeros_like(t)
    v = np.zeros_like(t)

    t0 = DWELL_START_S
    t1 = t0 + t_acc
    t2 = t0 + t_motion
    t3 = t2 + DWELL_STOP_S
    t4 = t3 + t_acc
    t5 = t3 + t_motion
    v_peak = ACCELERATION * t_acc

    acc = (t >= t0) & (t < t1)
    tau = t[acc] - t0
    x[acc] = 0.5 * ACCELERATION * tau**2
    v[acc] = ACCELERATION * tau

    dec = (t >= t1) & (t < t2)
    tau = t[dec] - t1
    x[dec] = 0.5 * STROKE_M + v_peak * tau - 0.5 * ACCELERATION * tau**2
    v[dec] = v_peak - ACCELERATION * tau

    dwell_end = (t >= t2) & (t < t3)
    x[dwell_end] = STROKE_M
    v[dwell_end] = 0.0

    ret_acc = (t >= t3) & (t < t4)
    tau = t[ret_acc] - t3
    x[ret_acc] = STROKE_M - 0.5 * ACCELERATION * tau**2
    v[ret_acc] = -ACCELERATION * tau

    ret_dec = (t >= t4) & (t < t5)
    tau = t[ret_dec] - t4
    x[ret_dec] = 0.5 * STROKE_M - v_peak * tau + 0.5 * ACCELERATION * tau**2
    v[ret_dec] = -v_peak + ACCELERATION * tau

    home = t >= t5
    x[home] = 0.0
    v[home] = 0.0
    return t, x, v


def source_charge(position_m: np.ndarray) -> np.ndarray:
    # Fixed PTFE charge-density model: the four-electrode measured transfer
    # charge sets the peak-to-peak charge, while three A/B boundaries define
    # the number of charge redistribution events during a stroke.
    s = np.clip(position_m / STROKE_M, 0.0, 1.0)
    return 0.5 * N4_TRANSFER_CHARGE_C * (-np.cos(EVENT_COUNT * math.pi * s))


def solve_load(time_s: np.ndarray, q_source_c: np.ndarray) -> dict[str, np.ndarray]:
    i_source = np.gradient(q_source_c, time_s)
    voltage = np.zeros_like(time_s)
    tau_rc = LOAD_OHM * INTERNAL_CAPACITANCE_F
    for i in range(1, len(time_s)):
        dt = time_s[i] - time_s[i - 1]
        alpha = math.exp(-dt / tau_rc)
        i_mid = 0.5 * (i_source[i] + i_source[i - 1])
        voltage[i] = voltage[i - 1] * alpha + LOAD_OHM * i_mid * (1.0 - alpha)
    i_load = voltage / LOAD_OHM
    power = voltage * i_load
    energy = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(time_s))])
    return {
        "source_current_A": i_source,
        "load_current_A": i_load,
        "voltage_V": voltage,
        "power_W": power,
        "energy_J": energy,
    }


def write_csv(path: Path, t: np.ndarray, x: np.ndarray, v: np.ndarray, q: np.ndarray, response: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "position_mm",
                "velocity_m_per_s",
                "source_charge_nC",
                "source_current_uA",
                "load_current_uA",
                "voltage_V",
                "power_mW",
                "energy_uJ",
            ]
        )
        for row in zip(
            t,
            x * 1e3,
            v,
            q * 1e9,
            response["source_current_A"] * 1e6,
            response["load_current_A"] * 1e6,
            response["voltage_V"],
            response["power_W"] * 1e3,
            response["energy_J"] * 1e6,
        ):
            writer.writerow([f"{float(value):.12e}" for value in row])


def write_svg(path: Path, time_s: np.ndarray, current_uA: np.ndarray) -> None:
    width, height = 980, 560
    left, right, top, bottom = 92, 34, 52, 76
    xmin, xmax = float(time_s.min()), float(time_s.max())
    ymin, ymax = float(current_uA.min()), float(current_uA.max())
    pad = 0.10 * (ymax - ymin if ymax != ymin else 1.0)
    ymin -= pad
    ymax += pad

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    points = " ".join(f"{sx(float(t)):.2f},{sy(float(i)):.2f}" for t, i in zip(time_s, current_uA))
    t_acc = math.sqrt(STROKE_M / ACCELERATION)
    markers = [
        (DWELL_START_S, "start dwell end"),
        (DWELL_START_S + t_acc, "forward accel end"),
        (DWELL_START_S + 2 * t_acc, "end reached"),
        (DWELL_START_S + 2 * t_acc + DWELL_STOP_S, "reverse start"),
        (DWELL_START_S + 3 * t_acc + DWELL_STOP_S, "reverse accel end"),
        (DWELL_START_S + 4 * t_acc + DWELL_STOP_S, "home reached"),
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="18" font-weight="700">四电极滚球式 TENG 完整周期电流图</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="12" fill="#55616d">固定 PTFE 电荷密度；Qtransfer = 650 nC；R = 1 Gohm；a = 5 m/s²；完整往复周期 T</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333" stroke-width="1.2"/>',
        f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="13">Time (s)</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-family="Arial, Microsoft YaHei" font-size="13">Current (uA)</text>',
    ]

    for frac in np.linspace(0, 1, 7):
        y_px = top + frac * (height - top - bottom)
        value = ymax - frac * (ymax - ymin)
        lines.append(f'<line x1="{left}" y1="{y_px:.1f}" x2="{width - right}" y2="{y_px:.1f}" stroke="#e1e5ea"/>')
        lines.append(f'<text x="{left - 8}" y="{y_px + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>')

    for frac in np.linspace(0, 1, 8):
        x_px = left + frac * (width - left - right)
        value = xmin + frac * (xmax - xmin)
        lines.append(f'<line x1="{x_px:.1f}" y1="{height - bottom}" x2="{x_px:.1f}" y2="{height - bottom + 5}" stroke="#333"/>')
        lines.append(f'<text x="{x_px:.1f}" y="{height - bottom + 21}" text-anchor="middle" font-family="Arial" font-size="11">{value:.2f}</text>')

    for marker_time, label in markers:
        x_px = sx(marker_time)
        lines.append(f'<line x1="{x_px:.1f}" y1="{top}" x2="{x_px:.1f}" y2="{height - bottom}" stroke="#c7cdd4" stroke-dasharray="5 6"/>')
        lines.append(f'<text x="{x_px + 4:.1f}" y="{top + 14}" font-family="Arial" font-size="10" fill="#6b7680" transform="rotate(90 {x_px + 4:.1f} {top + 14})">{label}</text>')

    lines.append(f'<line x1="{left}" y1="{sy(0):.1f}" x2="{width - right}" y2="{sy(0):.1f}" stroke="#8a949e" stroke-dasharray="4 4"/>')
    lines.append(f'<polyline points="{points}" fill="none" stroke="#34a853" stroke-width="2.2"/>')
    lines.append(f'<text x="{width - 270}" y="{top + 22}" font-family="Arial" font-size="13" fill="#34a853">N=4, R=1 Gohm, full cycle</text>')
    lines.append(f'<text x="{width - 270}" y="{top + 42}" font-family="Arial" font-size="12" fill="#55616d">Ipk = {max(abs(current_uA)):.2f} uA</text>')
    lines.append(f'<text x="{width - 270}" y="{top + 62}" font-family="Arial" font-size="12" fill="#55616d">x: 0-100-0 mm</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_doc(path: Path, time_s: np.ndarray, current_uA: np.ndarray) -> None:
    text = f"""# 四电极完整周期电流图生成说明

## 为什么重新整理

此前复现图与前一版结果不同，是因为绘图脚本调用了仍在变化的主仿真脚本，导致模型假设和参数发生漂移。当前脚本已改为独立冻结版本，不再导入或调用主仿真脚本。

## 冻结参数

```text
四电极结构：A-B-A-B
有效相邻交替边界数：3
PTFE 电荷密度：固定
四电极实测转移电荷：650 nC
运动距离：100 mm
加速度/减速度：5 m/s²
起点停留：30 ms
终点停留：30 ms
负载电阻：1 GΩ
等效内电容：112.2 pF
```

## 完整周期

完整周期为：

```text
起点停留 -> 正向加速/减速到 100 mm -> 终点停留 -> 反向加速/减速回到 0 mm
```

周期总时长：

```text
T = {time_s[-1]:.6f} s
```

## 图中结果

```text
峰值电流：{max(abs(current_uA)):.3f} μA
最大正电流：{current_uA.max():.3f} μA
最大负电流：{current_uA.min():.3f} μA
```

## 专利表述

在相同 PTFE 球电荷密度和相同机械激励条件下，四电极交替电极结构通过增加 A/B 电极组之间的交替边界，使滚球在完整往复周期内产生多次感应电荷重分布，形成多个交流电流脉冲。输出增强来自电极结构对电荷转移过程的调控，而非 PTFE 球表面带电量增加。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    t, x, v = build_full_cycle_motion()
    q = source_charge(x)
    response = solve_load(t, q)
    current_uA = response["load_current_A"] * 1e6
    write_csv(PACKAGE_CSV, t, x, v, q, response)
    write_svg(OUTPUT_SVG, t, current_uA)
    write_doc(DOC, t, current_uA)
    print(OUTPUT_SVG)


if __name__ == "__main__":
    main()
