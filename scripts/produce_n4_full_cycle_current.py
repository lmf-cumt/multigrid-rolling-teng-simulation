from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "results_multigrid_calibrated" / "n4_full_cycle_current"
SOURCE_CURRENT_CSV = PACKAGE_DIR / "n4_full_cycle_source_current.csv"
SOURCE_CURRENT_SVG = PACKAGE_DIR / "n4_full_cycle_source_current.svg"
LOAD_RESPONSE_CSV = PACKAGE_DIR / "n4_full_cycle_current_1G.csv"
LOAD_RESPONSE_SVG = PACKAGE_DIR / "n4_full_cycle_current_1G.svg"
TRANSFER_CHARGE_CSV = PACKAGE_DIR / "n4_full_cycle_transfer_charge.csv"
TRANSFER_CHARGE_SVG = PACKAGE_DIR / "n4_full_cycle_transfer_charge.svg"
DOC = PACKAGE_DIR / "四电极完整周期电流图生成说明.md"


# Frozen patent-figure parameters. Do not import the mutable comparison script.
STROKE_M = 100e-3
ACCELERATION = 4.0
DWELL_START_S = 100e-3
DWELL_STOP_S = 100e-3
LOAD_OHM = 1e9
INTERNAL_CAPACITANCE_F = 112.20184543019652e-12
N4_TRANSFER_CHARGE_C = 650e-9
SAMPLES = 12001
CURRENT_ALIGNMENT_PHASE_SHIFT = 0.434


# Empirical four-electrode charge-source template extracted from
# c:\Users\lmf\Desktop\20260507\4ELE\2-TENG\a-5-Q-650nC.txt.
# The 120 stable trough-to-trough cycles were smoothed with a 20 ms moving
# average, phase-normalized, averaged, and then scaled to 650 nC peak-to-peak.
EMPIRICAL_TEMPLATE_PHASE = np.array(
    [
        0.00000,
        0.01250,
        0.02500,
        0.03750,
        0.05000,
        0.06250,
        0.07500,
        0.08750,
        0.10000,
        0.11250,
        0.12500,
        0.13750,
        0.15000,
        0.16250,
        0.17500,
        0.18750,
        0.20000,
        0.21250,
        0.22500,
        0.23750,
        0.25000,
        0.26250,
        0.27500,
        0.28750,
        0.30000,
        0.31250,
        0.32500,
        0.33750,
        0.35000,
        0.36250,
        0.37500,
        0.38750,
        0.40000,
        0.41250,
        0.42500,
        0.43750,
        0.45000,
        0.46250,
        0.47500,
        0.48750,
        0.50000,
        0.51250,
        0.52500,
        0.53750,
        0.55000,
        0.56250,
        0.57500,
        0.58750,
        0.60000,
        0.61250,
        0.62500,
        0.63750,
        0.65000,
        0.66250,
        0.67500,
        0.68750,
        0.70000,
        0.71250,
        0.72500,
        0.73750,
        0.75000,
        0.76250,
        0.77500,
        0.78750,
        0.80000,
        0.81250,
        0.82500,
        0.83750,
        0.85000,
        0.86250,
        0.87500,
        0.88750,
        0.90000,
        0.91250,
        0.92500,
        0.93750,
        0.95000,
        0.96250,
        0.97500,
        0.98750,
        1.00000,
    ],
    dtype=float,
)

EMPIRICAL_TEMPLATE_CHARGE_NC = np.array(
    [
        -323.690,
        -320.958,
        -308.219,
        -278.256,
        -231.183,
        -175.089,
        -122.486,
        -74.590,
        -23.596,
        24.141,
        53.976,
        54.028,
        32.822,
        13.439,
        0.048,
        -1.803,
        22.443,
        59.358,
        83.566,
        92.127,
        96.671,
        104.077,
        112.165,
        117.189,
        118.575,
        117.847,
        117.190,
        119.017,
        125.147,
        136.329,
        153.484,
        177.173,
        203.665,
        227.884,
        249.421,
        269.112,
        288.143,
        305.229,
        316.888,
        323.046,
        324.996,
        321.285,
        307.141,
        278.147,
        233.931,
        179.965,
        123.875,
        71.718,
        25.264,
        -15.907,
        -42.264,
        -38.464,
        -15.299,
        1.045,
        7.621,
        3.893,
        -18.946,
        -47.967,
        -63.875,
        -69.735,
        -75.869,
        -84.043,
        -90.318,
        -92.225,
        -91.839,
        -92.211,
        -95.191,
        -103.003,
        -118.467,
        -142.058,
        -169.320,
        -194.642,
        -218.071,
        -242.641,
        -267.669,
        -289.311,
        -303.854,
        -312.333,
        -318.140,
        -322.138,
        -323.834,
    ],
    dtype=float,
)


def build_full_cycle_motion() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_acc = math.sqrt(STROKE_M / ACCELERATION)
    t_motion = 2.0 * t_acc
    duration = 2.0 * t_motion + DWELL_START_S + DWELL_STOP_S
    t = np.linspace(0.0, duration, SAMPLES)
    x = np.zeros_like(t)
    v = np.zeros_like(t)

    t0 = 0.0
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


def source_charge(time_s: np.ndarray) -> np.ndarray:
    # The previous cosine-in-position model over-smoothed the measured charge
    # waveform. Use the averaged measured cycle as the equivalent charge source.
    phase = np.clip(time_s / time_s[-1], 0.0, 1.0)
    q_template = EMPIRICAL_TEMPLATE_CHARGE_NC.copy()
    trough = min(q_template[0], q_template[-1], q_template.min())
    q_template[0] = trough
    q_template[-1] = trough
    q_template = (q_template - q_template.min()) / (q_template.max() - q_template.min())
    q_template = (q_template - 0.5) * N4_TRANSFER_CHARGE_C
    aligned_phase = (phase - CURRENT_ALIGNMENT_PHASE_SHIFT) % 1.0
    return np.interp(aligned_phase, EMPIRICAL_TEMPLATE_PHASE, q_template)


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


def motion_markers() -> list[tuple[float, str]]:
    t_acc = math.sqrt(STROKE_M / ACCELERATION)
    t_motion = 2.0 * t_acc
    reverse_start = t_motion + DWELL_STOP_S
    home_reached = reverse_start + t_motion
    return [
        (t_acc, "forward accel end"),
        (t_motion, "end reached"),
        (reverse_start, "reverse start"),
        (reverse_start + t_acc, "reverse accel end"),
        (home_reached, "home reached"),
        (home_reached + DWELL_START_S, "home dwell end"),
    ]


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


def write_transfer_charge_csv(path: Path, t: np.ndarray, q_source_c: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q_transfer_c = q_source_c - q_source_c[0]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "source_charge_centered_nC",
                "transfer_charge_from_start_nC",
            ]
        )
        for row in zip(t, q_source_c * 1e9, q_transfer_c * 1e9):
            writer.writerow([f"{float(value):.12e}" for value in row])


def write_svg(path: Path, time_s: np.ndarray, current_uA: np.ndarray, *, current_kind: str) -> None:
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
    markers = motion_markers()
    if current_kind == "source":
        title = "四电极滚球式 TENG 实测对齐源电流图"
        subtitle = f"实测电荷模板校准；source current = dQ/dt；Qtransfer = 650 nC；a = 4 m/s²；dwell = 100 ms；T = {time_s[-1]:.6f} s"
        legend = "N=4, source current, full cycle"
        stroke = "#34a853"
    else:
        title = "四电极滚球式 TENG 1 GΩ 负载响应电流图"
        subtitle = f"派生负载响应；R = 1 GΩ；Ceq = 112.2 pF；a = 4 m/s²；dwell = 100 ms；T = {time_s[-1]:.6f} s"
        legend = "N=4, R=1 GΩ load response"
        stroke = "#9c27b0"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="18" font-weight="700">{title}</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="12" fill="#55616d">{subtitle}</text>',
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
    lines.append(f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="2.2"/>')
    lines.append(f'<text x="{width - 270}" y="{top + 22}" font-family="Arial" font-size="13" fill="{stroke}">{legend}</text>')
    lines.append(f'<text x="{width - 270}" y="{top + 42}" font-family="Arial" font-size="12" fill="#55616d">Ipk = {max(abs(current_uA)):.2f} uA</text>')
    lines.append(f'<text x="{width - 270}" y="{top + 62}" font-family="Arial" font-size="12" fill="#55616d">x: 0-100-0 mm</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_transfer_charge_svg(path: Path, time_s: np.ndarray, transfer_charge_nC: np.ndarray) -> None:
    width, height = 980, 560
    left, right, top, bottom = 92, 34, 52, 76
    xmin, xmax = float(time_s.min()), float(time_s.max())
    ymin, ymax = float(transfer_charge_nC.min()), float(transfer_charge_nC.max())
    pad = 0.08 * (ymax - ymin if ymax != ymin else 1.0)
    ymin -= pad
    ymax += pad

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    points = " ".join(f"{sx(float(t)):.2f},{sy(float(q)):.2f}" for t, q in zip(time_s, transfer_charge_nC))
    markers = motion_markers()

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="18" font-weight="700">四电极滚球式 TENG 转移电荷-时间曲线</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="12" fill="#55616d">实测电荷模板校准；Qtransfer = 650 nC；a = 4 m/s²；dwell = 100 ms；T = {time_s[-1]:.6f} s</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333" stroke-width="1.2"/>',
        f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="13">Time (s)</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-family="Arial, Microsoft YaHei" font-size="13">Transferred charge (nC)</text>',
    ]

    for frac in np.linspace(0, 1, 7):
        y_px = top + frac * (height - top - bottom)
        value = ymax - frac * (ymax - ymin)
        lines.append(f'<line x1="{left}" y1="{y_px:.1f}" x2="{width - right}" y2="{y_px:.1f}" stroke="#e1e5ea"/>')
        lines.append(f'<text x="{left - 8}" y="{y_px + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.0f}</text>')

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
    lines.append(f'<polyline points="{points}" fill="none" stroke="#1a73e8" stroke-width="2.2"/>')
    lines.append(f'<text x="{width - 310}" y="{top + 22}" font-family="Arial" font-size="13" fill="#1a73e8">N=4, transfer charge from initial state</text>')
    lines.append(f'<text x="{width - 310}" y="{top + 42}" font-family="Arial" font-size="12" fill="#55616d">Qpp = {transfer_charge_nC.max() - transfer_charge_nC.min():.1f} nC</text>')
    lines.append(f'<text x="{width - 310}" y="{top + 62}" font-family="Arial" font-size="12" fill="#55616d">x: 0-100-0 mm</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_doc(path: Path, time_s: np.ndarray, source_current_uA: np.ndarray, load_current_uA: np.ndarray) -> None:
    text = f"""# 四电极完整周期电流图生成说明

## 为什么重新整理

此前复现图与前一版结果不同，是因为绘图脚本调用了仍在变化的主仿真脚本，导致模型假设和参数发生漂移。当前脚本已改为独立冻结版本，不再导入或调用主仿真脚本。

## 冻结参数

```text
四电极结构：A-B-A-B
电荷源模型：实测周期模板校准
电流图口径：源电流/电荷计电流，source current = dQ/dt
PTFE 电荷密度：固定，波形形状由实测电荷曲线校准
四电极实测转移电荷：650 nC
运动距离：100 mm
加速度/减速度：4 m/s²
终点停留：{DWELL_STOP_S * 1e3:.0f} ms
起点停留：{DWELL_START_S * 1e3:.0f} ms
派生负载响应：R = 1 GΩ，Ceq = 112.2 pF
```

## 完整周期

完整周期为：

```text
正向加速/减速到 100 mm -> 终点停留 -> 反向加速/减速回到 0 mm -> 起点停留
```

周期总时长：

```text
T = {time_s[-1]:.6f} s
```

## 图中结果

```text
源电流峰值：{max(abs(source_current_uA)):.3f} μA
源电流最大正值：{source_current_uA.max():.3f} μA
源电流最大负值：{source_current_uA.min():.3f} μA
1 GΩ 负载响应峰值：{max(abs(load_current_uA)):.3f} μA
转移电荷峰峰值：650.000 nC
```

同时输出转移电荷曲线和 1 GΩ 派生负载响应：

```text
n4_full_cycle_source_current.svg
n4_full_cycle_source_current.csv
n4_full_cycle_transfer_charge.svg
n4_full_cycle_transfer_charge.csv
n4_full_cycle_current_1G.svg
n4_full_cycle_current_1G.csv
```

## 实测对齐说明

电荷源模板来自 `a-5-Q-650nC.txt`。电流图使用 `a-4-I-6uA.txt` 校验相位和输出口径：实测电流对应源电流/电荷计电流，而不是 1 GΩ 负载响应。

## 电流波形平滑性说明

当前生成的电流波形可能存在局部不平滑现象，原因如下：

1. **线性插值导致导数阶梯状**：`source_charge()` 使用 `np.interp` 对 81 点实测模板做分段线性插值，电荷 Q(t) 分段线性，其导数 I = dQ/dt 分段常数，在模板点之间跳变。
2. **模板端点斜率不匹配**：虽然强制 Q(0)=Q(1) 连续，但模板两端斜率不同，相位绕回处电荷曲线存在折角，导致电流突变。
3. **实测噪声被微分放大**：模板本身含残余高频噪声，数值微分对噪声敏感。

如需更平滑的电流波形，可考虑：
- 将 `np.interp` 替换为三次样条插值（`scipy.interpolate.CubicSpline`），保证 C² 连续；
- 对模板先做 Savitzky-Golay 滤波再插值；
- 使用周期性边界样条确保端点导数连续。

## 专利表述

在相同 PTFE 球电荷密度和相同机械激励条件下，四电极交替电极结构通过增加 A/B 电极组之间的交替边界，使滚球在完整往复周期内产生多次感应电荷重分布，形成多个交流电流脉冲。输出增强来自电极结构对电荷转移过程的调控，而非 PTFE 球表面带电量增加。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    t, x, v = build_full_cycle_motion()
    q = source_charge(t)
    response = solve_load(t, q)
    source_current_uA = response["source_current_A"] * 1e6
    load_current_uA = response["load_current_A"] * 1e6
    transfer_charge_nC = (q - q[0]) * 1e9
    write_csv(SOURCE_CURRENT_CSV, t, x, v, q, response)
    write_csv(LOAD_RESPONSE_CSV, t, x, v, q, response)
    write_transfer_charge_csv(TRANSFER_CHARGE_CSV, t, q)
    write_svg(SOURCE_CURRENT_SVG, t, source_current_uA, current_kind="source")
    write_svg(LOAD_RESPONSE_SVG, t, load_current_uA, current_kind="load")
    write_transfer_charge_svg(TRANSFER_CHARGE_SVG, t, transfer_charge_nC)
    write_doc(DOC, t, source_current_uA, load_current_uA)
    print(SOURCE_CURRENT_SVG)
    print(TRANSFER_CHARGE_SVG)
    print(LOAD_RESPONSE_SVG)


if __name__ == "__main__":
    main()
