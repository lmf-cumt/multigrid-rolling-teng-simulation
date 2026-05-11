"""模型对比：解析余弦模型 vs 实测模板位置驱动模型

对比两个电荷源模型在相同运动条件下的源电流与负载电流波形：
  - **解析模型** (Analytical)：Q(x) = ½·Qpp·[-cos(N_events·π·x/STROKE)]，纯余弦驱动
  - **实测模板模型** (Empirical)：120周期实测电荷平均 → 三次样条插值 → 位置驱动
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
COMPARE_DIR = ROOT / "results_multigrid_calibrated" / "model_comparison"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 共享物理参数
# ---------------------------------------------------------------------------
STROKE_M = 100e-3
LOAD_OHM = 1e9
INTERNAL_CAPACITANCE_F = 112.20184543019652e-12
N4_TRANSFER_CHARGE_C = 650e-9
SAMPLES = 6000  # 兼顾速度

# 运动参数 (采用工作区脚本默认值)
ACCELERATION = 4.0
DWELL_START_S = 100e-3
DWELL_STOP_S = 100e-3

# 解析模型特有参数
ANALYTICAL_EVENT_COUNT = 3  # A-B-A-B 有 3 条相邻 A/B 交替边界

# ---------------------------------------------------------------------------
# 实测电荷模板 (从 produce_n4_full_cycle_current.py 提取)
# ---------------------------------------------------------------------------
EMPIRICAL_TEMPLATE_PHASE = np.array(
    [
        0.00000, 0.01250, 0.02500, 0.03750, 0.05000, 0.06250, 0.07500, 0.08750,
        0.10000, 0.11250, 0.12500, 0.13750, 0.15000, 0.16250, 0.17500, 0.18750,
        0.20000, 0.21250, 0.22500, 0.23750, 0.25000, 0.26250, 0.27500, 0.28750,
        0.30000, 0.31250, 0.32500, 0.33750, 0.35000, 0.36250, 0.37500, 0.38750,
        0.40000, 0.41250, 0.42500, 0.43750, 0.45000, 0.46250, 0.47500, 0.48750,
        0.50000, 0.51250, 0.52500, 0.53750, 0.55000, 0.56250, 0.57500, 0.58750,
        0.60000, 0.61250, 0.62500, 0.63750, 0.65000, 0.66250, 0.67500, 0.68750,
        0.70000, 0.71250, 0.72500, 0.73750, 0.75000, 0.76250, 0.77500, 0.78750,
        0.80000, 0.81250, 0.82500, 0.83750, 0.85000, 0.86250, 0.87500, 0.88750,
        0.90000, 0.91250, 0.92500, 0.93750, 0.95000, 0.96250, 0.97500, 0.98750,
        1.00000,
    ],
    dtype=float,
)

EMPIRICAL_TEMPLATE_CHARGE_NC = np.array(
    [
        -323.690, -320.958, -308.219, -278.256, -231.183, -175.089, -122.486, -74.590,
        -23.596, 24.141, 53.976, 54.028, 32.822, 13.439, 0.048, -1.803,
        22.443, 59.358, 83.566, 92.127, 96.671, 104.077, 112.165, 117.189,
        118.575, 117.847, 117.190, 119.017, 125.147, 136.329, 153.484, 177.173,
        203.665, 227.884, 249.421, 269.112, 288.143, 305.229, 316.888, 323.046,
        324.996, 321.285, 307.141, 278.147, 233.931, 179.965, 123.875, 71.718,
        25.264, -15.907, -42.264, -38.464, -15.299, 1.045, 7.621, 3.893,
        -18.946, -47.967, -63.875, -69.735, -75.869, -84.043, -90.318, -92.225,
        -91.839, -92.211, -95.191, -103.003, -118.467, -142.058, -169.320, -194.642,
        -218.071, -242.641, -267.669, -289.311, -303.854, -312.333, -318.140, -322.138,
        -323.834,
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# 运动轨迹 (统一)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 电荷源模型 A：解析余弦
# ---------------------------------------------------------------------------
def source_charge_analytical(position_m: np.ndarray) -> np.ndarray:
    s = np.clip(position_m / STROKE_M, 0.0, 1.0)
    return 0.5 * N4_TRANSFER_CHARGE_C * (-np.cos(ANALYTICAL_EVENT_COUNT * math.pi * s))


# ---------------------------------------------------------------------------
# 电荷源模型 B：实测模板 + 位置驱动
# ---------------------------------------------------------------------------
def compute_position_phase(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    phase = np.zeros_like(t)
    direction = np.ones_like(t)
    for i in range(1, len(t)):
        if v[i] > 1e-12:
            direction[i] = 1
        elif v[i] < -1e-12:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    forward = direction > 0
    phase[forward] = x[forward] / (2.0 * STROKE_M)
    phase[~forward] = 1.0 - x[~forward] / (2.0 * STROKE_M)
    return phase


def source_charge_empirical(time_s: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    phase = compute_position_phase(time_s, x, v)
    q_template = EMPIRICAL_TEMPLATE_CHARGE_NC.copy()
    trough = min(q_template[0], q_template[-1], q_template.min())
    q_template[0] = trough
    q_template[-1] = trough
    q_template = (q_template - q_template.min()) / (q_template.max() - q_template.min())
    q_template = (q_template - 0.5) * N4_TRANSFER_CHARGE_C
    cs = CubicSpline(EMPIRICAL_TEMPLATE_PHASE.copy(), q_template, bc_type="periodic")
    return cs(phase % 1.0)


# ---------------------------------------------------------------------------
# RC 负载求解
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 写 CSV
# ---------------------------------------------------------------------------
def write_comparison_csv(
    path: Path,
    t: np.ndarray,
    q_ana: np.ndarray,
    r_ana: dict[str, np.ndarray],
    q_emp: np.ndarray,
    r_emp: dict[str, np.ndarray],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s",
            "Q_analytical_nC", "Isrc_analytical_uA", "Iload_analytical_uA",
            "Q_empirical_nC", "Isrc_empirical_uA", "Iload_empirical_uA",
        ])
        for i in range(len(t)):
            writer.writerow([
                f"{t[i]:.12e}",
                f"{q_ana[i] * 1e9:.12e}",
                f"{r_ana['source_current_A'][i] * 1e6:.12e}",
                f"{r_ana['load_current_A'][i] * 1e6:.12e}",
                f"{q_emp[i] * 1e9:.12e}",
                f"{r_emp['source_current_A'][i] * 1e6:.12e}",
                f"{r_emp['load_current_A'][i] * 1e6:.12e}",
            ])


# ---------------------------------------------------------------------------
# SVG 对比图
# ---------------------------------------------------------------------------
def write_comparison_svg(
    path: Path,
    time_s: np.ndarray,
    label_a: str,   current_a: np.ndarray,   color_a: str,
    label_b: str,   current_b: np.ndarray,   color_b: str,
    chart_title: str,   y_label: str,
) -> None:
    width, height = 1080, 600
    left, right, top, bottom = 100, 40, 56, 90
    xmin, xmax = float(time_s.min()), float(time_s.max())
    all_data = np.concatenate([current_a, current_b])
    ymin, ymax = float(all_data.min()), float(all_data.max())
    pad = 0.10 * (ymax - ymin if ymax != ymin else 1.0)
    ymin -= pad
    ymax += pad

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    points_a = " ".join(f"{sx(float(t)):.2f},{sy(float(i)):.2f}" for t, i in zip(time_s, current_a))
    points_b = " ".join(f"{sx(float(t)):.2f},{sy(float(i)):.2f}" for t, i in zip(time_s, current_b))

    t_acc = math.sqrt(STROKE_M / ACCELERATION)
    t_motion = 2.0 * t_acc
    t_stop_end = t_motion + DWELL_STOP_S
    markers = [
        (t_acc, "fwd accel end"),
        (t_motion, "end dwell start"),
        (t_stop_end, "rev start"),
        (t_stop_end + t_acc, "rev accel end"),
        (t_stop_end + t_motion, "home reached"),
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="18" font-weight="700">{chart_title}</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="12" fill="#55616d">a = {ACCELERATION} m/s² | dwell = {DWELL_START_S*1e3:.0f}/{DWELL_STOP_S*1e3:.0f} ms | R = 1 GΩ | Qpp = 650 nC</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333" stroke-width="1.2"/>',
        f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="13">Time (s)</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-family="Arial, Microsoft YaHei" font-size="13">{y_label}</text>',
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

    # zero line
    lines.append(f'<line x1="{left}" y1="{sy(0):.1f}" x2="{width - right}" y2="{sy(0):.1f}" stroke="#8a949e" stroke-dasharray="4 4"/>')

    # curves
    lines.append(f'<polyline points="{points_a}" fill="none" stroke="{color_a}" stroke-width="2.2"/>')
    lines.append(f'<polyline points="{points_b}" fill="none" stroke="{color_b}" stroke-width="2.2" stroke-dasharray="8 4"/>')

    # legends
    lines.append(f'<text x="{width - 420}" y="{top + 24}" font-family="Arial" font-size="13" fill="{color_a}">── {label_a}  Ipk={max(abs(current_a)):.2f} uA</text>')
    lines.append(f'<text x="{width - 420}" y="{top + 48}" font-family="Arial" font-size="13" fill="{color_b}">- - {label_b}  Ipk={max(abs(current_b)):.2f} uA</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 电荷波形对比 SVG
# ---------------------------------------------------------------------------
def write_charge_comparison_svg(
    path: Path,
    time_s: np.ndarray,
    q_ana_nC: np.ndarray,
    q_emp_nC: np.ndarray,
) -> None:
    width, height = 1080, 560
    left, right, top, bottom = 100, 40, 52, 76
    xmin, xmax = float(time_s.min()), float(time_s.max())
    ymin = min(q_ana_nC.min(), q_emp_nC.min())
    ymax = max(q_ana_nC.max(), q_emp_nC.max())
    pad = 0.08 * (ymax - ymin if ymax != ymin else 1.0)
    ymin -= pad
    ymax += pad

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    pts_a = " ".join(f"{sx(float(t)):.2f},{sy(float(q)):.2f}" for t, q in zip(time_s, q_ana_nC))
    pts_b = " ".join(f"{sx(float(t)):.2f},{sy(float(q)):.2f}" for t, q in zip(time_s, q_emp_nC))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="18" font-weight="700">电荷源模型对比：解析余弦 vs 实测模板</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="12" fill="#55616d">Qpp = 650 nC | a = {ACCELERATION} m/s² | dwell = {DWELL_START_S*1e3:.0f}/{DWELL_STOP_S*1e3:.0f} ms</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333" stroke-width="1.2"/>',
        f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, Microsoft YaHei" font-size="13">Time (s)</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-family="Arial, Microsoft YaHei" font-size="13">Source charge Q(t) (nC)</text>',
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

    lines.append(f'<polyline points="{pts_a}" fill="none" stroke="#e67e22" stroke-width="2.2"/>')
    lines.append(f'<polyline points="{pts_b}" fill="none" stroke="#1a73e8" stroke-width="2.2" stroke-dasharray="8 4"/>')
    lines.append(f'<text x="{width - 420}" y="{top + 24}" font-family="Arial" font-size="13" fill="#e67e22">── Analytical cos(3πx/L)</text>')
    lines.append(f'<text x="{width - 420}" y="{top + 48}" font-family="Arial" font-size="13" fill="#1a73e8">- - Empirical template (pos-driven)</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 统计对比
# ---------------------------------------------------------------------------
def print_stats(label: str, isrc: np.ndarray, iload: np.ndarray, q: np.ndarray) -> None:
    print(f"  [{label}]")
    print(f"    Qpp          = {q.max() - q.min():.1f} nC")
    print(f"    Isrc peak    = {max(abs(isrc)):.3f} uA")
    print(f"    Isrc max+    = {isrc.max():.3f} uA")
    print(f"    Isrc max-    = {isrc.min():.3f} uA")
    print(f"    Iload peak   = {max(abs(iload)):.3f} uA")
    print(f"    Iload RMS    = {np.sqrt(np.mean(iload**2)):.3f} uA")
    # dwell 段电流
    t_acc = math.sqrt(STROKE_M / ACCELERATION)
    t_motion = 2.0 * t_acc
    t_dwell_start = t_motion
    t_dwell_end = t_motion + DWELL_STOP_S
    # 估算 dwell 段位置 (基于采样点数)
    n_total = len(isrc)
    t_total = 2.0 * t_motion + DWELL_START_S + DWELL_STOP_S
    idx_dwell_start = int(t_dwell_start / t_total * n_total)
    idx_dwell_end = int(t_dwell_end / t_total * n_total)
    dwell_isrc = isrc[idx_dwell_start:idx_dwell_end]
    dwell_iload = iload[idx_dwell_start:idx_dwell_end]
    print(f"    Isrc RMS (dwell) = {np.sqrt(np.mean(dwell_isrc**2)):.4f} uA")
    print(f"    Iload RMS (dwell)= {np.sqrt(np.mean(dwell_iload**2)):.4f} uA")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t, x, v = build_full_cycle_motion()
    print(f"周期 T = {t[-1]:.6f} s,  采样点 = {len(t)}")

    # --- 模型 A: 解析余弦 ---
    q_ana = source_charge_analytical(x)
    r_ana = solve_load(t, q_ana)
    isrc_ana = r_ana["source_current_A"] * 1e6
    iload_ana = r_ana["load_current_A"] * 1e6

    # --- 模型 B: 实测模板 ---
    q_emp = source_charge_empirical(t, x, v)
    r_emp = solve_load(t, q_emp)
    isrc_emp = r_emp["source_current_A"] * 1e6
    iload_emp = r_emp["load_current_A"] * 1e6

    # --- 统计 ---
    print("\n===== 模型对比统计 =====")
    print_stats("Analytical (cos)", isrc_ana, iload_ana, q_ana * 1e9)
    print_stats("Empirical (template)", isrc_emp, iload_emp, q_emp * 1e9)

    # --- 输出 ---
    write_comparison_csv(COMPARE_DIR / "comparison_data.csv", t, q_ana, r_ana, q_emp, r_emp)

    write_comparison_svg(
        COMPARE_DIR / "comparison_source_current.svg",
        t,
        "Analytical cos(3πx/L)", isrc_ana, "#e67e22",
        "Empirical template", isrc_emp, "#1a73e8",
        "源电流 (Source Current) 对比：解析模型 vs 实测模板",
        "Source Current (uA)",
    )

    write_comparison_svg(
        COMPARE_DIR / "comparison_load_current.svg",
        t,
        "Analytical cos(3πx/L)", iload_ana, "#e67e22",
        "Empirical template", iload_emp, "#9c27b0",
        "负载电流 (Load Current, R=1GΩ) 对比：解析模型 vs 实测模板",
        "Load Current (uA)",
    )

    write_charge_comparison_svg(
        COMPARE_DIR / "comparison_charge.svg",
        t, q_ana * 1e9, q_emp * 1e9,
    )

    print(f"\n输出目录: {COMPARE_DIR}")
    for f in sorted(COMPARE_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
