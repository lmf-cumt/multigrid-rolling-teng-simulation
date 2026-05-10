from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import mph


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results_comsol_field"
DEPTH = 16e-3
PARALLEL_UNITS = 16 * 9
Q2_MEASURED = 380e-9
Q4_MEASURED = 650e-9
R_LOAD = 1e9
PCB_THICKNESS = 1e-3
ELECTRODE_W = 16e-3
GAP = 5e-3
BALL_R = 2e-3
BALL_COUNT = 4
ACCELERATION = 5.0
STROKE = 100e-3
DWELL = 30e-3


def add_rect(geom, tag: str, pos: tuple[float, float], size: tuple[float, float]) -> None:
    geom.create(tag, "Rectangle")
    geom.feature(tag).set("pos", [str(pos[0]), str(pos[1])])
    geom.feature(tag).set("size", [str(size[0]), str(size[1])])
    geom.feature(tag).set("selresult", "on")
    geom.feature(tag).set("selresultshow", "all")


def add_circle(geom, tag: str, pos: tuple[float, float], radius: float) -> None:
    geom.create(tag, "Circle")
    geom.feature(tag).set("pos", [str(pos[0]), str(pos[1])])
    geom.feature(tag).set("r", str(radius))
    geom.feature(tag).set("selresult", "on")
    geom.feature(tag).set("selresultshow", "all")


def build_and_solve(client: mph.Client, electrode_count: int, group_x: float, index: int) -> float:
    total_len = electrode_count * ELECTRODE_W + (electrode_count - 1) * GAP
    margin = 20e-3
    air_h = 18e-3
    model = client.create(f"field_N{electrode_count}_{index}")
    jm = model.java
    jm.param().set("sigma_ptfe", "-50e-6[C/m^2]", "Fixed PTFE equivalent surface charge density")
    jm.component().create("comp1", True)
    comp = jm.component("comp1")
    comp.geom().create("geom1", 2)
    geom = comp.geom("geom1")
    geom.lengthUnit("m")

    xmin = -margin
    width = total_len + 2 * margin
    add_rect(geom, "air", (xmin, -PCB_THICKNESS), (width, air_h + PCB_THICKNESS))
    add_rect(geom, "pcb", (xmin, -PCB_THICKNESS), (width, PCB_THICKNESS))
    for i in range(electrode_count):
        x0 = i * (ELECTRODE_W + GAP)
        add_rect(geom, f"elec{i+1}", (x0, 0.0), (ELECTRODE_W, 70e-6))

    offsets = (np.arange(BALL_COUNT) - (BALL_COUNT - 1) / 2.0) * (2.0 * BALL_R)
    for k, off in enumerate(offsets, start=1):
        add_circle(geom, f"ball{k}", (group_x + float(off), BALL_R + 0.25e-3), BALL_R)
    geom.run()

    comp.physics().create("es", "Electrostatics", "geom1")
    es = comp.physics("es")
    expr_terms = []
    for i in range(electrode_count):
        tag = f"elec{i+1}"
        bc_tag = f"g{i+1}"
        es.create(bc_tag, "Ground", 1)
        es.feature(bc_tag).selection().named(f"geom1_{tag}_bnd")
        if i % 2 == 0:
            int_tag = f"intA{i+1}"
            comp.cpl().create(int_tag, "Integration")
            comp.cpl(int_tag).selection().named(f"geom1_{tag}_bnd")
            expr_terms.append(f"{int_tag}(es.nD)")
    for k in range(1, BALL_COUNT + 1):
        es.create(f"scd{k}", "SurfaceChargeDensity", 1)
        es.feature(f"scd{k}").selection().named(f"geom1_ball{k}_bnd")
        es.feature(f"scd{k}").set("rhoqs", "sigma_ptfe")

    comp.mesh().create("mesh1")
    comp.mesh("mesh1").autoMeshSize(6)
    comp.mesh("mesh1").run()
    jm.study().create("std1")
    jm.study("std1").create("stat", "Stationary")
    jm.study("std1").feature("stat").activate("es", True)
    model.solve()
    expr = "+".join(expr_terms)
    raw_per_depth = float(model.evaluate(expr))
    client.remove(model)
    return raw_per_depth * DEPTH * PARALLEL_UNITS


def build_motion(samples: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    t_acc = math.sqrt(STROKE / ACCELERATION)
    t_motion = 2.0 * t_acc
    duration = DWELL + t_motion + DWELL + t_motion + DWELL
    t = np.linspace(0.0, duration, samples)
    x = np.zeros_like(t)
    t0 = DWELL
    t1 = t0 + t_acc
    t2 = t0 + t_motion
    t3 = t2 + DWELL
    t4 = t3 + t_acc
    t5 = t3 + t_motion
    v_peak = ACCELERATION * t_acc
    acc = (t >= t0) & (t < t1)
    tau = t[acc] - t0
    x[acc] = 0.5 * ACCELERATION * tau**2
    dec = (t >= t1) & (t < t2)
    tau = t[dec] - t1
    x[dec] = 0.5 * STROKE + v_peak * tau - 0.5 * ACCELERATION * tau**2
    x[(t >= t2) & (t < t3)] = STROKE
    racc = (t >= t3) & (t < t4)
    tau = t[racc] - t3
    x[racc] = STROKE - 0.5 * ACCELERATION * tau**2
    rdec = (t >= t4) & (t < t5)
    tau = t[rdec] - t4
    x[rdec] = 0.5 * STROKE - v_peak * tau + 0.5 * ACCELERATION * tau**2
    return t, x


def load_response(t: np.ndarray, q: np.ndarray, resistance: float) -> dict[str, np.ndarray]:
    current = np.gradient(q, t)
    voltage = resistance * current
    power = voltage * current
    power = np.maximum(power, 0.0)
    energy = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(t))])
    return {"current_A": current, "voltage_V": voltage, "power_W": power, "energy_J": energy}


def svg_plot(path: Path, title: str, ylabel: str, t: np.ndarray, series: list[tuple[str, np.ndarray, str]]) -> None:
    width, height = 860, 460
    left, right, top, bottom = 88, 28, 42, 64
    xmin, xmax = float(t.min()), float(t.max())
    ymin = min(float(y.min()) for _, y, _ in series)
    ymax = max(float(y.max()) for _, y, _ in series)
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    pad = 0.1 * (ymax - ymin)
    ymin -= pad
    ymax += pad
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * (width - left - right)
    sy = lambda y: top + (ymax - y) / (ymax - ymin) * (height - top - bottom)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Microsoft YaHei, Arial" font-size="17" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="12">Time (s)</text>',
        f'<text x="17" y="{height/2}" text-anchor="middle" transform="rotate(-90 17 {height/2})" font-family="Arial" font-size="12">{ylabel}</text>',
    ]
    for frac in np.linspace(0, 1, 6):
        yy = top + frac * (height - top - bottom)
        val = ymax - frac * (ymax - ymin)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e1e5ea"/>')
        lines.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{val:.2e}</text>')
    for idx, (name, y, color) in enumerate(series):
        pts = " ".join(f"{sx(float(a)):.2f},{sy(float(b)):.2f}" for a, b in zip(t, y))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        lines.append(f'<text x="{width-right-160}" y="{top+28+20*idx}" font-family="Arial" font-size="12" fill="{color}">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=15)
    parser.add_argument("--outdir", default=str(OUTDIR))
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    client = mph.Client(port=2036, host="localhost")

    raw_rows = []
    raw_by_n = {}
    for n in (2, 4, 6):
        total_len = n * ELECTRODE_W + (n - 1) * GAP
        x_start = ELECTRODE_W / 2
        x_end = total_len - ELECTRODE_W / 2
        positions = np.linspace(x_start, x_end, args.positions)
        q_raw = []
        for idx, x in enumerate(positions):
            q = build_and_solve(client, n, float(x), idx)
            q_raw.append(q)
            raw_rows.append({"electrode_count": n, "position_m": x, "raw_QA_C": q})
        raw_by_n[n] = (positions, np.array(q_raw))

    with (outdir / "comsol_raw_position_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["electrode_count", "position_m", "raw_QA_C"])
        writer.writeheader()
        writer.writerows(raw_rows)

    q2_pp = float(np.ptp(raw_by_n[2][1]))
    scale = Q2_MEASURED / q2_pp if q2_pp else 1.0
    t, x_motion = build_motion()
    colors = {2: "#2f6f9f", 4: "#34a853", 6: "#b26a00"}
    q_series = []
    i_series = []
    p_series = []
    e_series = []
    summary = []
    for n in (2, 4, 6):
        positions, q_raw = raw_by_n[n]
        q_scaled_pos = (q_raw - q_raw[0]) * scale
        total_len = n * ELECTRODE_W + (n - 1) * GAP
        x_start = ELECTRODE_W / 2
        x_end = total_len - ELECTRODE_W / 2
        x_pos = x_start + np.clip(x_motion / STROKE, 0, 1) * (x_end - x_start)
        q_t = np.interp(x_pos, positions, q_scaled_pos)
        response = load_response(t, q_t, R_LOAD)
        q_pp = float(np.ptp(q_scaled_pos))
        summary.append(
            {
                "electrode_count": n,
                "comsol_scaled_transfer_charge_nC": q_pp * 1e9,
                "current_peak_uA_1G": float(np.max(np.abs(response["current_A"]))) * 1e6,
                "voltage_peak_V_1G": float(np.max(np.abs(response["voltage_V"]))),
                "power_mean_mW_1G": float(np.mean(response["power_W"])) * 1e3,
                "energy_uJ_1G": float(response["energy_J"][-1]) * 1e6,
                "raw_charge_peak_to_peak_C": float(np.ptp(q_raw)),
            }
        )
        with (outdir / f"comsol_N{n}_timeseries_1G.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "position_m", "charge_C", "current_A", "voltage_V", "power_W", "energy_J"])
            for row in zip(t, x_pos, q_t, response["current_A"], response["voltage_V"], response["power_W"], response["energy_J"]):
                writer.writerow([f"{float(v):.12e}" for v in row])
        q_series.append((f"N={n}", q_t * 1e9, colors[n]))
        i_series.append((f"N={n}", response["current_A"] * 1e6, colors[n]))
        p_series.append((f"N={n}", response["power_W"] * 1e3, colors[n]))
        e_series.append((f"N={n}", response["energy_J"] * 1e6, colors[n]))

    with (outdir / "comsol_field_summary_1G.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    metadata = {
        "solver": "COMSOL Electrostatics stationary position sweep",
        "charge_density": "-50e-6 C/m^2 fixed on PTFE ball boundaries",
        "scale": "single global scale from N=2 measured transfer charge 380 nC",
        "N4_measured_transfer_charge_C": Q4_MEASURED,
        "load_resistance_ohm": R_LOAD,
        "positions_per_structure": args.positions,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    svg_plot(outdir / "comsol_charge_comparison_1G.svg", "COMSOL field-derived charge, full cycle", "Charge (nC)", t, q_series)
    svg_plot(outdir / "comsol_current_comparison_1G.svg", "COMSOL field-derived current, full cycle", "Current (uA)", t, i_series)
    svg_plot(outdir / "comsol_power_comparison_1G.svg", "COMSOL field-derived power, R=1 Gohm", "Power (mW)", t, p_series)
    svg_plot(outdir / "comsol_energy_comparison_1G.svg", "COMSOL field-derived energy, R=1 Gohm", "Energy (uJ)", t, e_series)


if __name__ == "__main__":
    main()
