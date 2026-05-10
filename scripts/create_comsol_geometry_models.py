from __future__ import annotations

from pathlib import Path
import argparse

import mph


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "comsol_models"
OUTDIR.mkdir(parents=True, exist_ok=True)


def add_rect(geom, tag: str, pos: tuple[float, float], size: tuple[float, float]) -> None:
    geom.create(tag, "Rectangle")
    geom.feature(tag).set("pos", [f"{pos[0]}", f"{pos[1]}"])
    geom.feature(tag).set("size", [f"{size[0]}", f"{size[1]}"])


def add_circle(geom, tag: str, pos: tuple[float, float], radius: float) -> None:
    geom.create(tag, "Circle")
    geom.feature(tag).set("pos", [f"{pos[0]}", f"{pos[1]}"])
    geom.feature(tag).set("r", f"{radius}")


def build_geometry(client: mph.Client, n: int, sigma_uc_per_m2: float, suffix: str) -> Path:
    electrode_width = 16e-3
    gap = 5e-3
    substrate_thickness = 1.6e-3
    electrode_thickness = 70e-6
    ball_radius = 2e-3
    air_height = 12e-3
    margin = 16e-3

    total_length = n * electrode_width + (n - 1) * gap
    xmin = -margin
    width = total_length + 2 * margin

    model = client.create(f"rolling_teng_N{n}_geometry")
    jm = model.java
    jm.label(f"rolling_teng_N{n}_geometry.mph")
    jm.param().set("Nelec", str(n), "Number of alternating grid electrodes")
    jm.param().set("we", f"{electrode_width}[m]", "Electrode width")
    jm.param().set("ge", f"{gap}[m]", "Insulating gap")
    jm.param().set("r_ball", f"{ball_radius}[m]", "PTFE ball radius")
    jm.param().set("sigma_t", f"{sigma_uc_per_m2}[uc/m^2]", "Shared calibrated equivalent triboelectric surface charge density")
    jm.param().set("Rload", "100[Mohm]", "External load resistance")
    jm.param().set("vscan", "0.1[m/s]", "Constant scan speed")

    jm.component().create("comp1", True)
    comp = jm.component("comp1")
    comp.geom().create("geom1", 2)
    geom = comp.geom("geom1")
    geom.lengthUnit("m")

    add_rect(geom, "substrate", (xmin, -substrate_thickness), (width, substrate_thickness))
    add_rect(geom, "air", (xmin, 0.0), (width, air_height))

    for i in range(n):
        x0 = i * (electrode_width + gap)
        tag = f"elec{i + 1}"
        add_rect(geom, tag, (x0, 0.0), (electrode_width, electrode_thickness))
        geom.feature(tag).label(f"{'A' if i % 2 == 0 else 'B'} electrode {i + 1}")

    # Representative ball group centered over the first electrode for visual/model setup.
    offsets = [(-1.5 + k) * 2 * ball_radius for k in range(4)]
    group_center = electrode_width / 2
    for k, off in enumerate(offsets, start=1):
        add_circle(geom, f"ball{k}", (group_center + off, ball_radius + electrode_thickness), ball_radius)

    geom.run()

    # Add physics interface as a starting point for refined COMSOL electrostatic solves.
    comp.physics().create("es", "Electrostatics", "geom1")
    comp.mesh().create("mesh1")
    comp.mesh("mesh1").autoMeshSize(4)
    comp.mesh("mesh1").run()
    jm.study().create("std1")
    jm.study("std1").create("stat", "Stationary")
    jm.study("std1").feature("stat").activate("es", True)

    out = OUTDIR / f"rolling_teng_N{n}_geometry_study{suffix}.mph"
    model.save(out)
    client.remove(model)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma-uc-per-m2", type=float, default=-50.0)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    client = mph.Client(port=2036, host="localhost")
    saved = [build_geometry(client, n, args.sigma_uc_per_m2, args.suffix) for n in (2, 4, 6)]
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
