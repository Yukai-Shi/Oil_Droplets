import argparse
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from config import FixedGrid3x3Config, RenderSettingConfig, StokesCylinderConfig
from utils.calc import calculate_velocity_grid


def parse_radii(radii_text: Optional[str], default_r: float) -> np.ndarray:
    if not radii_text:
        return np.full(FixedGrid3x3Config.NUM_CYLINDERS, default_r, dtype=np.float32)

    parts = [p.strip() for p in radii_text.split(",") if p.strip()]
    values = np.array([float(v) for v in parts], dtype=np.float32)
    if values.size != FixedGrid3x3Config.NUM_CYLINDERS:
        raise ValueError(
            f"--radii must have exactly {FixedGrid3x3Config.NUM_CYLINDERS} values, got {values.size}"
        )
    return values


def draw_rotation_arrow(ax, cx: float, cy: float, rr: float, omega: float):
    if rr <= 0:
        return
    # Draw a small tangential arrow near cylinder boundary.
    if omega >= 0:
        theta0, theta1 = -30.0, 30.0   # CCW
    else:
        theta0, theta1 = 30.0, -30.0   # CW

    t0 = np.deg2rad(theta0)
    t1 = np.deg2rad(theta1)
    rad = rr * 0.95
    p0 = (cx + rad * np.cos(t0), cy + rad * np.sin(t0))
    p1 = (cx + rad * np.cos(t1), cy + rad * np.sin(t1))
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=1.0,
        color="#0f172a",
        zorder=4,
    )
    ax.add_patch(arr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate streamline plot for 3x3 fixed cylinders with uniform inflow."
    )
    parser.add_argument(
        "--uin",
        type=float,
        default=0.03,
        help="Uniform inflow speed in +x direction (m/s equivalent unit).",
    )
    parser.add_argument(
        "--vin",
        type=float,
        default=0.0,
        help="Uniform inflow speed in y direction.",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=StokesCylinderConfig.FIXED_OMEGA,
        help="Cylinder angular speed used for all cylinders (same direction, same speed).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.012,
        help="Default radius for all 9 cylinders when --radii is not provided.",
    )
    parser.add_argument(
        "--radii",
        type=str,
        default="",
        help="Comma-separated 9 radii (override --radius). Example: 0.01,0.012,...",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=120,
        help="Grid resolution for streamline computation.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="previews/flow_3x3_uniform_inflow.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    x_cyl, y_cyl = FixedGrid3x3Config.grid_coords()
    r_cyl = parse_radii(args.radii, args.radius)
    # Enforce same-direction, same-speed rotation for all cylinders.
    omega_shared = float(args.omega)
    omegas = np.full(FixedGrid3x3Config.NUM_CYLINDERS, omega_shared, dtype=np.float32)

    x = np.linspace(RenderSettingConfig.X_LIM[0], RenderSettingConfig.X_LIM[1], args.grid)
    y = np.linspace(RenderSettingConfig.Y_LIM[0], RenderSettingConfig.Y_LIM[1], args.grid)
    X, Y = np.meshgrid(x, y)

    U, V = calculate_velocity_grid(X, Y, x_cyl, y_cyl, r_cyl, omegas)

    # Add uniform inflow (requested "horizontal inflow").
    U = U + float(args.uin)
    V = V + float(args.vin)

    # Mask cylinder interior so streamlines do not run through solids.
    mask = np.zeros_like(X, dtype=bool)
    for cx, cy, rr in zip(x_cyl, y_cyl, r_cyl):
        if rr <= 0:
            continue
        mask |= ((X - cx) ** 2 + (Y - cy) ** 2) <= float(rr) ** 2
    U = np.where(mask, np.nan, U)
    V = np.where(mask, np.nan, V)

    speed = np.sqrt(U ** 2 + V ** 2)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=150)
    ax.set_aspect("equal")
    ax.set_xlim(*RenderSettingConfig.X_LIM)
    ax.set_ylim(*RenderSettingConfig.Y_LIM)
    ax.grid(True, alpha=0.12, linestyle=":")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    vmax = np.nanpercentile(speed, 95)
    ax.streamplot(
        X,
        Y,
        U,
        V,
        color=speed,
        cmap="Blues",
        linewidth=1.0,
        arrowsize=0.9,
        density=1.2,
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(vmax, 1e-6)),
    )

    for i, (cx, cy, rr) in enumerate(zip(x_cyl, y_cyl, r_cyl)):
        if rr <= 0:
            continue
        circle = plt.Circle((cx, cy), rr, color="#4CAF50", alpha=0.35, ec="black", lw=0.8)
        ax.add_patch(circle)
        draw_rotation_arrow(ax, float(cx), float(cy), float(rr), omega_shared)
        ax.text(cx, cy, str(i), ha="center", va="center", fontsize=8, color="black")

    ax.set_title(
        f"3x3 Cylinders + Uniform Inflow\n"
        f"Uin=({args.uin:.3f}, {args.vin:.3f}), omega(all)={omega_shared:.3f}"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)

    print(args.out)
    direction = "CCW" if omega_shared >= 0 else "CW"
    print(f"rotation: same direction={direction}, same speed={abs(omega_shared):.6f}")


if __name__ == "__main__":
    main()
