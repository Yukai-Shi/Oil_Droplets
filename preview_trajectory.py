import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from config import (
    FixedGrid3x3Config,
    LayoutModeConfig,
    RenderSettingConfig,
    StokesCylinderConfig,
    TrainingSettingConfig,
)


def generate_path(path_type: str, steps: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, steps)

    if path_type == "soft_snake2":
        x0, x1 = -0.025, 0.055
        y0, y1 = 0.060, -0.020
        xs = x0 + (x1 - x0) * s
        ys = y0 + (y1 - y0) * s - 0.010 * np.sin(2.0 * np.pi * s)

    elif path_type == "soft_snake2_easy":
        x0, x1 = -0.025, 0.050
        y0, y1 = 0.060, -0.015
        xs = x0 + (x1 - x0) * s
        ys = y0 + (y1 - y0) * s - 0.006 * np.sin(2.0 * np.pi * s)

    elif path_type == "bezier":
        p0 = np.array([-0.025, 0.060], dtype=np.float32)
        p1 = np.array([0.000, 0.050], dtype=np.float32)
        p2 = np.array([0.055, -0.010], dtype=np.float32)
        xs = (1 - s) ** 2 * p0[0] + 2 * (1 - s) * s * p1[0] + s ** 2 * p2[0]
        ys = (1 - s) ** 2 * p0[1] + 2 * (1 - s) * s * p1[1] + s ** 2 * p2[1]

    elif path_type == "bend2":
        x0, x1 = -0.025, -0.070
        y0, y1 = 0.060, -0.010
        bend = 0.080
        xs = x0 + (x1 - x0) * s
        ys = y0 + (y1 - y0) * s + bend * s * (1.0 - s) * np.sin(2.0 * np.pi * s)

    elif path_type == "bend":
        x0, x1 = -0.025, -0.070
        y0, y1 = 0.060, -0.010
        xs = x0 + (x1 - x0) * s
        ys = y0 + (y1 - y0) * s + 0.060 * s * (1.0 - s)

    elif path_type == "line":
        x0, x1 = -0.025, 0.050
        y0, y1 = 0.060, -0.010
        xs = x0 + (x1 - x0) * s
        ys = y0 + (y1 - y0) * s

    else:
        raise ValueError(f"Unknown path_type: {path_type}")

    return np.stack([xs, ys], axis=1).astype(np.float32)


def build_cylinder_layout(
    layout_mode: str,
    preview_radius: float,
    free_random_layout: bool,
    seed: int,
):
    r_min = StokesCylinderConfig.MIN_R
    r_max = StokesCylinderConfig.MAX_R
    r = float(np.clip(preview_radius, r_min, r_max))

    if layout_mode == "fixed_grid_3x3":
        x, y = FixedGrid3x3Config.grid_coords()
        rr = np.full_like(x, r, dtype=np.float32)
        return x, y, rr

    if layout_mode == "free_layout":
        n = StokesCylinderConfig.NUM_CYLINDERS
        if not free_random_layout:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        rng = np.random.default_rng(seed)
        x = rng.uniform(StokesCylinderConfig.X_RANGE[0], StokesCylinderConfig.X_RANGE[1], size=n).astype(np.float32)
        y = rng.uniform(StokesCylinderConfig.Y_RANGE[0], StokesCylinderConfig.Y_RANGE[1], size=n).astype(np.float32)
        rr = np.full(n, r, dtype=np.float32)
        return x, y, rr

    raise ValueError(f"Unknown layout_mode: {layout_mode}")


def summarize_path(path: np.ndarray) -> str:
    seg = path[1:] - path[:-1]
    length = float(np.sum(np.linalg.norm(seg, axis=1)))
    start = path[0]
    end = path[-1]
    return (
        f"start=({start[0]:.4f}, {start[1]:.4f}) | "
        f"end=({end[0]:.4f}, {end[1]:.4f}) | "
        f"points={len(path)} | "
        f"arc_length~{length:.4f}"
    )


def plot_scene(
    path: np.ndarray,
    path_type: str,
    layout_mode: str,
    cyl_x: np.ndarray,
    cyl_y: np.ndarray,
    cyl_r: np.ndarray,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120)

    ax.set_xlim(*RenderSettingConfig.X_LIM)
    ax.set_ylim(*RenderSettingConfig.Y_LIM)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Visualize optimization bounds inside render bounds.
    x0, x1 = StokesCylinderConfig.X_RANGE
    y0, y1 = StokesCylinderConfig.Y_RANGE
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            linestyle="--",
            linewidth=1.0,
            edgecolor="#666666",
            alpha=0.7,
            label="search range",
        )
    )

    ax.plot(path[:, 0], path[:, 1], "-", lw=2.0, color="#1f77b4", label=f"path: {path_type}")
    ax.scatter(path[0, 0], path[0, 1], s=44, color="green", label="start", zorder=4)
    ax.scatter(path[-1, 0], path[-1, 1], s=44, color="red", label="end", zorder=4)

    if len(cyl_x) > 0:
        for i, (x, y, r) in enumerate(zip(cyl_x, cyl_y, cyl_r)):
            center_style = dict(s=12, color="black", alpha=0.8, zorder=5)
            ax.scatter(float(x), float(y), **center_style)
            ax.text(float(x), float(y), f"{i}", fontsize=7, ha="left", va="bottom")

            if r > 0:
                circ = plt.Circle(
                    (float(x), float(y)),
                    float(r),
                    color="#2ca02c",
                    alpha=0.25,
                    ec="black",
                    lw=0.8,
                    zorder=3,
                )
                ax.add_patch(circ)

    ax.set_title(f"Trajectory + Cylinder Layout | mode={layout_mode}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Preview trajectory in environment with cylinder layout.")
    parser.add_argument(
        "--path-type",
        default=TrainingSettingConfig.PATH_TYPE,
        help="soft_snake2 | soft_snake2_easy | bezier | bend2 | bend | line",
    )
    parser.add_argument(
        "--layout-mode",
        default=LayoutModeConfig.LAYOUT_MODE,
        help="free_layout | fixed_grid_3x3",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=TrainingSettingConfig.EPISODE_LENGTH + 1,
        help="Number of trajectory points",
    )
    parser.add_argument(
        "--preview-radius",
        type=float,
        default=0.010,
        help="Preview radius for drawn cylinders",
    )
    parser.add_argument(
        "--free-random-layout",
        action="store_true",
        help="When mode=free_layout, draw one random layout sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for free random layout preview",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Default: previews/scene_<path_type>_<layout_mode>.png",
    )
    args = parser.parse_args()

    os.makedirs("previews", exist_ok=True)
    out_path = args.out or os.path.join(
        "previews",
        f"scene_{args.path_type}_{args.layout_mode}.png",
    )

    path = generate_path(args.path_type, args.steps)
    cyl_x, cyl_y, cyl_r = build_cylinder_layout(
        layout_mode=args.layout_mode,
        preview_radius=args.preview_radius,
        free_random_layout=args.free_random_layout,
        seed=args.seed,
    )
    plot_scene(
        path=path,
        path_type=args.path_type,
        layout_mode=args.layout_mode,
        cyl_x=cyl_x,
        cyl_y=cyl_y,
        cyl_r=cyl_r,
        save_path=out_path,
    )

    print(summarize_path(path))
    print(
        f"layout_mode={args.layout_mode} | cylinders={len(cyl_x)} | "
        f"preview_radius={float(args.preview_radius):.4f}"
    )
    if args.layout_mode == "free_layout" and not args.free_random_layout:
        print("free_layout default reset has no fixed cylinder positions; use --free-random-layout to preview one sample.")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
