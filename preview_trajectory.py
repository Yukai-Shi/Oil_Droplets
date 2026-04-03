import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from config import (
    FixedGrid3x3Config,
    Gate3LevelConfig,
    get_logic_box_ranges,
    get_logic_port_coordinates,
    LayoutModeConfig,
    LogicBoxConfig,
    RenderSettingConfig,
    StokesCylinderConfig,
    TrainingSettingConfig,
)


def parse_layout_mode(layout_mode: str):
    mode = str(layout_mode)
    if mode.endswith("_inflow_u_fixed"):
        return mode[:-15], True
    if mode.endswith("_inflow_u"):
        return mode[:-9], True
    if mode.endswith("_inflow"):
        return mode[:-7], True
    return mode, False


def generate_path(path_type: str, steps: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, steps)
    t = np.linspace(0.0, 2.0 * np.pi, steps)
    gx, gy = FixedGrid3x3Config.grid_coords()
    c_center = np.array([gx[4], gy[4]], dtype=np.float32)
    c1 = np.array([gx[1], gy[1]], dtype=np.float32)
    c5 = np.array([gx[5], gy[5]], dtype=np.float32)

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

    elif path_type == "orbit_center":
        rad = 0.045
        xs = c_center[0] + rad * np.cos(t)
        ys = c_center[1] + rad * np.sin(t)

    elif path_type == "orbit_cyl_1":
        rad = 0.035
        xs = c1[0] + rad * np.cos(t)
        ys = c1[1] + rad * np.sin(t)

    elif path_type == "orbit_pair_1_5":
        mid = 0.5 * (c1 + c5)
        ax = 0.060
        ay = 0.040
        xs = mid[0] + ax * np.sin(t)
        ys = mid[1] + ay * np.sin(t) * np.cos(t)
    elif path_type == "square":
        center = np.array([0.0, 0.0], dtype=np.float32)
        half = 0.055
        n_exp = 4.0
        c = np.cos(t)
        s2 = np.sin(t)
        xs = center[0] + half * np.sign(c) * np.abs(c) ** (2.0 / n_exp)
        ys = center[1] + half * np.sign(s2) * np.abs(s2) ** (2.0 / n_exp)
    elif path_type == "square_hard":
        center = np.array([0.0, 0.0], dtype=np.float32)
        half = 0.05
        p0 = center + np.array([-half, +half], dtype=np.float32)
        p1 = center + np.array([+half, +half], dtype=np.float32)
        p2 = center + np.array([+half, -half], dtype=np.float32)
        p3 = center + np.array([-half, -half], dtype=np.float32)
        u = np.linspace(0.0, 4.0, steps)
        xs = np.zeros(steps, dtype=np.float32)
        ys = np.zeros(steps, dtype=np.float32)
        for i, ui in enumerate(u):
            if ui < 1.0:
                w = ui
                p = (1.0 - w) * p0 + w * p1
            elif ui < 2.0:
                w = ui - 1.0
                p = (1.0 - w) * p1 + w * p2
            elif ui < 3.0:
                w = ui - 2.0
                p = (1.0 - w) * p2 + w * p3
            else:
                w = min(ui - 3.0, 1.0)
                p = (1.0 - w) * p3 + w * p0
            xs[i] = float(p[0])
            ys[i] = float(p[1])
    elif path_type in {"gate3_top", "gate3_mid", "gate3_bottom"}:
        x0 = float(Gate3LevelConfig.START_X)
        xg = float(np.mean(Gate3LevelConfig.FIXED_X))
        x1 = float(Gate3LevelConfig.CHECK_X + 0.06)
        lane_map = {
            "gate3_top": float(Gate3LevelConfig.LANE_TARGET_Y[0]),
            "gate3_mid": float(Gate3LevelConfig.LANE_TARGET_Y[1]),
            "gate3_bottom": float(Gate3LevelConfig.LANE_TARGET_Y[2]),
        }
        yg = lane_map[path_type]
        y1 = yg
        y0 = float(Gate3LevelConfig.START_Y)

        def cubic_bezier(p0, p1, p2, p3, u):
            return (
                ((1.0 - u) ** 3)[:, None] * p0[None, :]
                + (3.0 * ((1.0 - u) ** 2) * u)[:, None] * p1[None, :]
                + (3.0 * (1.0 - u) * (u ** 2))[:, None] * p2[None, :]
                + (u ** 3)[:, None] * p3[None, :]
            )

        n1 = max(3, steps // 2 + 1)
        n2 = max(3, steps - n1 + 1)

        p0 = np.array([x0, y0], dtype=np.float32)
        p1 = np.array([x0 + 0.35 * (xg - x0), y0 + 0.15 * (yg - y0)], dtype=np.float32)
        p2 = np.array([x0 + 0.80 * (xg - x0), yg], dtype=np.float32)
        p3 = np.array([xg, yg], dtype=np.float32)

        q0 = p3
        q1 = np.array([xg + 0.25 * (x1 - xg), yg], dtype=np.float32)
        q2 = np.array([xg + 0.75 * (x1 - xg), y1], dtype=np.float32)
        q3 = np.array([x1, y1], dtype=np.float32)

        seg1 = cubic_bezier(p0, p1, p2, p3, np.linspace(0.0, 1.0, n1))
        seg2 = cubic_bezier(q0, q1, q2, q3, np.linspace(0.0, 1.0, n2))
        path = np.concatenate([seg1[:-1], seg2], axis=0)
        xs = path[:, 0]
        ys = path[:, 1]
    elif path_type == "logic_route":
        (x0_box, x1_box), (y0_box, y1_box) = get_logic_box_ranges()
        p = get_logic_port_coordinates()

        ports = {}
        for i, yy in enumerate(np.asarray(p["left_y"], dtype=np.float32)):
            ports[f"L{i}"] = ("left", np.array([x0_box, float(yy)], dtype=np.float32))
        for i, yy in enumerate(np.asarray(p["right_y"], dtype=np.float32)):
            ports[f"R{i}"] = ("right", np.array([x1_box, float(yy)], dtype=np.float32))
        ports["T0"] = ("top", np.array([float(p["top_x"]), y1_box], dtype=np.float32))
        ports["B0"] = ("bottom", np.array([float(p["bottom_x"]), y0_box], dtype=np.float32))

        src = str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")).upper()
        tgt = str(getattr(LogicBoxConfig, "TARGET_PORT", "R1")).upper()
        if src not in ports:
            src = "L1"
        if tgt not in ports:
            tgt = "R1"
        src_side, p0 = ports[src]
        tgt_side, p3 = ports[tgt]
        p0 = p0.astype(np.float32).copy()
        p3 = p3.astype(np.float32).copy()
        if src_side == "left":
            p0[0] = float(x0_box) + 1e-4

        def cubic_bezier(p0, p1, p2, p3, u):
            return (
                ((1.0 - u) ** 3)[:, None] * p0[None, :]
                + (3.0 * ((1.0 - u) ** 2) * u)[:, None] * p1[None, :]
                + (3.0 * (1.0 - u) * (u ** 2))[:, None] * p2[None, :]
                + (u ** 3)[:, None] * p3[None, :]
            )

        if tgt_side == "right":
            p1 = np.array([p0[0] + 0.30 * (x1_box - x0_box), p0[1]], dtype=np.float32)
            p2 = np.array([p0[0] + 0.80 * (x1_box - x0_box), p3[1]], dtype=np.float32)
        elif tgt_side == "top":
            p1 = np.array([p0[0] + 0.35 * (x1_box - x0_box), p0[1]], dtype=np.float32)
            p2 = np.array([p3[0], p0[1] + 0.75 * (y1_box - p0[1])], dtype=np.float32)
        elif tgt_side == "bottom":
            p1 = np.array([p0[0] + 0.35 * (x1_box - x0_box), p0[1]], dtype=np.float32)
            p2 = np.array([p3[0], p0[1] + 0.75 * (y0_box - p0[1])], dtype=np.float32)
        else:
            p1 = np.array([0.5 * (p0[0] + p3[0]), p0[1]], dtype=np.float32)
            p2 = np.array([0.5 * (p0[0] + p3[0]), p3[1]], dtype=np.float32)

        path = cubic_bezier(p0, p1, p2, p3, np.linspace(0.0, 1.0, steps))
        xs = path[:, 0]
        ys = path[:, 1]

    else:
        raise ValueError(f"Unknown path_type: {path_type}")

    return np.stack([xs, ys], axis=1).astype(np.float32)


def build_cylinder_layout(
    layout_mode: str,
    preview_radius: float,
    free_random_layout: bool,
    seed: int,
):
    base_mode, _ = parse_layout_mode(layout_mode)
    r_min = StokesCylinderConfig.MIN_R
    r_max = StokesCylinderConfig.MAX_R
    r = float(np.clip(preview_radius, r_min, r_max))

    if base_mode == "fixed_grid_3x3":
        x, y = FixedGrid3x3Config.grid_coords()
        rr = np.full_like(x, r, dtype=np.float32)
        return x, y, rr

    if base_mode == "free_layout":
        n = StokesCylinderConfig.NUM_CYLINDERS
        if not free_random_layout:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        rng = np.random.default_rng(seed)
        x = rng.uniform(StokesCylinderConfig.X_RANGE[0], StokesCylinderConfig.X_RANGE[1], size=n).astype(np.float32)
        y = rng.uniform(StokesCylinderConfig.Y_RANGE[0], StokesCylinderConfig.Y_RANGE[1], size=n).astype(np.float32)
        rr = np.full(n, r, dtype=np.float32)
        return x, y, rr

    if base_mode == "gate3_layout":
        fx = Gate3LevelConfig.FIXED_X.astype(np.float32)
        fy = Gate3LevelConfig.FIXED_Y.astype(np.float32)
        fr = Gate3LevelConfig.FIXED_R.astype(np.float32)
        if not free_random_layout:
            return fx, fy, fr

        n_var = max(0, StokesCylinderConfig.NUM_CYLINDERS - Gate3LevelConfig.NUM_FIXED)
        rng = np.random.default_rng(seed)
        x = rng.uniform(
            StokesCylinderConfig.X_RANGE[0],
            StokesCylinderConfig.X_RANGE[1],
            size=n_var,
        ).astype(np.float32)
        y = rng.uniform(
            StokesCylinderConfig.Y_RANGE[0],
            StokesCylinderConfig.Y_RANGE[1],
            size=n_var,
        ).astype(np.float32)
        rr = np.full(n_var, r, dtype=np.float32)
        return (
            np.concatenate([fx, x], axis=0),
            np.concatenate([fy, y], axis=0),
            np.concatenate([fr, rr], axis=0),
        )

    if base_mode == "logic_box_layout":
        n = StokesCylinderConfig.NUM_CYLINDERS
        if not free_random_layout:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        rng = np.random.default_rng(seed)
        (x0, x1), (y0, y1) = get_logic_box_ranges()
        x = rng.uniform(x0, x1, size=n).astype(np.float32)
        y = rng.uniform(y0, y1, size=n).astype(np.float32)
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

    base_mode, _ = parse_layout_mode(layout_mode)
    if base_mode == "logic_box_layout":
        (bx0, bx1), (by0, by1) = get_logic_box_ranges()
        p = get_logic_port_coordinates()
        ax.add_patch(
            plt.Rectangle(
                (bx0, by0),
                bx1 - bx0,
                by1 - by0,
                fill=False,
                linestyle="-",
                linewidth=1.2,
                edgecolor="#111827",
                alpha=0.9,
                label="logic box",
            )
        )
        for i, yy in enumerate(np.asarray(p["left_y"], dtype=np.float32)):
            ax.plot([bx0], [float(yy)], marker="o", ms=4, color="#1f2937")
            ax.text(bx0 - 0.01, float(yy), f"L{i}", fontsize=7, ha="right", va="center")
        for i, yy in enumerate(np.asarray(p["right_y"], dtype=np.float32)):
            ax.plot([bx1], [float(yy)], marker="o", ms=4, color="#1f2937")
            ax.text(bx1 + 0.01, float(yy), f"R{i}", fontsize=7, ha="left", va="center")
        ax.plot([float(p["top_x"])], [by1], marker="o", ms=4, color="#1f2937")
        ax.text(float(p["top_x"]), by1 + 0.01, "T0", fontsize=7, ha="center", va="bottom")
        ax.plot([float(p["bottom_x"])], [by0], marker="o", ms=4, color="#1f2937")
        ax.text(float(p["bottom_x"]), by0 - 0.01, "B0", fontsize=7, ha="center", va="top")

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
        help="soft_snake2 | soft_snake2_easy | bezier | bend2 | bend | line | orbit_center | orbit_cyl_1 | orbit_pair_1_5 | square | square_hard | gate3_top | gate3_mid | gate3_bottom | logic_route",
    )
    parser.add_argument(
        "--layout-mode",
        default=LayoutModeConfig.LAYOUT_MODE,
        help="free_layout | fixed_grid_3x3 | gate3_layout | logic_box_layout (and *_inflow)",
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
    base_mode, _ = parse_layout_mode(args.layout_mode)
    if base_mode == "free_layout" and not args.free_random_layout:
        print("free_layout default reset has no fixed cylinder positions; use --free-random-layout to preview one sample.")
    if base_mode == "gate3_layout" and not args.free_random_layout:
        print("gate3_layout preview shows only 3 fixed gate cylinders; add --free-random-layout to sample extra design cylinders.")
    if base_mode == "logic_box_layout" and not args.free_random_layout:
        print("logic_box_layout preview has no fixed cylinders by default; add --free-random-layout to sample internal cylinders.")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
