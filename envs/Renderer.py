import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from config import *
from utils.calc import calculate_velocity_grid


class FluidRenderer:
    def __init__(
        self,
        width: int = RenderSettingConfig.WIDTH,
        height: int = RenderSettingConfig.HEIGHT,
        dpi: int = RenderSettingConfig.DPI
    ):
        self.fig, self.ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.ax.set_aspect("equal")

        self.grid_res = 30
        x = np.linspace(RenderSettingConfig.X_LIM[0], RenderSettingConfig.X_LIM[1], self.grid_res)
        y = np.linspace(RenderSettingConfig.Y_LIM[0], RenderSettingConfig.Y_LIM[1], self.grid_res)
        self.X, self.Y = np.meshgrid(x, y)

    def _setup_axes(self, xlim, ylim):
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.grid(True, alpha=0.1, linestyle=":")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

    def render(
        self,
        scene: dict,
        follower_path: Optional[List[Tuple[float, float]]] = None,
        target_pos: Optional[np.ndarray] = None,
        target_path: Optional[List[Tuple[float, float]]] = None,
        title: Optional[str] = None,
        draw_flow: bool = False
    ) -> np.ndarray:
        self._setup_axes(RenderSettingConfig.X_LIM, RenderSettingConfig.Y_LIM)

        cyl_x = np.array(scene["cylinders"]["x"], dtype=np.float32)
        cyl_y = np.array(scene["cylinders"]["y"], dtype=np.float32)
        cyl_r = np.array(scene["cylinders"]["r"], dtype=np.float32)
        omegas = np.array(scene["cylinders"]["omegas"], dtype=np.float32)
        fixed_count = int(scene["cylinders"].get("fixed_count", 0))
        inflow = scene.get("inflow", {})
        inflow_u = float(inflow.get("u", 0.0))
        inflow_v = float(inflow.get("v", 0.0))

        if draw_flow and len(cyl_x) > 0:
            U, V = calculate_velocity_grid(self.X, self.Y, cyl_x, cyl_y, cyl_r, omegas)
            U = U + inflow_u
            V = V + inflow_v
            speed = np.sqrt(U ** 2 + V ** 2)

            if np.max(speed) > 1e-6:
                self.ax.streamplot(
                    self.X,
                    self.Y,
                    U,
                    V,
                    color=speed,
                    cmap="Blues",
                    linewidth=1.0,
                    arrowsize=0.8,
                    density=1.0,
                    zorder=0,
                    norm=matplotlib.colors.Normalize(
                        vmin=0,
                        vmax=np.max(speed) * 0.8
                    )
                )

        if abs(inflow_u) > 1e-12 or abs(inflow_v) > 1e-12:
            ax_x0, ax_x1 = RenderSettingConfig.X_LIM
            ax_y0, ax_y1 = RenderSettingConfig.Y_LIM
            cx = ax_x0 + 0.10 * (ax_x1 - ax_x0)
            cy = ax_y1 - 0.10 * (ax_y1 - ax_y0)
            scale = 0.06
            self.ax.arrow(
                cx,
                cy,
                inflow_u * scale,
                inflow_v * scale,
                head_width=0.006,
                head_length=0.008,
                fc="#111827",
                ec="#111827",
                length_includes_head=True,
                zorder=6,
            )
            self.ax.text(
                cx,
                cy + 0.01,
                f"inflow=({inflow_u:.3f},{inflow_v:.3f})",
                fontsize=8,
                color="#111827",
                ha="left",
                va="bottom",
                zorder=6,
            )

        for i in range(len(cyl_x)):
            if cyl_r[i] <= 0:
                continue

            color = "green" if omegas[i] > 0 else "red"
            if i < fixed_count:
                color = "#2563eb"
            circ = plt.Circle(
                (cyl_x[i], cyl_y[i]),
                cyl_r[i],
                color=color,
                alpha=0.35 if i >= fixed_count else 0.45,
                ec="black" if i >= fixed_count else "#1e3a8a",
                lw=0.8 if i >= fixed_count else 1.0,
                zorder=2
            )
            self.ax.add_patch(circ)
            self.ax.text(
                cyl_x[i],
                cyl_y[i],
                f"{i}",
                fontsize=6,
                color="black",
                ha="center",
                va="center",
                weight="bold"
            )

        if follower_path is not None and len(follower_path) > 1:
            xs = [p[0] for p in follower_path]
            ys = [p[1] for p in follower_path]
            self.ax.plot(xs, ys, "-", color="orange", lw=1.5, alpha=0.9)

        if target_path is not None and len(target_path) > 1:
            xs = [p[0] for p in target_path]
            ys = [p[1] for p in target_path]
            self.ax.plot(xs, ys, "--", color="green", lw=1.2, alpha=0.9)

        gate = scene.get("gate", None)
        if gate is not None:
            cx = float(gate.get("check_x", 0.0))
            target_y = gate.get("target_y", [])
            seeds = gate.get("seed_points", [])
            self.ax.axvline(cx, color="#6b7280", linestyle=":", lw=1.0, alpha=0.8, zorder=1)
            for yy in target_y:
                self.ax.plot(cx, float(yy), marker="x", color="#111827", ms=5, zorder=5)
            for sp in seeds:
                self.ax.plot(float(sp[0]), float(sp[1]), marker="o", color="#374151", ms=3, zorder=5)

        logic_box = scene.get("logic_box", None)
        if logic_box is not None:
            xr = logic_box.get("x_range", [0.0, 0.0])
            yr = logic_box.get("y_range", [0.0, 0.0])
            bx0, bx1 = float(xr[0]), float(xr[1])
            by0, by1 = float(yr[0]), float(yr[1])
            if bool(logic_box.get("show_box", True)):
                self.ax.add_patch(
                    plt.Rectangle(
                        (bx0, by0),
                        bx1 - bx0,
                        by1 - by0,
                        fill=False,
                        linestyle="-",
                        linewidth=1.2,
                        edgecolor="#111827",
                        alpha=0.9,
                        zorder=4,
                    )
                )
            if bool(logic_box.get("show_fixed_centers", False)):
                fixed_centers = logic_box.get("fixed_centers", [])
                for cp in fixed_centers:
                    cx, cy = float(cp[0]), float(cp[1])
                    self.ax.plot(
                        cx,
                        cy,
                        marker="o",
                        ms=2.8,
                        color="#9ca3af",
                        alpha=0.9,
                        zorder=4.5,
                    )
            ports = logic_box.get("ports", {})
            src = str(logic_box.get("source_port", ""))
            tgt = str(logic_box.get("target_port", ""))
            route_pairs = logic_box.get("route_pairs", [])
            src_set = set()
            tgt_set = set()
            if isinstance(route_pairs, (list, tuple)):
                for pair in route_pairs:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        src_set.add(str(pair[0]).upper())
                        tgt_set.add(str(pair[1]).upper())
            for name, info in ports.items():
                xy = info.get("xy", [0.0, 0.0])
                px, py = float(xy[0]), float(xy[1])
                color = "#1f2937"
                if name.upper() in src_set:
                    color = "#16a34a"
                if name.upper() in tgt_set:
                    color = "#dc2626"
                if name == src and len(src_set) == 0:
                    color = "#16a34a"
                elif name == tgt and len(tgt_set) == 0:
                    color = "#dc2626"
                self.ax.plot(px, py, marker="o", ms=4, color=color, zorder=6)
                if info.get("side") == "left":
                    self.ax.text(px - 0.008, py, name, fontsize=7, ha="right", va="center")
                elif info.get("side") == "right":
                    self.ax.text(px + 0.008, py, name, fontsize=7, ha="left", va="center")
                elif info.get("side") == "top":
                    self.ax.text(px, py + 0.008, name, fontsize=7, ha="center", va="bottom")
                else:
                    self.ax.text(px, py - 0.008, name, fontsize=7, ha="center", va="top")

            seed_points = logic_box.get("seed_points", [])
            for i, sp in enumerate(seed_points):
                sx, sy = float(sp[0]), float(sp[1])
                self.ax.plot(sx, sy, marker="D", ms=4, color="#0f766e", zorder=6)
                self.ax.text(sx - 0.006, sy + 0.004, f"s{i}", fontsize=7, color="#0f766e")

            seed_streamlines = logic_box.get("seed_streamlines", [])
            seed_colors = ["#0ea5e9", "#f59e0b", "#22c55e", "#a855f7", "#ef4444"]
            for i, tr in enumerate(seed_streamlines):
                hist = tr.get("history", [])
                if not isinstance(hist, (list, tuple)) or len(hist) < 2:
                    continue
                arr = np.asarray(hist, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                color = seed_colors[i % len(seed_colors)]
                if bool(tr.get("collision", False)):
                    color = "#dc2626"
                self.ax.plot(
                    arr[:, 0],
                    arr[:, 1],
                    linestyle="-.",
                    lw=1.3,
                    color=color,
                    alpha=0.95,
                    zorder=5,
                )
                self.ax.plot(
                    float(arr[0, 0]),
                    float(arr[0, 1]),
                    marker="D",
                    ms=4,
                    color=color,
                    zorder=6,
                )
                self.ax.plot(
                    float(arr[-1, 0]),
                    float(arr[-1, 1]),
                    marker="x",
                    ms=5,
                    color=color,
                    zorder=6,
                )
            if len(src_set) > 0 and len(tgt_set) > 0:
                route_text = " | ".join(
                    [f"{str(p[0]).upper()}->{str(p[1]).upper()}" for p in route_pairs if isinstance(p, (list, tuple)) and len(p) == 2]
                )
                if len(route_text) > 0:
                    self.ax.text(
                        bx0,
                        by1 + 0.008,
                        route_text,
                        fontsize=7,
                        color="#0f172a",
                        ha="left",
                        va="bottom",
                        zorder=7,
                    )

        px, py = scene["particle"]["x"], scene["particle"]["y"]
        self.ax.plot(px, py, "ro", ms=5, zorder=5)

        if target_pos is not None:
            self.ax.plot(
                float(target_pos[0]),
                float(target_pos[1]),
                "s",
                color="green",
                ms=5,
                alpha=0.8,
                zorder=4
            )

        if title:
            self.ax.set_title(title, fontsize=9)

        self.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4)

        rgb_buf = buf[:, :, 1:].copy()
        return rgb_buf

    def close(self):
        plt.close(self.fig)
