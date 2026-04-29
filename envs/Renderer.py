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
        self.fig, self.ax = plt.subplots(
            figsize=(width / dpi, height / dpi),
            dpi=dpi,
            constrained_layout=True,
        )
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
                cy + 0.012,
                f"inflow=({inflow_u:.3f},{inflow_v:.3f})",
                fontsize=8,
                color="#111827",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.75),
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
            if float(cyl_r[i]) >= 0.0035:
                ang = np.deg2rad(float((i * 137.507764) % 360.0))
                rr = max(0.35 * float(cyl_r[i]), 0.0018)
                tx = float(cyl_x[i] + rr * np.cos(ang))
                ty = float(cyl_y[i] + rr * np.sin(ang))
                self.ax.text(
                    tx,
                    ty,
                    f"{i}",
                    fontsize=6,
                    color="black",
                    ha="center",
                    va="center",
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.60),
                    zorder=6,
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
            route_mode = str(logic_box.get("route_mode", "")).strip().lower()
            # Fixed inlet-color convention for readability across route sets.
            inlet_colors = {
                "L0": "#2563eb",  # blue
                "L1": "#f59e0b",  # amber
                "L2": "#10b981",  # emerald
            }
            neutral_port = "#9ca3af"
            src_set = set()
            tgt_set = set()
            active_pairs = []
            tgt_to_src = {}
            if isinstance(route_pairs, (list, tuple)):
                for pair in route_pairs:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        s_name = str(pair[0]).upper()
                        t_name = str(pair[1]).upper()
                        src_set.add(s_name)
                        tgt_set.add(t_name)
                        active_pairs.append((s_name, t_name))
                        if t_name not in tgt_to_src:
                            tgt_to_src[t_name] = s_name
            if len(active_pairs) == 0 and len(src) > 0 and len(tgt) > 0:
                s_name = str(src).upper()
                t_name = str(tgt).upper()
                active_pairs = [(s_name, t_name)]
                src_set.add(s_name)
                tgt_set.add(t_name)
                tgt_to_src[t_name] = s_name

            for name, info in ports.items():
                xy = info.get("xy", [0.0, 0.0])
                px, py = float(xy[0]), float(xy[1])
                nm = str(name).upper()
                side = str(info.get("side", "")).lower()
                color = neutral_port
                if side == "left":
                    if nm in src_set:
                        color = inlet_colors.get(nm, "#374151")
                else:
                    if nm in tgt_set:
                        src_name = tgt_to_src.get(nm, "")
                        color = inlet_colors.get(src_name, "#374151")
                self.ax.plot(px, py, marker="o", ms=4, color=color, zorder=6)
                x0, x1 = self.ax.get_xlim()
                y0, y1 = self.ax.get_ylim()
                dx = 0.012 * (x1 - x0)
                dy = 0.012 * (y1 - y0)
                if info.get("side") == "left":
                    self.ax.text(
                        px + dx, py, name, fontsize=7, ha="left", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.70),
                        zorder=7,
                    )
                elif info.get("side") == "right":
                    self.ax.text(
                        px - dx, py, name, fontsize=7, ha="right", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.70),
                        zorder=7,
                    )
                elif info.get("side") == "top":
                    self.ax.text(
                        px, py - dy, name, fontsize=7, ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.70),
                        zorder=7,
                    )
                else:
                    self.ax.text(
                        px, py + dy, name, fontsize=7, ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.70),
                        zorder=7,
                    )

            left_port_y = []
            for name, info in ports.items():
                if str(info.get("side", "")).lower() == "left":
                    pxy = info.get("xy", [0.0, 0.0])
                    left_port_y.append((str(name).upper(), float(pxy[1])))

            def _infer_source_from_seed(seed_xy):
                if not isinstance(seed_xy, (list, tuple)) or len(seed_xy) < 2:
                    return ""
                if len(left_port_y) == 0:
                    return ""
                sy0 = float(seed_xy[1])
                best_name = left_port_y[0][0]
                best_d = abs(sy0 - left_port_y[0][1])
                for nm, yy in left_port_y[1:]:
                    d = abs(sy0 - yy)
                    if d < best_d:
                        best_d = d
                        best_name = nm
                return best_name

            seed_streamlines = logic_box.get("seed_streamlines", [])
            x0_ax, x1_ax = self.ax.get_xlim()
            y0_ax, y1_ax = self.ax.get_ylim()
            seg_len = 0.035 * max(float(x1_ax) - float(x0_ax), float(y1_ax) - float(y0_ax))
            for i, tr in enumerate(seed_streamlines):
                hist = tr.get("history", [])
                tr_src = str(tr.get("source_port", "")).upper()
                if len(tr_src) == 0:
                    tr_src = _infer_source_from_seed(tr.get("seed", None))
                color = "#ef4444"
                tr_alpha = 0.95 if (not bool(tr.get("collision", False))) else 0.70
                arr = None
                if isinstance(hist, (list, tuple)) and len(hist) >= 2:
                    arr = np.asarray(hist, dtype=np.float32)
                    if arr.ndim != 2 or arr.shape[1] < 2:
                        arr = None

                if arr is not None and arr.shape[0] >= 2:
                    self.ax.plot(
                        arr[:, 0],
                        arr[:, 1],
                        linestyle="-",
                        lw=1.6,
                        color=color,
                        alpha=tr_alpha,
                        zorder=5,
                    )
                    self.ax.plot(
                        float(arr[0, 0]),
                        float(arr[0, 1]),
                        marker="o",
                        ms=3.8,
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
                else:
                    seed = tr.get("seed", None)
                    if isinstance(seed, (list, tuple)) and len(seed) >= 2:
                        sx, sy = float(seed[0]), float(seed[1])
                        ivx = float(tr.get("init_vx", 0.0))
                        ivy = float(tr.get("init_vy", 0.0))
                        vn = float(np.hypot(ivx, ivy))
                        if vn > 1e-12:
                            ex = sx + (ivx / vn) * seg_len
                            ey = sy + (ivy / vn) * seg_len
                        else:
                            ex = sx + 0.6 * seg_len
                            ey = sy
                        self.ax.plot(
                            [sx, ex],
                            [sy, ey],
                            linestyle="-",
                            lw=1.5,
                            color=color,
                            alpha=tr_alpha,
                            zorder=5,
                        )
                        self.ax.plot(sx, sy, marker="o", ms=3.8, color=color, zorder=6)
                        self.ax.plot(ex, ey, marker="x", ms=5, color=color, zorder=6)

            show_inlet_legend = route_mode in {
                "multi_map",
                "multi_route",
                "multi_map_switch",
                "multi_switch",
                "mapping_switch",
            }
            if show_inlet_legend:
                x0_ax, x1_ax = self.ax.get_xlim()
                y0_ax, y1_ax = self.ax.get_ylim()
                lx = float(x0_ax + 0.012 * (x1_ax - x0_ax))
                ly0 = float(y1_ax - 0.028 * (y1_ax - y0_ax))
                dy = float(0.024 * (y1_ax - y0_ax))
                for i, nm in enumerate(["L0", "L1", "L2"]):
                    ly = ly0 - i * dy
                    col = inlet_colors.get(nm, "#374151")
                    self.ax.plot(lx, ly, marker="o", ms=4, color=col, zorder=7)
                    self.ax.text(
                        lx + 0.008 * (x1_ax - x0_ax),
                        ly,
                        nm,
                        fontsize=7,
                        color=col,
                        ha="left",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.70),
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
            self.ax.set_title(title, fontsize=9, pad=10)

        self.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4)

        rgb_buf = buf[:, :, 1:].copy()
        return rgb_buf

    def close(self):
        plt.close(self.fig)
