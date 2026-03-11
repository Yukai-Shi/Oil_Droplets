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

        if draw_flow and len(cyl_x) > 0:
            U, V = calculate_velocity_grid(self.X, self.Y, cyl_x, cyl_y, cyl_r, omegas)
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

        for i in range(len(cyl_x)):
            if cyl_r[i] <= 0:
                continue

            color = "green" if omegas[i] > 0 else "red"
            circ = plt.Circle(
                (cyl_x[i], cyl_y[i]),
                cyl_r[i],
                color=color,
                alpha=0.4,
                ec="black",
                lw=0.8,
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