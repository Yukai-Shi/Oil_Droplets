import numpy as np
from typing import Tuple

from config import StokesCylinderConfig, TrajectorySettingConfig, RenderSettingConfig
from utils.calc import calculate_point_velocity, is_legal


class Particle:
    def __init__(self, pos: np.ndarray):
        self.pos_x = float(pos[0])
        self.pos_y = float(pos[1])
        self.vx = 0.0
        self.vy = 0.0

    def reset(self, pos: np.ndarray):
        self.pos_x = float(pos[0])
        self.pos_y = float(pos[1])
        self.vx = 0.0
        self.vy = 0.0


class FluidEnv:
    def __init__(
        self,
        start_pos: Tuple[float, float] = (0.0, 0.0),
        seed=None
    ):
        self.max_n = StokesCylinderConfig.NUM_CYLINDERS
        self.fixed_omega = StokesCylinderConfig.FIXED_OMEGA

        self.cylinders_x = np.zeros(self.max_n, dtype=np.float32)
        self.cylinders_y = np.zeros(self.max_n, dtype=np.float32)
        self.cylinders_r = np.zeros(self.max_n, dtype=np.float32)

        self.particle = Particle(np.array(start_pos, dtype=np.float32))

        self.simul_dt = TrajectorySettingConfig.SIMUL_DT
        self.action_dt = TrajectorySettingConfig.ACTION_DT
        self.simul_nums = int(self.action_dt / self.simul_dt)

        self.step_count = 0
        self.reset(np.asarray(start_pos, dtype=np.float32))

    def reset(self, pos: np.ndarray) -> np.ndarray:
        self.step_count = 0

        self.cylinders_x.fill(0.0)
        self.cylinders_y.fill(0.0)
        self.cylinders_r.fill(0.0)

        self.particle.reset(pos.astype(np.float32))
        return self.get_state()

    def get_state(self) -> np.ndarray:
        xp = np.array([self.particle.pos_x, self.particle.pos_y], dtype=np.float32)
        layout = np.concatenate(
            [self.cylinders_x, self.cylinders_y, self.cylinders_r],
            axis=0
        ).astype(np.float32)
        return np.concatenate([xp, layout], axis=0).astype(np.float32)

    def apply_layout(self, action: np.ndarray):
        new_layout = np.asarray(action, dtype=np.float32).reshape(self.max_n, 3)
        self.cylinders_x[:] = new_layout[:, 0]
        self.cylinders_y[:] = new_layout[:, 1]
        self.cylinders_r[:] = new_layout[:, 2]

    def step(self, action: np.ndarray = None):
        self.step_count += 1
        done = False

        if action is not None:
            self.apply_layout(action)

        omegas = np.full(self.max_n, self.fixed_omega, dtype=np.float32)
        history_pos = []

        for _ in range(self.simul_nums):
            vx, vy = calculate_point_velocity(
                self.particle.pos_x,
                self.particle.pos_y,
                self.cylinders_x,
                self.cylinders_y,
                self.cylinders_r,
                omegas
            )
            self.particle.vx = float(vx)
            self.particle.vy = float(vy)

            self.particle.pos_x += self.particle.vx * self.simul_dt
            self.particle.pos_y += self.particle.vy * self.simul_dt

            history_pos.append((self.particle.pos_x, self.particle.pos_y))

            centers = np.stack([self.cylinders_x, self.cylinders_y], axis=1)
            if not is_legal(
                [self.particle.pos_x, self.particle.pos_y],
                centers=centers,
                radii=self.cylinders_r
            ):
                done = True
                break

            x_min, x_max = RenderSettingConfig.X_LIM
            y_min, y_max = RenderSettingConfig.Y_LIM
            if not (x_min <= self.particle.pos_x <= x_max and y_min <= self.particle.pos_y <= y_max):
                done = True
                break

        extra_info = {
            "history_pos": history_pos,
            "particle_pos": (self.particle.pos_x, self.particle.pos_y),
            "layout": {
                "x": self.cylinders_x.copy(),
                "y": self.cylinders_y.copy(),
                "r": self.cylinders_r.copy(),
            }
        }

        return self.get_state(), done, extra_info