from typing import Optional, Tuple

import numpy as np

from config import (
    FixedGrid3x3Config,
    Gate3LevelConfig,
    GlobalOmegaControlConfig,
    get_logic_box_ranges,
    get_logic_max_radius,
    InflowConfig,
    LayoutModeConfig,
    LogicBoxConfig,
    RenderSettingConfig,
    StokesCylinderConfig,
    TrajectorySettingConfig,
)
from utils.calc import calculate_point_velocity, is_legal


def _parse_layout_mode(layout_mode: str):
    mode = str(layout_mode)
    if mode.endswith("_inflow_u_fixed"):
        return mode[:-15], True, True, True
    # Backward-compatible alias: *_inflow_u now means fixed horizontal inflow.
    if mode.endswith("_inflow_u"):
        return mode[:-9], True, True, True
    if mode.endswith("_inflow"):
        return mode[:-7], True, False, False
    return mode, False, False, False


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
        seed=None,
        layout_mode: Optional[str] = None,
    ):
        self.layout_mode = layout_mode or LayoutModeConfig.LAYOUT_MODE
        (
            self.base_layout_mode,
            self.use_inflow,
            self.horizontal_inflow_only,
            self.fixed_horizontal_inflow,
        ) = _parse_layout_mode(self.layout_mode)
        if self.base_layout_mode not in {
            "free_layout",
            "fixed_grid_3x3",
            "gate3_layout",
            "logic_box_layout",
        }:
            raise ValueError(f"Unknown layout_mode: {self.layout_mode}")

        if self.base_layout_mode == "fixed_grid_3x3":
            self.max_n = FixedGrid3x3Config.NUM_CYLINDERS
            self.grid_x, self.grid_y = FixedGrid3x3Config.grid_coords()
            self.fixed_count = self.max_n
            self.design_n = 0
        elif self.base_layout_mode == "gate3_layout":
            self.max_n = StokesCylinderConfig.NUM_CYLINDERS
            self.grid_x, self.grid_y = None, None
            self.fixed_count = Gate3LevelConfig.NUM_FIXED
            self.design_n = self.max_n - self.fixed_count
            if self.design_n <= 0:
                raise ValueError(
                    "StokesCylinderConfig.NUM_CYLINDERS must be greater than "
                    "Gate3LevelConfig.NUM_FIXED."
                )
            self.gate_fixed_x = Gate3LevelConfig.FIXED_X.astype(np.float32).copy()
            self.gate_fixed_y = Gate3LevelConfig.FIXED_Y.astype(np.float32).copy()
            self.gate_fixed_r = Gate3LevelConfig.FIXED_R.astype(np.float32).copy()
            if len(self.gate_fixed_x) != self.fixed_count:
                raise ValueError("Gate3 fixed cylinder config length mismatch.")
        elif self.base_layout_mode == "logic_box_layout":
            self.max_n = StokesCylinderConfig.NUM_CYLINDERS
            self.grid_x, self.grid_y = None, None
            self.fixed_count = 0
            self.design_n = self.max_n
            self.logic_fixed_layout = bool(getattr(LogicBoxConfig, "FIXED_LAYOUT_ENABLE", False))
            if self.logic_fixed_layout:
                fx = np.asarray(getattr(LogicBoxConfig, "FIXED_LAYOUT_X", []), dtype=np.float32).reshape(-1)
                fy = np.asarray(getattr(LogicBoxConfig, "FIXED_LAYOUT_Y", []), dtype=np.float32).reshape(-1)
                if fx.size != self.max_n or fy.size != self.max_n:
                    raise ValueError(
                        f"LogicBox fixed layout expects {self.max_n} centers, got x={fx.size}, y={fy.size}"
                    )
                self.logic_fixed_x = fx.copy()
                self.logic_fixed_y = fy.copy()
            else:
                self.logic_fixed_x = None
                self.logic_fixed_y = None
        else:
            self.max_n = StokesCylinderConfig.NUM_CYLINDERS
            self.grid_x, self.grid_y = None, None
            self.fixed_count = 0
            self.design_n = self.max_n
            self.logic_fixed_layout = False
            self.logic_fixed_x = None
            self.logic_fixed_y = None

        self.default_omega = float(StokesCylinderConfig.FIXED_OMEGA)
        self.fixed_omega = float(self.default_omega)
        self.optimize_omega = bool(GlobalOmegaControlConfig.OPTIMIZE_OMEGA)
        self.omega_action_dim = int(self.optimize_omega)
        self.inflow_u = float(InflowConfig.U_IN) if self.use_inflow else 0.0
        self.inflow_v = (
            0.0 if self.horizontal_inflow_only else (float(InflowConfig.V_IN) if self.use_inflow else 0.0)
        )
        self.optimize_inflow_u = bool(
            self.use_inflow and (not self.fixed_horizontal_inflow) and InflowConfig.OPTIMIZE_U
        )
        self.optimize_inflow_v = bool(
            self.use_inflow and (not self.horizontal_inflow_only) and InflowConfig.OPTIMIZE_V
        )
        self.inflow_action_dim = int(self.optimize_inflow_u) + int(self.optimize_inflow_v)
        self.inflow_control_enabled = True

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
        self.cylinders_r.fill(0.0)
        self.fixed_omega = float(self.default_omega)

        if self.base_layout_mode == "fixed_grid_3x3":
            self.cylinders_x[:] = self.grid_x
            self.cylinders_y[:] = self.grid_y
        elif self.base_layout_mode == "gate3_layout":
            self.cylinders_x.fill(0.0)
            self.cylinders_y.fill(0.0)
            self.cylinders_x[: self.fixed_count] = self.gate_fixed_x
            self.cylinders_y[: self.fixed_count] = self.gate_fixed_y
            self.cylinders_r[: self.fixed_count] = self.gate_fixed_r
        elif self.base_layout_mode == "logic_box_layout" and self.logic_fixed_layout:
            self.cylinders_x[:] = self.logic_fixed_x
            self.cylinders_y[:] = self.logic_fixed_y
        else:
            self.cylinders_x.fill(0.0)
            self.cylinders_y.fill(0.0)

        self.particle.reset(pos.astype(np.float32))
        return self.get_state()

    def get_state(self) -> np.ndarray:
        xp = np.array([self.particle.pos_x, self.particle.pos_y], dtype=np.float32)
        if self.base_layout_mode == "fixed_grid_3x3":
            layout = self.cylinders_r.astype(np.float32)
        else:
            layout = np.concatenate(
                [self.cylinders_x, self.cylinders_y, self.cylinders_r], axis=0
            ).astype(np.float32)
        return np.concatenate([xp, layout], axis=0).astype(np.float32)

    def apply_layout(self, action: np.ndarray):
        r_low = StokesCylinderConfig.MIN_R
        r_high = StokesCylinderConfig.MAX_R

        def _clip_design_radii(raw_r: np.ndarray, upper: float):
            rr = np.clip(np.asarray(raw_r, dtype=np.float32), 0.0, float(upper))
            active = rr > 0.0
            rr[active] = np.maximum(rr[active], float(r_low))
            return rr.astype(np.float32)

        action_vec = np.asarray(action, dtype=np.float32).reshape(-1)

        layout_vec = action_vec
        tail_dim = self.inflow_action_dim + self.omega_action_dim
        if tail_dim > 0:
            if action_vec.size <= tail_dim:
                raise ValueError(
                    f"Action dim too small for layout+tail controls, got {action_vec.size}"
                )
            layout_vec = action_vec[:-tail_dim]
            tail_vec = action_vec[-tail_dim:]
            k = 0
            if self.inflow_action_dim > 0:
                if self.inflow_control_enabled:
                    if self.optimize_inflow_u:
                        self.inflow_u = float(
                            np.clip(tail_vec[k], InflowConfig.U_MIN, InflowConfig.U_MAX)
                        )
                        k += 1
                    if self.optimize_inflow_v:
                        self.inflow_v = float(
                            np.clip(tail_vec[k], InflowConfig.V_MIN, InflowConfig.V_MAX)
                        )
                        k += 1
                else:
                    if self.optimize_inflow_u:
                        k += 1
                    if self.optimize_inflow_v:
                        k += 1
                    self.inflow_u = float(
                        np.clip(InflowConfig.TARGET_U, InflowConfig.U_MIN, InflowConfig.U_MAX)
                    )
                    self.inflow_v = 0.0 if self.horizontal_inflow_only else float(
                        np.clip(InflowConfig.TARGET_V, InflowConfig.V_MIN, InflowConfig.V_MAX)
                    )
            elif self.use_inflow:
                # Keep configured inflow as fixed value.
                self.inflow_u = float(InflowConfig.U_IN)
                self.inflow_v = 0.0 if self.horizontal_inflow_only else float(InflowConfig.V_IN)

            if self.omega_action_dim > 0:
                omega_cmd = float(tail_vec[k])
                mode = str(GlobalOmegaControlConfig.MODE).lower()
                if mode == "continuous":
                    self.fixed_omega = float(
                        np.clip(
                            omega_cmd,
                            float(GlobalOmegaControlConfig.OMEGA_MIN),
                            float(GlobalOmegaControlConfig.OMEGA_MAX),
                        )
                    )
                else:
                    omega_abs = max(abs(self.default_omega), 1e-8)
                    self.fixed_omega = float(omega_abs if omega_cmd >= 0.0 else -omega_abs)
        else:
            if self.use_inflow:
                self.inflow_u = float(InflowConfig.U_IN)
                self.inflow_v = 0.0 if self.horizontal_inflow_only else float(InflowConfig.V_IN)
            self.fixed_omega = float(self.default_omega)

        if self.horizontal_inflow_only:
            # In *_inflow_u_fixed mode, vertical inflow is disabled by definition.
            self.inflow_v = 0.0

        if self.base_layout_mode == "fixed_grid_3x3":
            radii = _clip_design_radii(layout_vec.reshape(self.max_n), r_high)
            self.cylinders_x[:] = self.grid_x
            self.cylinders_y[:] = self.grid_y
            self.cylinders_r[:] = radii
            return
        if self.base_layout_mode == "gate3_layout":
            if layout_vec.size != self.design_n * 3:
                raise ValueError(
                    f"Gate3 layout expects {(self.design_n * 3)} dims, got {layout_vec.size}"
                )
            new_layout = layout_vec.reshape(self.design_n, 3)
            self.cylinders_x[: self.fixed_count] = self.gate_fixed_x
            self.cylinders_y[: self.fixed_count] = self.gate_fixed_y
            self.cylinders_r[: self.fixed_count] = np.clip(
                self.gate_fixed_r, r_low, r_high
            )
            self.cylinders_x[self.fixed_count :] = new_layout[:, 0]
            self.cylinders_y[self.fixed_count :] = new_layout[:, 1]
            self.cylinders_r[self.fixed_count :] = _clip_design_radii(new_layout[:, 2], r_high)
            return
        if self.base_layout_mode == "logic_box_layout":
            (x0, x1), (y0, y1) = get_logic_box_ranges()
            margin = 1e-4
            max_box_r = max(
                0.0,
                min(
                    0.5 * (float(x1) - float(x0)) - margin,
                    0.5 * (float(y1) - float(y0)) - margin,
                ),
            )
            r_eff_high = min(float(r_high), float(max_box_r), float(get_logic_max_radius()))
            logic_forbid_elimination = bool(getattr(LogicBoxConfig, "FORBID_ELIMINATION", False))
            logic_min_active_r = max(float(r_low), float(getattr(LogicBoxConfig, "MIN_ACTIVE_R", r_low)))
            if self.logic_fixed_layout:
                # Compatibility:
                # - legacy fixed-layout physical action: N radii
                # - stage-switch compatible action: N*3 (x/y/r), use only r
                if int(layout_vec.size) == int(self.max_n):
                    raw_r = layout_vec.reshape(self.max_n)
                elif int(layout_vec.size) == int(self.max_n * 3):
                    raw_r = layout_vec.reshape(self.max_n, 3)[:, 2]
                else:
                    raise ValueError(
                        f"LogicBox fixed layout expects {self.max_n} or {self.max_n * 3} dims, got {layout_vec.size}"
                    )
                radii = _clip_design_radii(raw_r, r_eff_high)
                # Keep fixed centers and ensure radius does not exceed local boundary clearance.
                x_fix = self.logic_fixed_x.astype(np.float32)
                y_fix = self.logic_fixed_y.astype(np.float32)
                r_bound = np.minimum.reduce(
                    [
                        x_fix - float(x0) - margin,
                        float(x1) - x_fix - margin,
                        y_fix - float(y0) - margin,
                        float(y1) - y_fix - margin,
                    ]
                )
                r_bound = np.maximum(r_bound, 0.0).astype(np.float32)
                radii = np.minimum(radii, r_bound).astype(np.float32)
                if logic_forbid_elimination:
                    # Keep every fixed cylinder alive while respecting local geometric bounds.
                    floor_bound = np.minimum(
                        np.full_like(radii, float(logic_min_active_r), dtype=np.float32),
                        r_bound,
                    )
                    radii = np.maximum(radii, floor_bound).astype(np.float32)
            else:
                new_layout = layout_vec.reshape(self.max_n, 3)
                radii = _clip_design_radii(new_layout[:, 2], r_eff_high)
                if logic_forbid_elimination:
                    radii = np.maximum(radii, float(logic_min_active_r)).astype(np.float32)

            if not logic_forbid_elimination:
                logic_exist_th = float(getattr(LogicBoxConfig, "EXIST_THRESHOLD", 0.0))
                if logic_exist_th > 0.0:
                    radii = np.where(radii < logic_exist_th, 0.0, radii).astype(np.float32)

            if self.logic_fixed_layout:
                self.cylinders_x[:] = self.logic_fixed_x
                self.cylinders_y[:] = self.logic_fixed_y
            else:
                x_min = float(x0) + radii + margin
                x_max = float(x1) - radii - margin
                y_min = float(y0) + radii + margin
                y_max = float(y1) - radii - margin
                self.cylinders_x[:] = np.minimum(np.maximum(new_layout[:, 0], x_min), x_max)
                self.cylinders_y[:] = np.minimum(np.maximum(new_layout[:, 1], y_min), y_max)
            self.cylinders_r[:] = radii
            return

        new_layout = layout_vec.reshape(self.max_n, 3)
        self.cylinders_x[:] = new_layout[:, 0]
        self.cylinders_y[:] = new_layout[:, 1]
        self.cylinders_r[:] = _clip_design_radii(new_layout[:, 2], r_high)

    def set_inflow_control_enabled(self, enabled: bool):
        self.inflow_control_enabled = bool(enabled)
        if self.use_inflow and (not self.inflow_control_enabled):
            if self.fixed_horizontal_inflow:
                self.inflow_u = float(np.clip(InflowConfig.U_IN, InflowConfig.U_MIN, InflowConfig.U_MAX))
            else:
                self.inflow_u = float(
                    np.clip(InflowConfig.TARGET_U, InflowConfig.U_MIN, InflowConfig.U_MAX)
                )
            self.inflow_v = 0.0 if self.horizontal_inflow_only else float(
                np.clip(InflowConfig.TARGET_V, InflowConfig.V_MIN, InflowConfig.V_MAX)
            )

    def set_logic_fixed_layout(self, enabled: bool, fixed_x=None, fixed_y=None):
        """
        Toggle logic-box fixed-layout mode at runtime.
        When enabling, optionally provide frozen centers (fixed_x/fixed_y).
        """
        if self.base_layout_mode != "logic_box_layout":
            return
        enabled = bool(enabled)
        if enabled:
            if fixed_x is not None and fixed_y is not None:
                fx = np.asarray(fixed_x, dtype=np.float32).reshape(-1)
                fy = np.asarray(fixed_y, dtype=np.float32).reshape(-1)
                if fx.size != self.max_n or fy.size != self.max_n:
                    raise ValueError(
                        f"set_logic_fixed_layout expects {self.max_n} centers, got x={fx.size}, y={fy.size}"
                    )
                self.logic_fixed_x = fx.copy()
                self.logic_fixed_y = fy.copy()
            else:
                # Freeze current layout centers if explicit centers are not provided.
                self.logic_fixed_x = self.cylinders_x.astype(np.float32).copy()
                self.logic_fixed_y = self.cylinders_y.astype(np.float32).copy()
            self.logic_fixed_layout = True
            self.cylinders_x[:] = self.logic_fixed_x
            self.cylinders_y[:] = self.logic_fixed_y
        else:
            self.logic_fixed_layout = False

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
                omegas,
            )
            vx += self.inflow_u
            vy += self.inflow_v
            self.particle.vx = float(vx)
            self.particle.vy = float(vy)

            self.particle.pos_x += self.particle.vx * self.simul_dt
            self.particle.pos_y += self.particle.vy * self.simul_dt

            history_pos.append((self.particle.pos_x, self.particle.pos_y))

            centers = np.stack([self.cylinders_x, self.cylinders_y], axis=1)
            if not is_legal(
                [self.particle.pos_x, self.particle.pos_y],
                centers=centers,
                radii=self.cylinders_r,
            ):
                done = True
                break

            x_min, x_max = RenderSettingConfig.X_LIM
            y_min, y_max = RenderSettingConfig.Y_LIM
            if not (
                x_min <= self.particle.pos_x <= x_max
                and y_min <= self.particle.pos_y <= y_max
            ):
                done = True
                break

        extra_info = {
            "history_pos": history_pos,
            "particle_pos": (self.particle.pos_x, self.particle.pos_y),
            "layout_mode": self.layout_mode,
            "inflow_u": float(self.inflow_u),
            "inflow_v": float(self.inflow_v),
            "global_omega": float(self.fixed_omega),
            "layout": {
                "x": self.cylinders_x.copy(),
                "y": self.cylinders_y.copy(),
                "r": self.cylinders_r.copy(),
            },
            "fixed_count": int(self.fixed_count),
        }

        return self.get_state(), done, extra_info
