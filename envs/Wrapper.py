import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import StokesCylinderConfig, TrainingSettingConfig
from envs.FluidEnv import FluidEnv
from utils.calc import calculate_point_velocity


def layout_overlap_penalty(x, y, r, margin=0.002):
    penalty = 0.0
    n = len(r)
    for i in range(n):
        if r[i] <= 0:
            continue
        for j in range(i + 1, n):
            if r[j] <= 0:
                continue
            dij = np.hypot(x[i] - x[j], y[i] - y[j])
            overlap = (r[i] + r[j] + margin) - dij
            if overlap > 0:
                penalty += overlap
    return penalty


def layout_boundary_penalty(x, y, r, xlim, ylim, margin=0.002):
    penalty = 0.0
    for xi, yi, ri in zip(x, y, r):
        if ri <= 0:
            continue
        penalty += max(0.0, xlim[0] - (xi - ri - margin))
        penalty += max(0.0, (xi + ri + margin) - xlim[1])
        penalty += max(0.0, ylim[0] - (yi - ri - margin))
        penalty += max(0.0, (yi + ri + margin) - ylim[1])
    return penalty


def path_blocking_penalty(path: np.ndarray, x: np.ndarray, y: np.ndarray, r: np.ndarray, clearance=0.006):
    """
    Penalize cylinders that intrude into a corridor around the target path.
    """
    penalty = 0.0
    for xi, yi, ri in zip(x, y, r):
        if ri <= 0:
            continue
        d = np.linalg.norm(path - np.array([xi, yi], dtype=np.float32), axis=1)
        min_d = float(np.min(d))
        overlap = (float(ri) + float(clearance)) - min_d
        if overlap > 0:
            penalty += overlap
    return penalty


def flow_path_alignment_metrics(
    path: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    omega: float,
    action_dt: float,
    samples=24,
    lane_offsets=(0.0, 0.006, -0.006),
):
    """
    Compare flow direction/speed along the target path against path tangents.
    Returns normalized penalties (lower is better).
    """
    n = int(path.shape[0])
    if n < 3:
        return 0.0, 0.0, 0.0

    m = min(samples, n - 2)
    idx = np.linspace(1, n - 2, num=m, dtype=np.int32)
    pts_center = path[idx]
    tang = path[idx + 1] - path[idx - 1]
    t_norm = np.linalg.norm(tang, axis=1, keepdims=True)
    t_hat = tang / np.maximum(t_norm, 1e-8)
    n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)

    omegas = np.full(len(x), float(omega), dtype=np.float32)
    cos_vals = []
    speed_vals = []
    for off in lane_offsets:
        pts = pts_center + float(off) * n_hat
        v = np.zeros((m, 2), dtype=np.float32)
        for k, p in enumerate(pts):
            vx, vy = calculate_point_velocity(
                float(p[0]),
                float(p[1]),
                x,
                y,
                r,
                omegas,
            )
            v[k, 0] = vx
            v[k, 1] = vy
        speed = np.linalg.norm(v, axis=1)
        v_hat = v / np.maximum(speed[:, None], 1e-8)
        cos_sim = np.sum(v_hat * t_hat, axis=1)
        cos_vals.append(np.clip(cos_sim, -1.0, 1.0))
        speed_vals.append(speed)

    cos_all = np.concatenate(cos_vals, axis=0)
    speed_all = np.concatenate(speed_vals, axis=0)

    # 0 is perfect; >0 is misalignment.
    dir_pen = float(np.mean(1.0 - cos_all))

    path_speed_ref = float(
        np.mean(np.linalg.norm(path[1:] - path[:-1], axis=1)) / max(action_dt, 1e-8)
    )
    speed_deficit = np.maximum(path_speed_ref - speed_all, 0.0) / max(path_speed_ref, 1e-8)
    speed_pen = float(np.mean(speed_deficit))

    reverse_ratio = float(np.mean((cos_all < 0.0).astype(np.float32)))
    return dir_pen, speed_pen, reverse_ratio


class TaskContext:
    def __init__(self):
        self.path = []
        self.current_step = 0
        self.particle_pos = np.array([0.0, 0.0], dtype=np.float32)

    def generate_fixed_path(self, path_type="bezier"):
        steps = TrainingSettingConfig.EPISODE_LENGTH + 1
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

        else:
            x0, x1 = -0.025, 0.050
            y0, y1 = 0.060, -0.010
            xs = x0 + (x1 - x0) * s
            ys = y0 + (y1 - y0) * s

        self.path = np.stack([xs, ys], axis=1).astype(np.float32)

    @property
    def goal(self) -> np.ndarray:
        idx = min(self.current_step, len(self.path) - 1)
        return self.path[idx]


class FollowerEnv(gym.Env):
    def __init__(self, fluid_env: FluidEnv, ctx: TaskContext):
        super().__init__()
        self.env = fluid_env
        self.ctx = ctx
        self.layout_mode = self.env.layout_mode

        if self.layout_mode == "fixed_grid_3x3":
            action_dim = self.env.max_n
        else:
            action_dim = self.env.max_n * 3

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        sample_state = self.env.get_state()
        obs_dim = sample_state.shape[0] + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.episode_return = 0.0
        self.prev_dist = None
        self.last_history = []

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        r_low, r_high = StokesCylinderConfig.MIN_R, StokesCylinderConfig.MAX_R

        if self.layout_mode == "fixed_grid_3x3":
            a = np.asarray(action, dtype=np.float32).reshape(self.env.max_n).copy()
            a = r_low + 0.5 * (a + 1.0) * (r_high - r_low)
            a = np.where(a < StokesCylinderConfig.EXIST_THRESHOLD, 0.0, a)
            return a.astype(np.float32)

        a = np.asarray(action, dtype=np.float32).reshape(self.env.max_n, 3).copy()
        x_low, x_high = StokesCylinderConfig.X_RANGE
        y_low, y_high = StokesCylinderConfig.Y_RANGE
        a[:, 0] = x_low + 0.5 * (a[:, 0] + 1.0) * (x_high - x_low)
        a[:, 1] = y_low + 0.5 * (a[:, 1] + 1.0) * (y_high - y_low)
        a[:, 2] = r_low + 0.5 * (a[:, 2] + 1.0) * (r_high - r_low)
        a[:, 2] = np.where(a[:, 2] < StokesCylinderConfig.EXIST_THRESHOLD, 0.0, a[:, 2])
        return a.reshape(-1).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        raw_state = self.env.get_state()
        dxdy = np.array(
            [
                self.ctx.goal[0] - self.env.particle.pos_x,
                self.ctx.goal[1] - self.env.particle.pos_y,
            ],
            dtype=np.float32,
        )
        return np.concatenate([raw_state, dxdy]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ctx.current_step = 0
        self.ctx.generate_fixed_path(path_type=TrainingSettingConfig.PATH_TYPE)
        ini_pos = np.array(self.ctx.path[0], dtype=np.float32)
        self.env.reset(ini_pos)
        self.ctx.particle_pos = ini_pos.copy()

        self.episode_return = 0.0
        self.last_history = [tuple(ini_pos)]

        dx = self.ctx.goal[0] - self.ctx.particle_pos[0]
        dy = self.ctx.goal[1] - self.ctx.particle_pos[1]
        self.prev_dist = float(np.hypot(dx, dy))

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # One-shot static optimization: reward is computed directly from
        # layout-induced flow quality w.r.t. target trajectory.
        physical_action = self._decode_action(action)
        self.env.apply_layout(physical_action)

        x = self.env.cylinders_x
        y = self.env.cylinders_y
        r = self.env.cylinders_r
        path_np = np.asarray(self.ctx.path, dtype=np.float32)

        if self.layout_mode == "fixed_grid_3x3":
            w_layout = 6.0
            w_block = 80.0
            w_flow_dir = 22.0
            w_flow_speed = 18.0
            w_flow_reverse = 14.0
            cluster_scale = 0.0
        else:
            w_layout = 8.0
            w_block = 90.0
            w_flow_dir = 28.0
            w_flow_speed = 22.0
            w_flow_reverse = 16.0
            cluster_scale = 0.005

        overlap_pen = layout_overlap_penalty(x, y, r, margin=0.001)
        bound_pen = layout_boundary_penalty(
            x,
            y,
            r,
            xlim=StokesCylinderConfig.X_RANGE,
            ylim=StokesCylinderConfig.Y_RANGE,
            margin=0.001,
        )
        layout_penalty = w_layout * (overlap_pen + bound_pen)

        active_count = int(np.sum(r > 0))
        sparse_penalty = 0.0

        cluster_penalty = 0.0
        if cluster_scale > 0.0:
            active_idx = np.where(r > 0)[0]
            for ii in range(len(active_idx)):
                for jj in range(ii + 1, len(active_idx)):
                    i = active_idx[ii]
                    j = active_idx[jj]
                    dij = np.hypot(x[i] - x[j], y[i] - y[j])
                    cluster_penalty += 1.0 / (dij + 1e-3)
            cluster_penalty *= cluster_scale

        block_pen = path_blocking_penalty(path_np, x, y, r, clearance=0.006)
        flow_dir_pen, flow_speed_pen, flow_reverse_ratio = flow_path_alignment_metrics(
            path=path_np,
            x=x,
            y=y,
            r=r,
            omega=self.env.fixed_omega,
            action_dt=self.env.action_dt,
            samples=24,
        )

        total_reward = 0.0
        total_reward -= layout_penalty
        total_reward -= cluster_penalty
        total_reward -= w_block * block_pen
        total_reward -= w_flow_dir * flow_dir_pen
        total_reward -= w_flow_speed * flow_speed_pen
        total_reward -= w_flow_reverse * flow_reverse_ratio

        # Run one short rollout only for visualization/diagnostics.
        _, rollout_terminated, extra_info = self.env.step(action=None)
        step_history = extra_info.get("history_pos", [])
        full_history = [tuple(self.ctx.particle_pos.copy())]
        if len(step_history) > 0:
            full_history.extend(step_history)
        cur_pos = np.array(
            [self.env.particle.pos_x, self.env.particle.pos_y], dtype=np.float32
        )
        self.ctx.particle_pos = cur_pos.copy()

        final_goal = np.array(self.ctx.path[-1], dtype=np.float32)
        final_dist = float(np.linalg.norm(cur_pos - final_goal))
        traj_np = np.asarray(full_history, dtype=np.float32)

        d2 = np.sum((path_np[:, None, :] - traj_np[None, :, :]) ** 2, axis=2)
        target_to_traj_mean = float(np.mean(np.sqrt(np.min(d2, axis=1))))
        traj_to_target_mean = float(np.mean(np.sqrt(np.min(d2, axis=0))))
        mean_path_fit_dist = float(np.mean(np.sqrt(np.min(d2, axis=0))))

        nearest_path_idx = int(np.max(np.argmin(d2, axis=0)))
        path_coverage = float(nearest_path_idx) / float(max(1, len(path_np) - 1))

        self.last_history = full_history
        self.episode_return = float(total_reward)

        obs = self._get_obs()
        terminated = True
        truncated = False

        return obs, float(total_reward), terminated, truncated, {
            "episode_return": float(total_reward),
            "history_pos": full_history,
            "target_path": self.ctx.path.tolist(),
            "final_pos": self.ctx.particle_pos.tolist(),
            "layout_x": self.env.cylinders_x.tolist(),
            "layout_y": self.env.cylinders_y.tolist(),
            "layout_r": self.env.cylinders_r.tolist(),
            "layout_omega": np.full(
                self.env.max_n, self.env.fixed_omega, dtype=np.float32
            ).tolist(),
            "terminated_early": bool(rollout_terminated),
            "layout_penalty": float(layout_penalty),
            "overlap_penalty": float(overlap_pen),
            "boundary_penalty": float(bound_pen),
            "sparse_penalty": float(sparse_penalty),
            "cluster_penalty": float(cluster_penalty),
            "active_count": int(active_count),
            "layout_mode": self.layout_mode,
            "mean_path_dist": float(mean_path_fit_dist),
            "final_dist": float(final_dist),
            "path_coverage": float(path_coverage),
            "nearest_path_idx": int(nearest_path_idx),
            "target_to_traj_mean": float(target_to_traj_mean),
            "traj_to_target_mean": float(traj_to_target_mean),
            "path_block_penalty": float(block_pen),
            "far_cylinder_penalty": 0.0,
            "flow_dir_penalty": float(flow_dir_pen),
            "flow_speed_penalty": float(flow_speed_pen),
            "flow_reverse_ratio": float(flow_reverse_ratio),
        }

    def get_scene(self) -> dict:
        omegas = np.full(self.env.max_n, self.env.fixed_omega, dtype=np.float32)
        return {
            "particle": {
                "x": float(self.ctx.particle_pos[0]),
                "y": float(self.ctx.particle_pos[1]),
            },
            "target": {
                "x": float(self.ctx.goal[0]),
                "y": float(self.ctx.goal[1]),
            },
            "cylinders": {
                "x": self.env.cylinders_x.tolist(),
                "y": self.env.cylinders_y.tolist(),
                "r": self.env.cylinders_r.tolist(),
                "omegas": omegas.tolist(),
            },
        }
