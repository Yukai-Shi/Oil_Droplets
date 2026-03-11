import numpy as np
import gymnasium as gym
import math
from gymnasium import spaces

from envs.FluidEnv import FluidEnv
from config import *


def layout_overlap_penalty(x, y, r, margin=0.002):
    """
    连续重叠惩罚：
    如果两个圆柱间距小于 r_i + r_j + margin，就按重叠量处罚
    """
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
    """
    连续边界惩罚：
    超出边界多少，就罚多少
    """
    penalty = 0.0
    for xi, yi, ri in zip(x, y, r):
        if ri <= 0:
            continue

        penalty += max(0.0, xlim[0] - (xi - ri - margin))
        penalty += max(0.0, (xi + ri + margin) - xlim[1])
        penalty += max(0.0, ylim[0] - (yi - ri - margin))
        penalty += max(0.0, (yi + ri + margin) - ylim[1])

    return penalty


class TaskContext:
    def __init__(self):
        self.path = []
        self.current_step = 0
        self.particle_pos = np.array([0.0, 0.0], dtype=np.float32)

    def generate_fixed_path(self, path_type="bezier"):
        steps = TrainingSettingConfig.EPISODE_LENGTH + 1
        s = np.linspace(0.0, 1.0, steps)

        if path_type == "bezier":
            p0 = np.array([-0.025, 0.060], dtype=np.float32)
            p1 = np.array([ 0.000, 0.050], dtype=np.float32)
            p2 = np.array([ 0.055,-0.010], dtype=np.float32)

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

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.max_n * 3,),
            dtype=np.float32
        )

        sample_state = self.env.get_state()
        obs_dim = sample_state.shape[0] + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.episode_return = 0.0
        self.prev_dist = None
        self.last_history = []

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        a = action.reshape(self.env.max_n, 3).copy()

        x_low, x_high = StokesCylinderConfig.X_RANGE
        y_low, y_high = StokesCylinderConfig.Y_RANGE
        r_low, r_high = StokesCylinderConfig.MIN_R, StokesCylinderConfig.MAX_R

        a[:, 0] = x_low + 0.5 * (a[:, 0] + 1.0) * (x_high - x_low)
        a[:, 1] = y_low + 0.5 * (a[:, 1] + 1.0) * (y_high - y_low)
        a[:, 2] = r_low + 0.5 * (a[:, 2] + 1.0) * (r_high - r_low)

        # 小于阈值视为不放置
        a[:, 2] = np.where(
            a[:, 2] < StokesCylinderConfig.EXIST_THRESHOLD,
            0.0,
            a[:, 2]
        )

        return a.reshape(-1).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        raw_state = self.env.get_state()
        dxdy = np.array([
            self.ctx.goal[0] - self.env.particle.pos_x,
            self.ctx.goal[1] - self.env.particle.pos_y
        ], dtype=np.float32)
        return np.concatenate([raw_state, dxdy]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.ctx.current_step = 0

        ini_pos = np.array([-0.025, 0.060], dtype=np.float32)
        self.env.reset(ini_pos)
        self.ctx.particle_pos = ini_pos.copy()

        self.ctx.generate_fixed_path(path_type="bend2")

        self.episode_return = 0.0
        self.last_history = [tuple(ini_pos)]

        dx = self.ctx.goal[0] - self.ctx.particle_pos[0]
        dy = self.ctx.goal[1] - self.ctx.particle_pos[1]
        self.prev_dist = float(np.hypot(dx, dy))

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        physical_action = self._decode_action(action)

        # one-shot: 整个 episode 只设置一次布局
        self.env.apply_layout(physical_action)

        x = self.env.cylinders_x
        y = self.env.cylinders_y
        r = self.env.cylinders_r

        # ---------------------------
        # 1) 布局软惩罚（不再直接终止）
        # ---------------------------
        overlap_pen = layout_overlap_penalty(x, y, r, margin=0.001)
        bound_pen = layout_boundary_penalty(
            x, y, r,
            xlim=StokesCylinderConfig.X_RANGE,
            ylim=StokesCylinderConfig.Y_RANGE,
            margin=0.001
        )

        # 权重可后续再调
        layout_penalty = 10.0 * overlap_pen + 10.0 * bound_pen

        # 可选：轻微惩罚激活圆柱过多，避免无脑全开
        active_count = int(np.sum(r > 0))
        sparse_penalty = 0

        # 可选：轻微惩罚布局过于聚团
        cluster_penalty = 0.0
        active_idx = np.where(r > 0)[0]
        for ii in range(len(active_idx)):
            for jj in range(ii + 1, len(active_idx)):
                i = active_idx[ii]
                j = active_idx[jj]
                dij = np.hypot(x[i] - x[j], y[i] - y[j])
                cluster_penalty += 1.0 / (dij + 1e-3)

        cluster_penalty *= 0.01

        # ---------------------------
        # 2) 合法或轻微非法布局 -> 跑完整条轨迹
        # ---------------------------
        total_reward = 0.0
        full_history = [tuple(self.ctx.particle_pos.copy())]
        terminated_early = False

        # 每个物理步都和目标轨迹当前点比较
        for t in range(1, TrainingSettingConfig.EPISODE_LENGTH + 1):
            self.ctx.current_step = t

            _, terminated, extra_info = self.env.step(action=None)

            step_history = extra_info.get("history_pos", [])
            if len(step_history) > 0:
                full_history.extend(step_history)

            cur_pos = np.array(
                [self.env.particle.pos_x, self.env.particle.pos_y],
                dtype=np.float32
            )
            self.ctx.particle_pos = cur_pos.copy()

            cur_goal = self.ctx.goal
            cur_dist = float(np.linalg.norm(cur_pos - cur_goal))

            # 基础误差项：越靠近目标点越好
            reward_t = -cur_dist

            # 进度奖励：比上一步更接近目标点就加分
            if self.prev_dist is not None:
                reward_t += 2.0 * (self.prev_dist - cur_dist)

            self.prev_dist = cur_dist
            total_reward += reward_t

            # 粒子撞柱 / 出界时，环境内部会 terminated
            if terminated:
                total_reward -= 20.0
                terminated_early = True
                break
        # ---------------------------
        # 3) 终点误差 + 位移不足惩罚
        # ---------------------------
        final_goal = np.array(self.ctx.path[-1], dtype=np.float32)
        final_pos = np.array(
            [self.env.particle.pos_x, self.env.particle.pos_y],
            dtype=np.float32
        )
        final_dist = float(np.linalg.norm(final_pos - final_goal))

        total_reward -= 10.0 * final_dist

        start_pos = np.array(full_history[0], dtype=np.float32)
        travel_dist = float(np.linalg.norm(final_pos - start_pos))
        min_required_travel = 0.03

        if travel_dist < min_required_travel:
            total_reward -= 20.0 * (min_required_travel - travel_dist)

        # ---------------------------
        # 3) 在整条轨迹 reward 基础上扣布局惩罚
        # ---------------------------
        total_reward -= layout_penalty
        # total_reward -= sparse_penalty
        total_reward -= cluster_penalty

        self.last_history = full_history
        self.episode_return = float(total_reward)

        obs = self._get_obs()

        # one-shot 模式：每次 step 就结束整个 episode
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
                self.env.max_n,
                self.env.fixed_omega,
                dtype=np.float32
            ).tolist(),
            "terminated_early": bool(terminated_early),
            "layout_penalty": float(layout_penalty),
            "overlap_penalty": float(overlap_pen),
            "boundary_penalty": float(bound_pen),
            "sparse_penalty": float(sparse_penalty),
            "cluster_penalty": float(cluster_penalty),
            "active_count": int(active_count),
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
            }
        }