import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import InflowConfig, RenderSettingConfig
from utils.calc import calculate_point_velocity, is_legal
from visualize_3x3_strokes import (
    make_target_safe,
    min_clearance_to_cylinders,
    path_score,
    render_case,
    segment_hits_cylinder,
)


WEST_STROKES = ("w", "e", "s", "t")

DEFAULT_STROKE = "w"
DEFAULT_RADIUS = 0.010
DEFAULT_PITCH = None
DEFAULT_INFLOW_U = max(0.002, float(InflowConfig.U_IN))
DEFAULT_INFLOW_V = 0.0
DEFAULT_TARGET_CLEARANCE = 0.004
DEFAULT_OMEGA_MIN_HZ = -3.0
DEFAULT_OMEGA_MAX_HZ = 3.0
DEFAULT_MAX_STEPS = 900
DEFAULT_SUBSTEPS = 24
DEFAULT_DT = 0.006
DEFAULT_TOTAL_TIMESTEPS = 300000
DEFAULT_EVAL_FREQ = 10000
DEFAULT_SEED = 0
DEFAULT_OUT_ROOT = "models/stroke/west_2x2"
DEFAULT_RUN_ALIAS = ""
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_DEVICE = "cuda"
TRACKING_DIST_SCALE = 0.018
TRACKING_PATH_SCORE_SCALE = 0.060
WAYPOINT_COUNT = 24
WAYPOINT_RADIUS = 0.011
WAYPOINT_DIST_SCALE = 0.026
OMEGA_ACTION_GAMMA = 2.0
OMEGA_CHANGE_PENALTY = 0.0
OMEGA_MAG_PENALTY = 0.015


def fixed_grid_2x2(pitch: float, center=(0.0, 0.0)):
    """Return 2x2 cylinder centers: top-left, top-right, bottom-left, bottom-right."""
    cx, cy = float(center[0]), float(center[1])
    half = 0.5 * float(pitch)
    pts = [
        (cx - half, cy + half),
        (cx + half, cy + half),
        (cx - half, cy - half),
        (cx + half, cy - half),
    ]
    return (
        np.asarray([p[0] for p in pts], dtype=np.float32),
        np.asarray([p[1] for p in pts], dtype=np.float32),
    )


def smooth_polyline(points, n=260, smooth_iters=8):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    seg = pts[1:] - pts[:-1]
    seg_len = np.sqrt(np.sum(seg * seg, axis=1))
    dist = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = max(float(dist[-1]), 1e-8)
    samples = np.linspace(0.0, total, int(n), dtype=np.float32)
    out = np.empty((int(n), 2), dtype=np.float32)
    out[:, 0] = np.interp(samples, dist, pts[:, 0])
    out[:, 1] = np.interp(samples, dist, pts[:, 1])
    for _ in range(int(smooth_iters)):
        old = out.copy()
        out[1:-1] = 0.20 * old[:-2] + 0.60 * old[1:-1] + 0.20 * old[2:]
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out.astype(np.float32)


def cubic_bezier(p0, p1, p2, p3, n=64):
    t = np.linspace(0.0, 1.0, int(n), dtype=np.float32)[:, None]
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    p3 = np.asarray(p3, dtype=np.float32)
    return (
        (1.0 - t) ** 3 * p0
        + 3.0 * (1.0 - t) ** 2 * t * p1
        + 3.0 * (1.0 - t) * t**2 * p2
        + t**3 * p3
    ).astype(np.float32)


def sample_waypoints(path: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    count = max(2, int(count))
    if pts.shape[0] <= count:
        return pts.copy()
    seg = pts[1:] - pts[:-1]
    seg_len = np.sqrt(np.sum(seg * seg, axis=1))
    dist = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = max(float(dist[-1]), 1e-8)
    samples = np.linspace(0.0, total, count, dtype=np.float32)
    out = np.empty((count, 2), dtype=np.float32)
    out[:, 0] = np.interp(samples, dist, pts[:, 0])
    out[:, 1] = np.interp(samples, dist, pts[:, 1])
    return out.astype(np.float32)


def target_west_stroke(stroke: str, n: int = 260):
    stroke = str(stroke).lower()
    if stroke == "w":
        # Match the hand-drawn target: two U-shaped cups around the top rotors.
        # The valleys stay between the top and bottom rows to avoid projection kinks.
        valley_y = 0.004
        segments = [
            ((-0.050, 0.058), (-0.050, 0.050), (-0.046, 0.010), (-0.037, valley_y)),
            ((-0.037, valley_y), (-0.014, valley_y), (-0.001, 0.048), (0.000, 0.064)),
            ((0.000, 0.064), (0.001, 0.048), (0.014, valley_y), (0.037, valley_y)),
            ((0.037, valley_y), (0.046, 0.010), (0.050, 0.050), (0.050, 0.058)),
        ]
        per_seg = max(16, int(np.ceil(float(n) / len(segments))))
        parts = [cubic_bezier(*seg, n=per_seg) for seg in segments]
        target = np.concatenate([part[:-1] for part in parts[:-1]] + [parts[-1]], axis=0)
    elif stroke == "e":
        # One-stroke lowercase-e style at the same scale as the 2x2 west W:
        # middle bar goes right first, then curls upward around the outer loop.
        segments = [
            ((-0.040, 0.006), (-0.014, 0.008), (0.030, 0.006), (0.047, 0.022)),
            ((0.047, 0.022), (0.056, 0.036), (0.048, 0.050), (0.018, 0.050)),
            ((0.018, 0.050), (-0.024, 0.050), (-0.052, 0.030), (-0.050, 0.004)),
            ((-0.050, 0.004), (-0.060, -0.016), (-0.054, -0.044), (-0.022, -0.050)),
            ((-0.022, -0.050), (0.010, -0.056), (0.046, -0.052), (0.058, -0.036)),
        ]
        per_seg = max(16, int(np.ceil(float(n) / len(segments))))
        parts = [cubic_bezier(*seg, n=per_seg) for seg in segments]
        target = np.concatenate([part[:-1] for part in parts[:-1]] + [parts[-1]], axis=0)
    elif stroke == "s":
        ss = np.linspace(0.0, 1.0, int(n), dtype=np.float32)
        xs = -0.095 + 0.190 * ss
        ys = 0.065 - 0.130 * ss + 0.040 * np.sin(2.0 * np.pi * ss)
        target = np.stack([xs, ys], axis=1).astype(np.float32)
    elif stroke == "t":
        # One-stroke t-like path: top cross first, then a downward stem and small hook.
        pts = [
            (-0.095, 0.030),
            (-0.018, 0.030),
            (0.000, 0.082),
            (0.018, 0.030),
            (0.006, -0.090),
            (0.073, -0.055),
        ]
        target = smooth_polyline(pts, n=n, smooth_iters=8)
    else:
        raise ValueError(f"Unknown west stroke: {stroke}. Choose from {', '.join(WEST_STROKES)}")
    return target.astype(np.float32), target[0].astype(np.float32)


class West2x2OmegaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        stroke=DEFAULT_STROKE,
        radius=DEFAULT_RADIUS,
        pitch=DEFAULT_PITCH,
        inflow_u=DEFAULT_INFLOW_U,
        inflow_v=DEFAULT_INFLOW_V,
        target_clearance=DEFAULT_TARGET_CLEARANCE,
        omega_min_hz=DEFAULT_OMEGA_MIN_HZ,
        omega_max_hz=DEFAULT_OMEGA_MAX_HZ,
        max_steps=DEFAULT_MAX_STEPS,
        substeps=DEFAULT_SUBSTEPS,
        dt=DEFAULT_DT,
    ):
        super().__init__()
        self.stroke = str(stroke).lower()
        self.radius = float(radius)
        self.pitch = 6.0 * self.radius if pitch is None else float(pitch)
        self.inflow_u = float(inflow_u)
        self.inflow_v = float(inflow_v)
        self.target_clearance = float(target_clearance)
        self.omega_min = 2.0 * np.pi * float(omega_min_hz)
        self.omega_max = 2.0 * np.pi * float(omega_max_hz)
        self.max_steps = int(max_steps)
        self.substeps = int(substeps)
        self.dt = float(dt)

        self.cyl_x, self.cyl_y = fixed_grid_2x2(pitch=self.pitch)
        self.radii = np.full(len(self.cyl_x), self.radius, dtype=np.float32)
        raw_target, raw_start = target_west_stroke(self.stroke, n=280)
        self.target, self.start = make_target_safe(
            raw_target,
            raw_start,
            self.cyl_x,
            self.cyl_y,
            self.radii,
            clearance=self.target_clearance,
            iters=100,
        )
        self.waypoints = sample_waypoints(self.target, WAYPOINT_COUNT)
        self.centers = np.stack([self.cyl_x, self.cyl_y], axis=1).astype(np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(10,), dtype=np.float32)
        self.reset()

    def _nearest_target(self, pos):
        d = self.target - np.asarray(pos, dtype=np.float32)[None, :]
        d2 = np.sum(d * d, axis=1)
        idx = int(np.argmin(d2))
        dist = float(np.sqrt(d2[idx]))
        return idx, dist

    def _target_tangent(self, idx):
        idx = int(np.clip(idx, 0, len(self.target) - 1))
        i0 = max(0, idx - 1)
        i1 = min(len(self.target) - 1, idx + 1)
        tang = self.target[i1] - self.target[i0]
        n = float(np.linalg.norm(tang))
        if n < 1e-8:
            return np.array([1.0, 0.0], dtype=np.float32)
        return (tang / n).astype(np.float32)

    def _action_to_omega(self, raw: float) -> float:
        raw = float(np.clip(raw, -1.0, 1.0))
        gamma = max(1.0, float(OMEGA_ACTION_GAMMA))
        if self.omega_min < 0.0 < self.omega_max:
            if raw >= 0.0:
                return float((raw ** gamma) * self.omega_max)
            return float(-((-raw) ** gamma) * abs(self.omega_min))
        return float(self.omega_min + 0.5 * (raw + 1.0) * (self.omega_max - self.omega_min))

    def _obs(self):
        idx, _ = self._nearest_target(self.pos)
        wp_idx = int(np.clip(getattr(self, "wp_idx", 1), 0, len(self.waypoints) - 1))
        tgt = self.waypoints[wp_idx]
        end = self.target[-1]
        omega_scale = max(abs(self.omega_min), abs(self.omega_max), 1e-8)
        return np.array(
            [
                self.pos[0],
                self.pos[1],
                self.vel[0],
                self.vel[1],
                self.omega / omega_scale,
                tgt[0] - self.pos[0],
                tgt[1] - self.pos[1],
                end[0] - self.pos[0],
                end[1] - self.pos[1],
                float(wp_idx) / float(max(1, len(self.waypoints) - 1)),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start.astype(np.float32).copy()
        self.vel = np.zeros(2, dtype=np.float32)
        self.omega = 0.0
        self.step_count = 0
        self.path = [self.pos.copy()]
        self.omega_trace = [float(self.omega)]
        self.wp_idx = 1 if len(self.waypoints) > 1 else 0
        self.prev_wp_dist = float(np.linalg.norm(self.pos - self.waypoints[self.wp_idx]))
        self.prev_idx, self.prev_dist = self._nearest_target(self.pos)
        self.prev_score = path_score(np.asarray(self.path, dtype=np.float32), self.target)
        return self._obs(), {}

    def step(self, action):
        raw = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        prev_omega = float(self.omega)
        self.omega = self._action_to_omega(raw)
        self.omega_trace.append(float(self.omega))
        omegas = np.full(len(self.cyl_x), float(self.omega), dtype=np.float32)
        collision = False
        out = False

        old_pos = self.pos.copy()
        for _ in range(max(1, self.substeps)):
            vx, vy = calculate_point_velocity(
                float(self.pos[0]), float(self.pos[1]), self.cyl_x, self.cyl_y, self.radii, omegas
            )
            self.vel = np.array([float(vx) + self.inflow_u, float(vy) + self.inflow_v], dtype=np.float32)
            nxt = self.pos + self.vel * self.dt
            if segment_hits_cylinder(self.pos, nxt, self.centers, self.radii) or not is_legal(
                nxt,
                centers=self.centers,
                radii=self.radii,
                xlim=RenderSettingConfig.X_LIM,
                ylim=RenderSettingConfig.Y_LIM,
            ):
                collision = True
                break
            self.pos = nxt.astype(np.float32)
            self.path.append(self.pos.copy())
            if self.pos[0] > 0.16 or abs(self.pos[1]) > 0.16:
                out = True
                break

        idx, dist = self._nearest_target(self.pos)
        idx_delta = int(idx - self.prev_idx)
        coverage = float(idx) / float(max(1, len(self.target) - 1))
        wp_idx_before = int(np.clip(self.wp_idx, 0, len(self.waypoints) - 1))
        wp = self.waypoints[wp_idx_before]
        wp_vec = wp - self.pos
        wp_dist = float(np.linalg.norm(wp_vec))
        if wp_dist > 1e-8:
            desired_dir = (wp_vec / wp_dist).astype(np.float32)
        else:
            desired_dir = self._target_tangent(idx)
        speed = float(np.linalg.norm(self.vel))
        direction_align = 0.0
        direction_speed = 0.0
        if speed > 1e-8:
            direction_align = float(np.dot(self.vel / speed, desired_dir))
            direction_speed = float(np.dot(self.vel, desired_dir))
        score_now = path_score(np.asarray(self.path, dtype=np.float32), self.target)
        score_gain = float(self.prev_score - score_now)
        if not np.isfinite(score_gain):
            score_gain = 0.0
        dist_norm = min(3.0, dist / max(float(TRACKING_DIST_SCALE), 1e-8))
        wp_dist_norm = min(3.0, wp_dist / max(float(WAYPOINT_DIST_SCALE), 1e-8))
        closeness = float(np.exp(-min(6.0, dist_norm * dist_norm)))
        wp_closeness = float(np.exp(-min(6.0, wp_dist_norm * wp_dist_norm)))
        reverse_pen = max(0.0, -float(idx_delta)) / float(max(1, len(self.target) - 1))
        wp_progress = max(0.0, float(self.prev_wp_dist) - wp_dist)
        omega_scale = max(abs(float(self.omega_min)), abs(float(self.omega_max)), 1e-8)
        omega_delta_norm = float((self.omega - prev_omega) / omega_scale)
        omega_norm = float(self.omega / omega_scale)

        reward = 120.0 * wp_progress
        reward += 22.0 * score_gain
        reward += 0.45 * direction_align * max(0.15, wp_closeness)
        reward += 22.0 * max(0.0, direction_speed) * max(0.15, wp_closeness)
        reward -= 0.55 * dist_norm
        reward -= 0.42 * wp_dist_norm
        reward -= 20.0 * reverse_pen
        if wp_progress <= 1e-5:
            reward -= 0.08
        reward -= 0.10 * min(3.0, score_now / max(float(TRACKING_PATH_SCORE_SCALE), 1e-8))
        reward -= 0.010
        reward -= 0.012 * raw * raw
        reward -= float(OMEGA_CHANGE_PENALTY) * omega_delta_norm * omega_delta_norm
        reward -= float(OMEGA_MAG_PENALTY) * omega_norm * omega_norm

        reached_wp = False
        advanced_wp = 0
        future_wp = self.waypoints[self.wp_idx :]
        if future_wp.shape[0] > 0:
            future_dist = np.linalg.norm(future_wp - self.pos[None, :], axis=1)
            hit_offsets = np.where(future_dist <= float(WAYPOINT_RADIUS))[0]
            if hit_offsets.size > 0:
                new_wp_idx = min(len(self.waypoints) - 1, self.wp_idx + int(np.max(hit_offsets)) + 1)
                advanced_wp = max(0, int(new_wp_idx - self.wp_idx))
                if advanced_wp > 0:
                    self.wp_idx = int(new_wp_idx)
                    reached_wp = True
                    wp = self.waypoints[self.wp_idx]
                    wp_dist = float(np.linalg.norm(wp - self.pos))
        if reached_wp:
            reward += 4.0 * float(advanced_wp)

        done = False
        success = False
        if collision:
            reward -= 85.0
            done = True
        elif out:
            reward -= 24.0 * (1.0 - coverage)
            done = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            reward -= 22.0 * min(2.0, score_now / max(float(TRACKING_PATH_SCORE_SCALE), 1e-8))

        reward = float(np.nan_to_num(reward, nan=-100.0, posinf=100.0, neginf=-100.0))
        self.prev_idx = idx
        self.prev_dist = dist
        self.prev_wp_dist = float(np.linalg.norm(self.pos - self.waypoints[self.wp_idx]))
        self.prev_score = score_now
        info = {
            "stroke": self.stroke,
            "success": bool(success),
            "collision": bool(collision),
            "out": bool(out),
            "nearest_idx": int(idx),
            "nearest_dist": float(dist),
            "path_score": float(score_now),
            "progress": float(coverage),
            "waypoint_idx": int(self.wp_idx),
            "waypoint_total": int(len(self.waypoints)),
            "waypoint_dist": float(self.prev_wp_dist),
            "waypoint_reached": bool(reached_wp),
            "tangent_align": float(direction_align),
            "tangent_speed": float(direction_speed),
            "tracking_closeness": float(closeness),
            "dist_norm": float(dist_norm),
            "omega": float(self.omega),
            "omega_hz": float(self.omega / (2.0 * np.pi)),
            "omega_delta_norm": float(omega_delta_norm),
            "omega_trace": list(self.omega_trace),
            "path": [p.tolist() for p in self.path],
            "final_pos": self.pos.tolist(),
            "old_pos": old_pos.tolist(),
        }
        return self._obs(), reward, bool(done), False, info

    def rollout_deterministic(self, model, vecnorm=None):
        obs, _ = self.reset()
        obs_in = vecnorm.normalize_obs(obs[None, :]) if vecnorm is not None else obs[None, :]
        total = 0.0
        info = {}
        for _ in range(self.max_steps):
            action, _ = model.predict(obs_in, deterministic=True)
            obs, reward, done, _, info = self.step(action)
            total += float(reward)
            obs_in = vecnorm.normalize_obs(obs[None, :]) if vecnorm is not None else obs[None, :]
            if done:
                break
        return float(total), info


def save_rollout_artifacts(env, info, out_dir, prefix):
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    traces_dir = out_dir / "traces"
    frames_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    path = np.asarray(info.get("path", env.path), dtype=np.float32)
    omega = float(info.get("omega", env.omega))
    png = frames_dir / f"{prefix}.png"
    render_case(
        png,
        env.stroke,
        env.target,
        env.start,
        path,
        env.cyl_x,
        env.cyl_y,
        env.radii,
        omega,
        env.inflow_u,
        env.inflow_v,
        min_target_clearance=min_clearance_to_cylinders(env.target, env.cyl_x, env.cyl_y, env.radii),
        grid_name="2x2 west",
    )

    csv_path = traces_dir / f"{prefix}_omega_trace.csv"
    omega_trace = list(info.get("omega_trace", env.omega_trace))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "omega_rad_s", "frequency_hz"])
        writer.writeheader()
        for i, om in enumerate(omega_trace):
            writer.writerow(
                {
                    "step": i,
                    "omega_rad_s": float(om),
                    "frequency_hz": float(om) / (2.0 * np.pi),
                }
            )
    return png, csv_path


class West2x2EvalCallback(BaseCallback):
    def __init__(self, eval_env, out_dir, eval_freq=DEFAULT_EVAL_FREQ, verbose=1):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.out_dir = Path(out_dir)
        self.eval_freq = int(eval_freq)
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        vecnorm = self.model.get_env()
        reward, info = self.eval_env.rollout_deterministic(self.model, vecnorm=vecnorm)
        score = -float(info.get("path_score", 1e9))
        metric = reward + 220.0 * score
        prefix = f"eval_t{self.num_timesteps}_R{reward:.1f}_score{float(info.get('path_score', 0.0)):.3f}"
        save_rollout_artifacts(self.eval_env, info, self.out_dir, prefix)
        if self.verbose:
            print(
                f"[West2x2Eval] t={self.num_timesteps} reward={reward:.2f} "
                f"path_score={float(info.get('path_score', 0.0)):.4f} "
                f"progress={float(info.get('progress', 0.0)):.3f} "
                f"wp={int(info.get('waypoint_idx', 0))}/{int(info.get('waypoint_total', 1)) - 1} "
                f"wp_dist={float(info.get('waypoint_dist', 0.0)):.4f} "
                f"close={float(info.get('tracking_closeness', 0.0)):.2f} "
                f"align={float(info.get('tangent_align', 0.0)):.2f} "
                f"vt={float(info.get('tangent_speed', 0.0)):.4f} "
                f"domega={float(info.get('omega_delta_norm', 0.0)):.2f} "
                f"f={float(info.get('omega_hz', 0.0)):.3f}Hz"
            )
        if metric > self.best_reward:
            self.best_reward = metric
            best_dir = self.out_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(best_dir / "best_model")
            try:
                vecnorm.save(best_dir / "vecnormalize_best.pkl")
            except Exception:
                pass
            save_rollout_artifacts(self.eval_env, info, best_dir, "best")
            print(f"[West2x2Best] updated | metric={metric:.2f} | saved to {best_dir}")
        return True


def build_run_tag(args, pitch):
    return (
        f"2x2_west_{str(args.stroke).lower()}_r{float(args.radius):.3f}_"
        f"p{float(pitch):.3f}_uin{float(args.inflow_u):.3f}_"
        f"om{float(args.omega_min_hz):+.1f}_{float(args.omega_max_hz):+.1f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 2x2 fixed-radius west-letter subtasks with inflow and real-time global omega control."
    )
    parser.add_argument("--stroke", default=DEFAULT_STROKE, choices=WEST_STROKES)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument(
        "--pitch",
        type=float,
        default=DEFAULT_PITCH,
        help="Center distance. Default is 6*radius, i.e. 3 cylinder diameters.",
    )
    parser.add_argument("--inflow-u", type=float, default=DEFAULT_INFLOW_U)
    parser.add_argument("--inflow-v", type=float, default=DEFAULT_INFLOW_V)
    parser.add_argument("--target-clearance", type=float, default=DEFAULT_TARGET_CLEARANCE)
    parser.add_argument("--omega-min-hz", type=float, default=DEFAULT_OMEGA_MIN_HZ)
    parser.add_argument("--omega-max-hz", type=float, default=DEFAULT_OMEGA_MAX_HZ)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-alias", default=DEFAULT_RUN_ALIAS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="SB3 device, e.g. cuda, cuda:0, cpu, or auto.")
    return parser.parse_args()


def main():
    args = parse_args()
    pitch = 6.0 * float(args.radius) if args.pitch is None else float(args.pitch)
    tag = build_run_tag(args, pitch)
    if len(str(args.run_alias).strip()) > 0:
        tag = f"{tag}__{str(args.run_alias).strip()}"
    out_dir = Path(args.out_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        return West2x2OmegaEnv(
            stroke=args.stroke,
            radius=args.radius,
            pitch=pitch,
            inflow_u=args.inflow_u,
            inflow_v=args.inflow_v,
            target_clearance=args.target_clearance,
            omega_min_hz=args.omega_min_hz,
            omega_max_hz=args.omega_max_hz,
            max_steps=args.max_steps,
            substeps=args.substeps,
            dt=args.dt,
        )

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0)
    eval_env = make_env()
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=float(args.learning_rate),
        gamma=0.995,
        ent_coef=0.002,
        device=str(args.device),
        tensorboard_log=str(Path("tb_logs") / "stroke_west_2x2"),
        seed=int(args.seed),
    )
    print(
        f"[West2x2Train] out={out_dir} stroke={args.stroke} radius={args.radius} pitch={pitch} "
        f"center_distance/diameter={pitch / max(2.0 * float(args.radius), 1e-8):.2f} "
        f"inflow=({args.inflow_u},{args.inflow_v}) omega=({args.omega_min_hz},{args.omega_max_hz})Hz "
        f"control=absolute_omega"
    )
    cb = West2x2EvalCallback(eval_env=eval_env, out_dir=out_dir, eval_freq=args.eval_freq)
    model.learn(total_timesteps=int(args.total_timesteps), callback=cb, tb_log_name=tag)
    model.save(out_dir / "final_model")
    vec_env.save(out_dir / "vecnormalize_final.pkl")
    reward, info = eval_env.rollout_deterministic(model, vecnorm=vec_env)
    save_rollout_artifacts(eval_env, info, out_dir, "final")
    print(f"[West2x2Train] final reward={reward:.2f} path_score={float(info.get('path_score', 0.0)):.4f}")
    vec_env.close()


if __name__ == "__main__":
    main()
