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

from config import GlobalOmegaControlConfig, InflowConfig, RenderSettingConfig
from utils.calc import calculate_point_velocity, is_legal
from visualize_3x3_strokes import (
    fixed_grid_3x3,
    make_target_safe,
    min_clearance_to_cylinders,
    path_score,
    render_case,
    segment_hits_cylinder,
    target_stroke,
)


class StrokeOmegaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        stroke="soft_u",
        radius=0.010,
        pitch=0.060,
        inflow_u=0.002,
        inflow_v=0.0,
        target_clearance=0.012,
        omega_min_hz=-1.0,
        omega_max_hz=1.0,
        max_delta_hz=0.08,
        max_steps=260,
        substeps=8,
        dt=0.006,
    ):
        super().__init__()
        self.stroke = str(stroke).lower()
        self.radius = float(radius)
        self.pitch = float(pitch)
        self.inflow_u = float(inflow_u)
        self.inflow_v = float(inflow_v)
        self.target_clearance = float(target_clearance)
        self.omega_min = 2.0 * np.pi * float(omega_min_hz)
        self.omega_max = 2.0 * np.pi * float(omega_max_hz)
        self.max_delta_omega = 2.0 * np.pi * float(max_delta_hz)
        self.max_steps = int(max_steps)
        self.substeps = int(substeps)
        self.dt = float(dt)

        self.cyl_x, self.cyl_y = fixed_grid_3x3(pitch=self.pitch)
        self.radii = np.full(len(self.cyl_x), self.radius, dtype=np.float32)
        raw_target, raw_start = target_stroke(self.stroke, n=260)
        self.target, self.start = make_target_safe(
            raw_target,
            raw_start,
            self.cyl_x,
            self.cyl_y,
            self.radii,
            clearance=self.target_clearance,
            iters=100,
        )
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

    def _obs(self):
        idx, dist = self._nearest_target(self.pos)
        tgt = self.target[idx]
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
                float(idx) / float(max(1, len(self.target) - 1)),
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
        self.prev_idx, self.prev_dist = self._nearest_target(self.pos)
        self.prev_score = path_score(np.asarray(self.path, dtype=np.float32), self.target)
        return self._obs(), {}

    def step(self, action):
        raw = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        self.omega = float(np.clip(self.omega + raw * self.max_delta_omega, self.omega_min, self.omega_max))
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
            if self.pos[0] > 0.15 or abs(self.pos[1]) > 0.15:
                out = True
                break

        idx, dist = self._nearest_target(self.pos)
        idx_delta = int(idx - self.prev_idx)
        progress = float(idx_delta) / float(max(1, len(self.target) - 1))
        coverage = float(idx) / float(max(1, len(self.target) - 1))
        tangent = self._target_tangent(idx)
        speed = float(np.linalg.norm(self.vel))
        tangent_align = 0.0
        if speed > 1e-8:
            tangent_align = float(np.dot(self.vel / speed, tangent))
        score_now = path_score(np.asarray(self.path, dtype=np.float32), self.target)
        score_gain = float(self.prev_score - score_now)
        if not np.isfinite(score_gain):
            score_gain = 0.0
        dist_norm = min(3.0, dist / 0.030)
        reverse_pen = max(0.0, -float(idx_delta)) / float(max(1, len(self.target) - 1))
        reward = 35.0 * progress
        reward += 10.0 * score_gain
        reward += 0.08 * tangent_align
        reward -= 0.22 * dist_norm
        reward -= 18.0 * reverse_pen
        reward -= 0.015
        reward -= 0.02 * raw * raw

        done = False
        success = False
        if collision:
            reward -= 80.0
            done = True
        elif out:
            reward -= 20.0 * (1.0 - coverage)
            done = True
        elif idx >= int(0.92 * (len(self.target) - 1)) and dist < 0.018:
            reward += 100.0
            reward += 30.0 * max(0.0, 0.05 - score_now) / 0.05
            done = True
            success = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            reward -= 20.0 * min(1.0, dist / 0.06)
            reward += 35.0 * coverage
            reward += 35.0 * max(0.0, 0.10 - min(score_now, 0.10)) / 0.10
        reward = float(np.nan_to_num(reward, nan=-100.0, posinf=100.0, neginf=-100.0))

        self.prev_idx = idx
        self.prev_dist = dist
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
            "tangent_align": float(tangent_align),
            "dist_norm": float(dist_norm),
            "omega": float(self.omega),
            "omega_hz": float(self.omega / (2.0 * np.pi)),
            "omega_trace": list(self.omega_trace),
            "path": [p.tolist() for p in self.path],
            "final_pos": self.pos.tolist(),
            "old_pos": old_pos.tolist(),
        }
        return self._obs(), float(reward), bool(done), False, info

    def rollout_deterministic(self, model, vecnorm=None):
        obs, _ = self.reset()
        if vecnorm is not None:
            obs_in = vecnorm.normalize_obs(obs[None, :])
        else:
            obs_in = obs[None, :]
        total = 0.0
        info = {}
        for _ in range(self.max_steps):
            action, _ = model.predict(obs_in, deterministic=True)
            obs, reward, done, _, info = self.step(action)
            total += float(reward)
            if vecnorm is not None:
                obs_in = vecnorm.normalize_obs(obs[None, :])
            else:
                obs_in = obs[None, :]
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


class StrokeEvalCallback(BaseCallback):
    def __init__(self, eval_env, out_dir, eval_freq=10000, verbose=1):
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
        metric = reward + 50.0 * score
        prefix = f"eval_t{self.num_timesteps}_R{reward:.1f}_score{float(info.get('path_score', 0.0)):.3f}"
        save_rollout_artifacts(self.eval_env, info, self.out_dir, prefix)
        if self.verbose:
            print(
                f"[StrokeEval] t={self.num_timesteps} reward={reward:.2f} "
                f"path_score={float(info.get('path_score', 0.0)):.4f} "
                f"progress={float(info.get('progress', 0.0)):.3f} "
                f"align={float(info.get('tangent_align', 0.0)):.2f} "
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
            print(f"[StrokeBest] updated | metric={metric:.2f} | saved to {best_dir}")
        return True


def build_run_tag(args):
    return (
        f"3x3_{str(args.stroke).lower()}_r{float(args.radius):.3f}_"
        f"uin{float(args.inflow_u):.3f}_om{float(args.omega_min_hz):+.1f}_{float(args.omega_max_hz):+.1f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3x3 fixed-radius stroke shaping with real-time omega control.")
    parser.add_argument("--stroke", default="soft_u")
    parser.add_argument("--radius", type=float, default=0.010)
    parser.add_argument("--pitch", type=float, default=0.060)
    parser.add_argument("--inflow-u", type=float, default=float(InflowConfig.U_IN))
    parser.add_argument("--inflow-v", type=float, default=0.0)
    parser.add_argument("--target-clearance", type=float, default=0.012)
    parser.add_argument("--omega-min-hz", type=float, default=-1.0)
    parser.add_argument("--omega-max-hz", type=float, default=1.0)
    parser.add_argument("--max-delta-hz", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.006)
    parser.add_argument("--total-timesteps", type=int, default=300000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default="models/stroke")
    parser.add_argument("--run-alias", default="")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda", help="SB3 device, e.g. cuda, cuda:0, cpu, or auto.")
    return parser.parse_args()


def main():
    args = parse_args()
    tag = build_run_tag(args)
    if len(str(args.run_alias).strip()) > 0:
        tag = f"{tag}__{str(args.run_alias).strip()}"
    out_dir = Path(args.out_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        return StrokeOmegaEnv(
            stroke=args.stroke,
            radius=args.radius,
            pitch=args.pitch,
            inflow_u=args.inflow_u,
            inflow_v=args.inflow_v,
            target_clearance=args.target_clearance,
            omega_min_hz=args.omega_min_hz,
            omega_max_hz=args.omega_max_hz,
            max_delta_hz=args.max_delta_hz,
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
        tensorboard_log=str(Path("tb_logs") / "stroke"),
        seed=int(args.seed),
    )
    print(
        f"[StrokeTrain] out={out_dir} stroke={args.stroke} radius={args.radius} "
        f"omega=({args.omega_min_hz},{args.omega_max_hz})Hz max_delta={args.max_delta_hz}Hz"
    )
    cb = StrokeEvalCallback(eval_env=eval_env, out_dir=out_dir, eval_freq=args.eval_freq)
    model.learn(total_timesteps=int(args.total_timesteps), callback=cb, tb_log_name=tag)
    model.save(out_dir / "final_model")
    vec_env.save(out_dir / "vecnormalize_final.pkl")
    reward, info = eval_env.rollout_deterministic(model, vecnorm=vec_env)
    save_rollout_artifacts(eval_env, info, out_dir, "final")
    print(f"[StrokeTrain] final reward={reward:.2f} path_score={float(info.get('path_score', 0.0)):.4f}")
    vec_env.close()


if __name__ == "__main__":
    main()
