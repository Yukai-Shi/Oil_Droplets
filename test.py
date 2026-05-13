import argparse
import csv
import os
import platform
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import (
    DynamicOmegaControlConfig,
    LayoutModeConfig,
    LogicBoxConfig,
    TrainingSettingConfig,
    get_logic_multi_route_pairs,
    get_logic_multi_route_sets,
    is_dynamic_omega_design_mode,
    is_dynamic_omega_mode,
    is_dynamic_omega_radius_design_mode,
    strip_dynamic_omega_suffix,
)
from envs.FluidEnv import FluidEnv
from envs.Renderer import FluidRenderer
from envs.SharedXYPolicy import SharedGeometryActorCriticPolicy, SharedXYActorCriticPolicy  # noqa: F401
from envs.Wrapper import FollowerEnv, TaskContext


# For MLP policies in SB3, CPU inference is typically more stable/faster than GPU.
DEVICE = "cpu"
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    DEVICE = "mps"


def parse_layout_mode(layout_mode: str):
    mode, _ = strip_dynamic_omega_suffix(str(layout_mode))
    if mode.endswith("_inflow_u_fixed"):
        return mode[:-15], True
    # Backward-compatible alias: *_inflow_u == *_inflow_u_fixed
    if mode.endswith("_inflow_u"):
        return mode[:-9], True
    if mode.endswith("_inflow"):
        return mode[:-7], True
    return mode, False


def _append_run_alias(tag: str) -> str:
    alias = str(getattr(TrainingSettingConfig, "RUN_ALIAS", "")).strip()
    if len(alias) == 0:
        return tag
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in alias)
    safe = safe.strip("._")
    if len(safe) == 0:
        return tag
    return f"{tag}__{safe}"


def _logic_bounds_suffix() -> str:
    mode = str(getattr(LogicBoxConfig, "BOUNDS_MODE", "local_box")).strip().lower()
    if mode in {"global", "global_field", "full_field", "render_field", "world"}:
        return "bnd_global"
    return "bnd_local"


def build_task_tag(layout_mode: str) -> str:
    base_mode, _ = parse_layout_mode(layout_mode)
    if base_mode == "logic_box_layout":
        bounds_suffix = _logic_bounds_suffix()
        route_mode = str(getattr(LogicBoxConfig, "ROUTE_MODE", "single")).strip().lower()
        if route_mode in {"multi_map_switch", "multi_switch", "mapping_switch"}:
            route_sets = get_logic_multi_route_sets()
            first_pairs = route_sets[0] if len(route_sets) > 0 else get_logic_multi_route_pairs()
            first_tag = "_".join([f"{str(s).upper()}to{str(t).upper()}" for s, t in first_pairs])
            return _append_run_alias(
                f"{layout_mode}_{TrainingSettingConfig.PATH_TYPE}_multisw{len(route_sets)}_{first_tag}__{bounds_suffix}"
            )
        if route_mode in {"multi", "multi_map", "multi_route", "mapping"}:
            pairs = get_logic_multi_route_pairs()
            pair_tag = "_".join([f"{str(s).upper()}to{str(t).upper()}" for s, t in pairs])
            return _append_run_alias(
                f"{layout_mode}_{TrainingSettingConfig.PATH_TYPE}_multi_{pair_tag}__{bounds_suffix}"
            )
        if route_mode in {"single_multi_target", "single_source_multi_target", "one_to_many", "one_to_three"}:
            src = str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")).upper()
            tgt_set = getattr(LogicBoxConfig, "TARGET_PORT_SET", ["R0", "R1", "R2"])
            tgt_tag = "".join([str(t).upper() for t in tgt_set])
            return _append_run_alias(
                f"{layout_mode}_{TrainingSettingConfig.PATH_TYPE}_{src}_to_{tgt_tag}__{bounds_suffix}"
            )
        src = str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")).upper()
        tgt = str(getattr(LogicBoxConfig, "TARGET_PORT", "R1")).upper()
        return _append_run_alias(
            f"{layout_mode}_{TrainingSettingConfig.PATH_TYPE}_{src}_to_{tgt}__{bounds_suffix}"
        )
    return _append_run_alias(f"{layout_mode}_{TrainingSettingConfig.PATH_TYPE}")


def default_model_paths(layout_mode: str):
    tag = build_task_tag(layout_mode)
    task_dir = os.path.join("models", "best", tag)
    return (
        os.path.join(task_dir, "best_model.zip"),
        os.path.join(task_dir, "vecnormalize_best.pkl"),
    )


def load_policy_action(
    model_path: str,
    vecnorm_path: str,
    layout_mode: str,
    logic_target_port: Optional[str] = None,
    logic_route_set_idx: Optional[int] = None,
):
    """
    Load trained policy and infer one one-shot action.
    """
    def _build_vec():
        _ctx = TaskContext()
        _fe = FluidEnv(
            start_pos=np.array([-0.025, 0.06], dtype=np.float32),
            layout_mode=layout_mode,
        )
        _env = FollowerEnv(fluid_env=_fe, ctx=_ctx, logic_seed_profile="eval")
        if logic_target_port is not None and hasattr(_env, "set_logic_forced_target_port"):
            _env.set_logic_forced_target_port(str(logic_target_port).upper())
        if logic_route_set_idx is not None and hasattr(_env, "set_logic_forced_route_set_idx"):
            _env.set_logic_forced_route_set_idx(int(logic_route_set_idx))
        _vec = DummyVecEnv([lambda: _env])
        if vecnorm_path and os.path.isfile(vecnorm_path):
            _vec = VecNormalize.load(vecnorm_path, _vec)
            _vec.training = False
            _vec.norm_reward = False
        return _env, _vec

    env, vec_env = _build_vec()
    try:
        model = PPO.load(model_path, env=vec_env, device=DEVICE)
    except Exception as exc:
        vec_env.close()
        base_mode, _ = parse_layout_mode(layout_mode)
        if (
            base_mode == "logic_box_layout"
            and bool(getattr(LogicBoxConfig, "FIXED_LAYOUT_ENABLE", False))
            and (not bool(getattr(LogicBoxConfig, "KEEP_XY_ACTION_WHEN_FIXED", False)))
        ):
            # Compatibility fallback for models trained with fixed-layout x/y/r action shape.
            LogicBoxConfig.KEEP_XY_ACTION_WHEN_FIXED = True
            print("[Compat] auto-enable KEEP_XY_ACTION_WHEN_FIXED=True and retry model load.")
            env, vec_env = _build_vec()
            model = PPO.load(model_path, env=vec_env, device=DEVICE)
        else:
            raise exc

    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    action = np.asarray(action).reshape(-1).astype(np.float32)
    physical_action = env._decode_action(action)
    vec_env.close()
    return physical_action


def run_long_rollout(
    physical_action: np.ndarray,
    layout_mode: str,
    rollout_steps: int,
    frame_stride: int,
    fps: int,
    output_gif: str,
    show_target_point: bool,
    show_target_path: bool,
    hide_logic_seeds: bool,
    logic_target_port: Optional[str] = None,
    logic_route_set_idx: Optional[int] = None,
):
    ctx = TaskContext()
    fe = FluidEnv(
        start_pos=np.array([-0.025, 0.06], dtype=np.float32),
        layout_mode=layout_mode,
    )
    env = FollowerEnv(fluid_env=fe, ctx=ctx, logic_seed_profile="eval")
    if logic_target_port is not None and hasattr(env, "set_logic_forced_target_port"):
        env.set_logic_forced_target_port(str(logic_target_port).upper())
    if logic_route_set_idx is not None and hasattr(env, "set_logic_forced_route_set_idx"):
        env.set_logic_forced_route_set_idx(int(logic_route_set_idx))
    renderer = FluidRenderer()

    # Reset and apply one-shot layout once.
    env.reset(seed=0)
    fe.apply_layout(physical_action)

    path_history = [tuple(ctx.particle_pos.tolist())]
    target_path = ctx.path.tolist()
    frames = []
    done_early = False

    for t in range(rollout_steps):
        ctx.current_step = min(t + 1, len(ctx.path) - 1)
        _, done, extra = fe.step(action=None)

        step_history = extra.get("history_pos", [])
        if len(step_history) > 0:
            path_history.extend([(float(px), float(py)) for px, py in step_history])

        ctx.particle_pos = np.array(
            [fe.particle.pos_x, fe.particle.pos_y],
            dtype=np.float32,
        )

        if t % max(1, frame_stride) == 0 or t == rollout_steps - 1 or done:
            scene = env.get_scene()
            if hide_logic_seeds:
                logic_box = scene.get("logic_box", None)
                if isinstance(logic_box, dict):
                    logic_box["seed_points"] = []
                    logic_box["seed_streamlines"] = []
            title = (
                f"t:{t + 1}/{rollout_steps} | "
                f"f:{(float(fe.fixed_omega) / (2.0 * np.pi)):.2f} Hz"
            )
            frame = renderer.render(
                scene=scene,
                follower_path=path_history,
                target_pos=ctx.goal if show_target_point else None,
                target_path=target_path if show_target_path else None,
                title=title,
                draw_flow=True,
            )
            frames.append(frame)

        if done:
            done_early = True
            break

    if len(frames) == 0:
        frame = renderer.render(
            scene=env.get_scene(),
            follower_path=path_history,
            target_pos=ctx.goal if show_target_point else None,
            target_path=target_path if show_target_path else None,
            title=f"f:{(float(fe.fixed_omega) / (2.0 * np.pi)):.2f} Hz",
            draw_flow=True,
        )
        frames.append(frame)

    for _ in range(12):
        frames.append(frames[-1])

    Path(output_gif).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_gif, frames, fps=fps)
    renderer.close()

    r = np.asarray(fe.cylinders_r, dtype=np.float32)
    base_mode, _ = parse_layout_mode(layout_mode)
    if base_mode == "logic_box_layout":
        killed_eps = float(getattr(LogicBoxConfig, "KILLED_EPS", 1e-6))
    else:
        killed_eps = 1e-12
    inactive_count = int(np.sum(r <= killed_eps))
    total_count = int(r.size)

    print(f"[Saved] {output_gif}")
    print(
        f"[Summary] frames={len(frames)}, path_points={len(path_history)}, "
        f"done_early={done_early}, final_pos=({ctx.particle_pos[0]:.4f}, {ctx.particle_pos[1]:.4f}), "
        f"omega={fe.fixed_omega:.4f}, inflow=({fe.inflow_u:.4f},{fe.inflow_v:.4f}), "
        f"dead={inactive_count}/{total_count}, killed_eps={killed_eps:.1e}"
    )


def _save_omega_trace_plot(records, output_png: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warn] omega trace plot skipped: {exc}")
        return

    steps = np.asarray([r["step"] for r in records], dtype=np.float32)
    freq = np.asarray([r["frequency_hz"] for r in records], dtype=np.float32)
    raw = np.asarray([r["raw_delta_action"] for r in records], dtype=np.float32)

    fig, ax1 = plt.subplots(figsize=(7.0, 3.0), dpi=160)
    ax1.plot(steps, freq, color="#1261a0", linewidth=2.0, label="frequency")
    ax1.set_xlabel("control step")
    ax1.set_ylabel("frequency (Hz)", color="#1261a0")
    ax1.tick_params(axis="y", labelcolor="#1261a0")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(steps, raw, color="#d9822b", linewidth=1.2, alpha=0.75, label="raw delta action")
    ax2.set_ylabel("raw delta action", color="#d9822b")
    ax2.tick_params(axis="y", labelcolor="#d9822b")
    ax2.set_ylim(-1.05, 1.05)

    target = records[-1].get("target_port", "")
    fig.suptitle(f"Omega Control Trace | target:{target}", fontsize=10)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def run_dynamic_omega_rollout(
    model_path: str,
    vecnorm_path: str,
    layout_mode: str,
    rollout_steps: int,
    frame_stride: int,
    fps: int,
    output_gif: str,
    show_target_point: bool,
    show_target_path: bool,
    hide_logic_seeds: bool,
    logic_target_port: Optional[str] = None,
    logic_route_set_idx: Optional[int] = None,
):
    ctx = TaskContext()
    fe = FluidEnv(
        start_pos=np.array([-0.025, 0.06], dtype=np.float32),
        layout_mode=layout_mode,
    )
    env = FollowerEnv(fluid_env=fe, ctx=ctx, logic_seed_profile="eval")
    if logic_target_port is not None and hasattr(env, "set_logic_forced_target_port"):
        env.set_logic_forced_target_port(str(logic_target_port).upper())
    if logic_route_set_idx is not None and hasattr(env, "set_logic_forced_route_set_idx"):
        env.set_logic_forced_route_set_idx(int(logic_route_set_idx))

    vec_env = DummyVecEnv([lambda: env])
    if vecnorm_path and os.path.isfile(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    model = PPO.load(model_path, env=vec_env, device=DEVICE)
    renderer = FluidRenderer()

    obs = vec_env.reset()
    path_history = [tuple(ctx.particle_pos.tolist())]
    target_path = ctx.path.tolist()
    frames = []
    total_reward = 0.0
    done_early = False
    last_info = {}
    omega_records = []
    max_steps = min(
        int(max(1, rollout_steps)),
        max(1, int(getattr(DynamicOmegaControlConfig, "MAX_STEPS", rollout_steps))),
    )

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action_flat = np.asarray(action, dtype=np.float32).reshape(-1)
        raw_delta_action = float(action_flat[-1]) if action_flat.size > 0 else 0.0
        obs, reward, done, info = vec_env.step(action)
        step_reward = float(reward[0]) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward)
        total_reward += step_reward
        last_info = info[0] if isinstance(info, (list, tuple)) else info

        hist = last_info.get("history_pos", [])
        if len(hist) > 0:
            path_history = [(float(px), float(py)) for px, py in hist]
        ctx.particle_pos = np.array(last_info.get("final_pos", ctx.particle_pos), dtype=np.float32)
        omega_now = float(last_info.get("global_omega", fe.fixed_omega))
        omega_records.append(
            {
                "step": int(t + 1),
                "omega_rad_s": omega_now,
                "frequency_hz": omega_now / (2.0 * np.pi),
                "raw_delta_action": raw_delta_action,
                "omega_delta_rad_s": float(last_info.get("omega_delta", np.nan)),
                "x": float(ctx.particle_pos[0]),
                "y": float(ctx.particle_pos[1]),
                "step_reward": step_reward,
                "total_reward": float(total_reward),
                "target_port": str(last_info.get("logic_target_port", logic_target_port)),
            }
        )

        done0 = bool(done[0]) if isinstance(done, (list, tuple, np.ndarray)) else bool(done)
        if t % max(1, frame_stride) == 0 or t == max_steps - 1 or done0:
            scene = env.get_scene()
            if done0 and isinstance(last_info, dict):
                scene["particle"] = {
                    "x": float(ctx.particle_pos[0]),
                    "y": float(ctx.particle_pos[1]),
                }
                scene["target"] = {
                    "x": float(target_path[-1][0]),
                    "y": float(target_path[-1][1]),
                }
                om = float(last_info.get("global_omega", fe.fixed_omega))
                scene["cylinders"] = {
                    "x": list(last_info.get("layout_x", fe.cylinders_x.tolist())),
                    "y": list(last_info.get("layout_y", fe.cylinders_y.tolist())),
                    "r": list(last_info.get("layout_r", fe.cylinders_r.tolist())),
                    "omegas": list(
                        last_info.get(
                            "layout_omega",
                            np.full(fe.max_n, om, dtype=np.float32).tolist(),
                        )
                    ),
                    "fixed_count": int(last_info.get("fixed_count", fe.fixed_count)),
                }
                scene["inflow"] = {
                    "u": float(last_info.get("inflow_u", fe.inflow_u)),
                    "v": float(last_info.get("inflow_v", fe.inflow_v)),
                }
                scene["global_omega"] = om
                logic_box = scene.get("logic_box", None)
                if isinstance(logic_box, dict):
                    logic_box["source_port"] = str(last_info.get("logic_source_port", logic_box.get("source_port", "")))
                    logic_box["target_port"] = str(last_info.get("logic_target_port", logic_box.get("target_port", "")))
                    logic_box["route_pairs"] = list(last_info.get("logic_route_pairs", logic_box.get("route_pairs", [])))
                    logic_box["seed_streamlines"] = list(last_info.get("logic_exits", []))
            if hide_logic_seeds:
                logic_box = scene.get("logic_box", None)
                if isinstance(logic_box, dict):
                    logic_box["seed_points"] = []
                    logic_box["seed_streamlines"] = []
            title = (
                f"t:{t + 1}/{max_steps} | R:{total_reward:.1f} | "
                f"f:{(float(last_info.get('global_omega', fe.fixed_omega)) / (2.0 * np.pi)):.2f} Hz"
            )
            frame = renderer.render(
                scene=scene,
                follower_path=path_history,
                target_pos=ctx.goal if show_target_point else None,
                target_path=target_path if show_target_path else None,
                title=title,
                draw_flow=True,
            )
            frames.append(frame)

        if done0:
            done_early = True
            break

    if len(frames) == 0:
        frames.append(
            renderer.render(
                scene=env.get_scene(),
                follower_path=path_history,
                target_pos=ctx.goal if show_target_point else None,
                target_path=target_path if show_target_path else None,
                title=f"f:{(float(fe.fixed_omega) / (2.0 * np.pi)):.2f} Hz",
                draw_flow=True,
            )
        )
    for _ in range(12):
        frames.append(frames[-1])

    Path(output_gif).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_gif, frames, fps=fps)
    trace_stem = Path(output_gif).with_suffix("")
    trace_csv = trace_stem.with_name(trace_stem.name + "_omega_trace.csv")
    trace_png = trace_stem.with_name(trace_stem.name + "_omega_trace.png")
    if len(omega_records) > 0:
        with trace_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(omega_records[0].keys()))
            writer.writeheader()
            writer.writerows(omega_records)
        _save_omega_trace_plot(omega_records, trace_png)
    renderer.close()
    vec_env.close()

    r = np.asarray(fe.cylinders_r, dtype=np.float32)
    killed_eps = float(getattr(LogicBoxConfig, "KILLED_EPS", 1e-6))
    inactive_count = int(np.sum(r <= killed_eps))
    print(f"[Saved] {output_gif}")
    if len(omega_records) > 0:
        print(f"[Saved] omega trace CSV: {trace_csv}")
        print(f"[Saved] omega trace PNG: {trace_png}")
    print(
        f"[Summary] dynamic_omega=True, frames={len(frames)}, path_points={len(path_history)}, "
        f"done_early={done_early}, reward={total_reward:.2f}, "
        f"target={last_info.get('logic_target_port', logic_target_port)}, "
        f"success={last_info.get('dynamic_success', None)}, "
        f"miss={last_info.get('logic_miss_ratio', None)}, "
        f"wrong={last_info.get('logic_wrong_side_ratio', None)}, "
        f"out={last_info.get('logic_outlet_error', None)}, "
        f"final_pos=({ctx.particle_pos[0]:.4f}, {ctx.particle_pos[1]:.4f}), "
        f"omega={float(last_info.get('global_omega', fe.fixed_omega)):.4f}, "
        f"inflow=({float(last_info.get('inflow_u', fe.inflow_u)):.4f},"
        f"{float(last_info.get('inflow_v', fe.inflow_v)):.4f}), "
        f"dead={inactive_count}/{int(r.size)}, killed_eps={killed_eps:.1e}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Long rollout visualization using best_model.zip")
    parser.add_argument(
        "--model",
        default=None,
        help="Model zip path. default: models/best/<task_tag>/best_model.zip",
    )
    parser.add_argument(
        "--vecnorm",
        default=None,
        help="VecNormalize pkl path. default: models/best/<task_tag>/vecnormalize_best.pkl",
    )
    parser.add_argument("--layout-mode", default=LayoutModeConfig.LAYOUT_MODE)
    parser.add_argument(
        "--logic-target-port",
        default=None,
        help="Force target port in single_multi_target mode, e.g. R0/R1/R2/T0/B0.",
    )
    parser.add_argument(
        "--logic-route-set-idx",
        type=int,
        default=None,
        help="Force route-set index in multi_map_switch mode for deterministic test/eval.",
    )
    parser.add_argument("--rollout-steps", type=int, default=400)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--show-target-point",
        action="store_true",
        help="Show red target point marker on frames.",
    )
    parser.add_argument(
        "--hide-target-path",
        action="store_true",
        help="Hide green dashed target trajectory.",
    )
    parser.add_argument(
        "--hide-logic-seeds",
        action="store_true",
        help="Hide logic-box seed points/seed streamlines overlays.",
    )
    parser.add_argument(
        "--show-port-only-overlays",
        action="store_true",
        help="When REWARD_MODE=port_only in logic_box mode, keep target path and seed overlays visible.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output gif path. default: models/best/<task_tag>/rollout/long_rollout_<task_tag>.gif",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    LayoutModeConfig.LAYOUT_MODE = str(args.layout_mode)
    dynamic_omega_mode = bool(is_dynamic_omega_mode(args.layout_mode))
    dynamic_omega_design_mode = bool(is_dynamic_omega_design_mode(args.layout_mode))
    dynamic_omega_radius_design_mode = bool(is_dynamic_omega_radius_design_mode(args.layout_mode))
    if dynamic_omega_mode and dynamic_omega_radius_design_mode:
        LogicBoxConfig.FIXED_LAYOUT_ENABLE = True
        LogicBoxConfig.FIXED_GEOMETRY_ENABLE = False
        LogicBoxConfig.KEEP_XY_ACTION_WHEN_FIXED = False
    elif dynamic_omega_mode and dynamic_omega_design_mode:
        LogicBoxConfig.FIXED_LAYOUT_ENABLE = False
        LogicBoxConfig.FIXED_GEOMETRY_ENABLE = False
        LogicBoxConfig.KEEP_XY_ACTION_WHEN_FIXED = True
    elif dynamic_omega_mode:
        LogicBoxConfig.FIXED_LAYOUT_ENABLE = True
        LogicBoxConfig.FIXED_GEOMETRY_ENABLE = True
        LogicBoxConfig.KEEP_XY_ACTION_WHEN_FIXED = False
    base_tag = build_task_tag(args.layout_mode)
    target_suffix = (
        f"_{str(args.logic_target_port).upper()}"
        if args.logic_target_port is not None and len(str(args.logic_target_port).strip()) > 0
        else ""
    )
    out = args.out or f"models/best/{base_tag}/rollout/long_rollout_{base_tag}{target_suffix}.gif"
    default_model, default_vecnorm = default_model_paths(args.layout_mode)

    model_path = args.model or default_model
    vecnorm_path = args.vecnorm or default_vecnorm

    # Backward-compatible fallback for legacy flat outputs.
    if args.model is None and not os.path.isfile(model_path):
        legacy_model = os.path.join("models", "best", "best_model.zip")
        if os.path.isfile(legacy_model):
            model_path = legacy_model
    if args.vecnorm is None and not os.path.isfile(vecnorm_path):
        legacy_vecnorm = os.path.join("models", "best", "vecnormalize_best.pkl")
        if os.path.isfile(legacy_vecnorm):
            vecnorm_path = legacy_vecnorm

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(
        f"[Config] mode={args.layout_mode}, rollout_steps={args.rollout_steps}, "
        f"frame_stride={args.frame_stride}, fps={args.fps}, model={model_path}, "
        f"target_port={args.logic_target_port}, route_set_idx={args.logic_route_set_idx}"
    )

    base_mode, _ = parse_layout_mode(args.layout_mode)
    reward_mode = str(getattr(LogicBoxConfig, "REWARD_MODE", "")).strip().lower()
    auto_hide_port_only = (
        base_mode == "logic_box_layout"
        and reward_mode in {"port_only", "port"}
        and (not args.show_port_only_overlays)
    )
    if auto_hide_port_only:
        print("[View] REWARD_MODE=port_only: auto-hide target path and logic seed overlays.")

    hide_target_path = bool(args.hide_target_path or auto_hide_port_only)
    hide_logic_seeds = bool(args.hide_logic_seeds or auto_hide_port_only)

    if dynamic_omega_mode:
        print("[Mode] dynamic omega rollout: model predicts delta-omega at every step.")
        run_dynamic_omega_rollout(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            layout_mode=args.layout_mode,
            rollout_steps=args.rollout_steps,
            frame_stride=args.frame_stride,
            fps=args.fps,
            output_gif=out,
            show_target_point=args.show_target_point,
            show_target_path=(not hide_target_path),
            hide_logic_seeds=hide_logic_seeds,
            logic_target_port=args.logic_target_port,
            logic_route_set_idx=args.logic_route_set_idx,
        )
        return

    physical_action = load_policy_action(
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        layout_mode=args.layout_mode,
        logic_target_port=args.logic_target_port,
        logic_route_set_idx=args.logic_route_set_idx,
    )
    print(f"[Action] decoded_dim={physical_action.shape[0]}")

    run_long_rollout(
        physical_action=physical_action,
        layout_mode=args.layout_mode,
        rollout_steps=args.rollout_steps,
        frame_stride=args.frame_stride,
        fps=args.fps,
        output_gif=out,
        show_target_point=args.show_target_point,
        show_target_path=(not hide_target_path),
        hide_logic_seeds=hide_logic_seeds,
        logic_target_port=args.logic_target_port,
        logic_route_set_idx=args.logic_route_set_idx,
    )


if __name__ == "__main__":
    main()
