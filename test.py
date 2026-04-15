import argparse
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
    LayoutModeConfig,
    LogicBoxConfig,
    TrainingSettingConfig,
    get_logic_multi_route_pairs,
    get_logic_multi_route_sets,
)
from envs.FluidEnv import FluidEnv
from envs.Renderer import FluidRenderer
from envs.SharedXYPolicy import SharedXYActorCriticPolicy  # noqa: F401
from envs.Wrapper import FollowerEnv, TaskContext


# For MLP policies in SB3, CPU inference is typically more stable/faster than GPU.
DEVICE = "cpu"
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    DEVICE = "mps"


def parse_layout_mode(layout_mode: str):
    mode = str(layout_mode)
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
    logic_route_set_idx: Optional[int] = None,
):
    ctx = TaskContext()
    fe = FluidEnv(
        start_pos=np.array([-0.025, 0.06], dtype=np.float32),
        layout_mode=layout_mode,
    )
    env = FollowerEnv(fluid_env=fe, ctx=ctx, logic_seed_profile="eval")
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
            title = f"Long Rollout | t={t + 1}/{rollout_steps}"
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
            title="Long Rollout",
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
    base_tag = build_task_tag(args.layout_mode)
    out = args.out or f"models/best/{base_tag}/rollout/long_rollout_{base_tag}.gif"
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
        f"route_set_idx={args.logic_route_set_idx}"
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

    physical_action = load_policy_action(
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        layout_mode=args.layout_mode,
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
        logic_route_set_idx=args.logic_route_set_idx,
    )


if __name__ == "__main__":
    main()
