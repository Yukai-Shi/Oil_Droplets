import argparse
import os
import platform
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import LayoutModeConfig, TrainingSettingConfig
from envs.FluidEnv import FluidEnv
from envs.Renderer import FluidRenderer
from envs.Wrapper import FollowerEnv, TaskContext


# For MLP policies in SB3, CPU inference is typically more stable/faster than GPU.
DEVICE = "cpu"
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    DEVICE = "mps"


def load_policy_action(model_path: str, vecnorm_path: str, layout_mode: str):
    """
    Load trained policy and infer one one-shot action.
    """
    ctx = TaskContext()
    fe = FluidEnv(
        start_pos=np.array([-0.025, 0.06], dtype=np.float32),
        layout_mode=layout_mode,
    )
    env = FollowerEnv(fluid_env=fe, ctx=ctx)
    vec_env = DummyVecEnv([lambda: env])

    if vecnorm_path and os.path.isfile(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env, device=DEVICE)
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
):
    ctx = TaskContext()
    fe = FluidEnv(
        start_pos=np.array([-0.025, 0.06], dtype=np.float32),
        layout_mode=layout_mode,
    )
    env = FollowerEnv(fluid_env=fe, ctx=ctx)
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

    print(f"[Saved] {output_gif}")
    print(
        f"[Summary] frames={len(frames)}, path_points={len(path_history)}, "
        f"done_early={done_early}, final_pos=({ctx.particle_pos[0]:.4f}, {ctx.particle_pos[1]:.4f})"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Long rollout visualization using best_model.zip")
    parser.add_argument("--model", default="models/best/best_model.zip")
    parser.add_argument("--vecnorm", default="models/best/vecnormalize_best.pkl")
    parser.add_argument("--layout-mode", default=LayoutModeConfig.LAYOUT_MODE)
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
        "--out",
        default=None,
        help="Output gif path. default: models/best/long_rollout_<mode>.gif",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.out or f"models/best/long_rollout_{args.layout_mode}.gif"

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    print(
        f"[Config] mode={args.layout_mode}, rollout_steps={args.rollout_steps}, "
        f"frame_stride={args.frame_stride}, fps={args.fps}"
    )

    physical_action = load_policy_action(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        layout_mode=args.layout_mode,
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
        show_target_path=(not args.hide_target_path),
    )


if __name__ == "__main__":
    main()
