# train_one_shot.py
import os
import platform
import torch
import numpy as np
import imageio.v2 as imageio

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    DEVICE = "mps"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from envs.Wrapper import TaskContext, FollowerEnv
from envs.FluidEnv import FluidEnv
from envs.Renderer import FluidRenderer
from utils.reseed import reseed_everything
from config import LayoutModeConfig

WORKERS = 4
LAYOUT_MODE = LayoutModeConfig.LAYOUT_MODE
TOTAL_TIMESTEPS = 1_000_00000
SAVE_DIR = "models"
EVAL_FREQ = 5000
N_EVAL_EPISODES = 1
PATIENCE_EVALS = 50
SEED = 0
TB_LOG_DIR = "tb_logs"

PPO_CFG = dict(
    policy="MlpPolicy",
    verbose=1,
    n_steps=128,
    batch_size=64,
    learning_rate=3e-4,
    ent_coef=0.01,
    gamma=0.99,
)

def render_evaluation_run(
    model_path,
    vecnorm_path,
    output_dir,
    filename_prefix,
    layout_mode=LAYOUT_MODE,
):
    print("[Visualization] Generating GIF visualization...")

    ctx = TaskContext()
    fe = FluidEnv(
        start_pos=np.array([-0.025, 0.06], dtype=np.float32),
        layout_mode=layout_mode,
    )
    follower_env = FollowerEnv(fluid_env=fe, ctx=ctx)
    vec_env = DummyVecEnv([lambda: follower_env])

    if os.path.isfile(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env, device=DEVICE)
    renderer = FluidRenderer()

    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    info0 = info[0]
    mean_path_dist = info0.get("mean_path_dist", None)
    final_dist = info0.get("final_dist", None)
    block_pen = info0.get("path_block_penalty", None)
    flow_dir_pen = info0.get("flow_dir_penalty", None)
    flow_speed_pen = info0.get("flow_speed_penalty", None)
    flow_reverse_ratio = info0.get("flow_reverse_ratio", None)

    path_history = info0.get("history_pos", [])
    target_path = info0.get("target_path", ctx.path.tolist())

    if len(path_history) == 0:
        path_history = [(-0.025, 0.06)]

    scene_final = {
        "particle": {
            "x": float(info0["final_pos"][0]),
            "y": float(info0["final_pos"][1]),
        },
        "target": {
            "x": float(target_path[-1][0]),
            "y": float(target_path[-1][1]),
        },
        "cylinders": {
            "x": info0["layout_x"],
            "y": info0["layout_y"],
            "r": info0["layout_r"],
            "omegas": info0["layout_omega"],
        }
    }

    print(
        f"[Visualization] reward={reward[0]:.2f}, path_len={len(path_history)}, "
        f"mean_path_dist={mean_path_dist}, final_dist={final_dist}, "
        f"path_block_penalty={block_pen}, flow_dir={flow_dir_pen}, "
        f"flow_speed={flow_speed_pen}, flow_reverse={flow_reverse_ratio}"
    )

    frames = []

    stride = max(1, len(path_history) // 80)

    for k in range(1, len(path_history) + 1, stride):
        current_path = path_history[:k]
        current_pos = current_path[-1]

        scene_k = {
            "particle": {
                "x": float(current_pos[0]),
                "y": float(current_pos[1]),
            },
            "target": scene_final["target"],
            "cylinders": scene_final["cylinders"],
        }

        target_idx = min(k - 1, len(target_path) - 1)
        current_target = np.array(target_path[target_idx], dtype=np.float32)

        title = f"One-Shot Eval | Reward: {reward[0]:.2f}"
        if mean_path_dist is not None and final_dist is not None:
            title += f" | mean: {float(mean_path_dist):.4f} | final: {float(final_dist):.4f}"
        if block_pen is not None:
            title += f" | block: {float(block_pen):.4f}"
        if flow_dir_pen is not None and flow_speed_pen is not None:
            title += f" | fdir: {float(flow_dir_pen):.3f}"
            title += f" | fspd: {float(flow_speed_pen):.3f}"
        if flow_reverse_ratio is not None:
            title += f" | frev: {float(flow_reverse_ratio):.2f}"

        frame = renderer.render(
            scene=scene_k,
            follower_path=current_path,
            target_pos=current_target,
            target_path=target_path,
            title=title,
            draw_flow=True
        )
        frames.append(frame)

    if len(frames) == 0:
        title = f"One-Shot Eval | Reward: {reward[0]:.2f}"
        if mean_path_dist is not None and final_dist is not None:
            title += f" | mean: {float(mean_path_dist):.4f} | final: {float(final_dist):.4f}"
        if block_pen is not None:
            title += f" | block: {float(block_pen):.4f}"
        if flow_dir_pen is not None and flow_speed_pen is not None:
            title += f" | fdir: {float(flow_dir_pen):.3f}"
            title += f" | fspd: {float(flow_speed_pen):.3f}"
        if flow_reverse_ratio is not None:
            title += f" | frev: {float(flow_reverse_ratio):.2f}"

        frame = renderer.render(
            scene=scene_final,
            follower_path=path_history,
            target_pos=np.array(target_path[-1], dtype=np.float32),
            target_path=target_path,
            title=title,
            draw_flow=True
        )
        frames.append(frame)

    for _ in range(10):
        frames.append(frames[-1])

    gif_path = os.path.join(output_dir, f"{filename_prefix}trajectory.gif")
    png_path = os.path.join(output_dir, f"{filename_prefix}final.png")

    imageio.mimsave(gif_path, frames, duration=0.08, loop=0)
    imageio.imwrite(png_path, frames[-1])

    print(f"[Visualization] GIF saved to {gif_path}")
    print(f"[Visualization] Final frame saved to {png_path}")

    renderer.close()

class EvalCallbackWithEarlyStop(EvalCallback):
    def __init__(self, **kwargs):
        self.patience = kwargs.pop("patience", 10)
        super().__init__(**kwargs)
        self.no_improve_count = 0
        self._best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.best_mean_reward > self._best_mean_reward:
                self._best_mean_reward = self.best_mean_reward
                self.no_improve_count = 0

                vn_path = os.path.join(self.best_model_save_path, "vecnormalize_best.pkl")
                self.training_env.save(vn_path)

                render_evaluation_run(
                    model_path=os.path.join(self.best_model_save_path, "best_model.zip"),
                    vecnorm_path=vn_path,
                    output_dir=self.best_model_save_path,
                    filename_prefix=f"best_{LAYOUT_MODE}_t{self.num_timesteps}_",
                    layout_mode=LAYOUT_MODE,
                )
            else:
                self.no_improve_count += 1

            if self.no_improve_count >= self.patience:
                return False

        return continue_training


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_model_dir = os.path.join(SAVE_DIR, "best")
    os.makedirs(best_model_dir, exist_ok=True)

    reseed_everything(SEED)

    def make_env_fn(seed=None, layout_mode=LAYOUT_MODE):
        def _init():
            ctx = TaskContext()
            fe = FluidEnv(
                start_pos=np.array([-0.025, 0.06], dtype=np.float32),
                layout_mode=layout_mode,
            )
            env = FollowerEnv(fluid_env=fe, ctx=ctx)
            return env
        return _init

    env_fns = [make_env_fn(seed=i, layout_mode=LAYOUT_MODE) for i in range(WORKERS)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    eval_env = DummyVecEnv([make_env_fn(seed=999, layout_mode=LAYOUT_MODE)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO(
        policy=PPO_CFG["policy"],
        env=vec_env,
        **{k: v for k, v in PPO_CFG.items() if k != "policy"},
        tensorboard_log=TB_LOG_DIR,
        device=DEVICE
    )

    eval_cb = EvalCallbackWithEarlyStop(
        eval_env=eval_env,
        best_model_save_path=best_model_dir,
        log_path=os.path.join(SAVE_DIR, "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        patience=PATIENCE_EVALS,
        verbose=1,
    )

    print(f"Start training one-shot layout optimization | mode={LAYOUT_MODE}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb])
