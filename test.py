import numpy as np
import imageio.v2 as imageio
from envs.Wrapper import TaskContext, FollowerEnv, LeaderEnv
from envs.FluidEnv import FluidEnv
from utils.sample import random_sample
from envs.Renderer import FluidRenderer
from config import RankineSettingConfig, RenderSettingConfig

def make_fluid_env():
    return FluidEnv(
        start_pos=random_sample(),
        seed=None
    )

# 随机策略
def follower_policy(obs: np.ndarray, action_space) -> np.ndarray:
    return (action_space.sample() * 0.1).astype(np.float32)

def leader_policy(obs: np.ndarray, action_space) -> np.ndarray:
    return (action_space.sample() * 0.5).astype(np.float32)

def run_episode(T=300, save_path="episode.mp4", fps=15):
    ctx = TaskContext()
    fluid = make_fluid_env()
    follower = FollowerEnv(fluid_env=fluid, ctx=ctx)
    leader = LeaderEnv(ctx=ctx)

    f_obs, _ = follower.reset(seed=0)
    l_obs, _ = leader.reset(seed=0)

    renderer = FluidRenderer()
    frames = []
    path = []         # Follower轨迹
    leader_path = []   # Leader轨迹

    for t in range(T):
        # Leader 先动
        l_act = leader_policy(l_obs, leader.action_space)
        l_obs, _, l_term, l_trunc, _ = leader.step(l_act)

        # Follower 再跟
        f_act = follower_policy(f_obs, follower.action_space)
        f_obs, _, f_term, f_trunc, _ = follower.step(f_act)

        path.append((float(ctx.particle_pos[0]), float(ctx.particle_pos[1])))
        leader_path.append((float(ctx.goal[0]), float(ctx.goal[1])))

        dist = np.hypot(ctx.particle_pos[0]-ctx.goal[0], ctx.particle_pos[1]-ctx.goal[1])
        title = f"t={t} | dist={dist:.3f}"
        print(follower.get_scene())
        frame = renderer.render(
            follower.get_scene(),
            path=path,
            leader_pos=ctx.goal,
            leader_path=leader_path,
            title=title,
            show=False
        )
        frames.append(frame)

        if l_term or f_term or l_trunc or f_trunc:
            print(f"Episode stopped at t={t}.")
            break
    
    if len(frames) == 0:
        print("No frames captured, nothing to save.")
    else:
        try:
            imageio.mimsave(save_path, frames, fps=fps)
            print(f"Saved video to {save_path}")
        except Exception as e:
            fallback = "episode.gif"
            imageio.mimsave(fallback, frames, fps=fps)
            print(f"Saving mp4 failed ({e}). Saved GIF to {fallback}")

    renderer.close()

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    run_episode(T=300, save_path="episode.mp4", fps=20)