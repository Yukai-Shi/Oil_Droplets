# train_one_shot.py
import argparse
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

from envs.Wrapper import TaskContext, FollowerEnv, logic_target_port_set
from envs.FluidEnv import FluidEnv
from envs.Renderer import FluidRenderer
from utils.reseed import reseed_everything
from config import (
    LayoutModeConfig,
    LogicBoxConfig,
    TrainingSettingConfig,
    get_logic_box_ranges,
    get_logic_multi_route_pairs,
)

WORKERS = 1
LAYOUT_MODE = LayoutModeConfig.LAYOUT_MODE
PATH_TYPE = TrainingSettingConfig.PATH_TYPE


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


def build_task_tag(layout_mode: str) -> str:
    base_mode, _ = parse_layout_mode(layout_mode)
    if base_mode == "logic_box_layout":
        route_mode = str(getattr(LogicBoxConfig, "ROUTE_MODE", "single")).strip().lower()
        if route_mode in {"multi", "multi_map", "multi_route", "mapping"}:
            pairs = get_logic_multi_route_pairs()
            pair_tag = "_".join([f"{str(s).upper()}to{str(t).upper()}" for s, t in pairs])
            return _append_run_alias(f"{layout_mode}_{PATH_TYPE}_multi_{pair_tag}")
        if route_mode in {"single_multi_target", "single_source_multi_target", "one_to_many", "one_to_three"}:
            src = str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")).upper()
            tgt_set = getattr(LogicBoxConfig, "TARGET_PORT_SET", ["R0", "R1", "R2"])
            tgt_tag = "".join([str(t).upper() for t in tgt_set])
            return _append_run_alias(f"{layout_mode}_{PATH_TYPE}_{src}_to_{tgt_tag}")
        src = str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")).upper()
        tgt = str(getattr(LogicBoxConfig, "TARGET_PORT", "R1")).upper()
        return _append_run_alias(f"{layout_mode}_{PATH_TYPE}_{src}_to_{tgt}")
    return _append_run_alias(f"{layout_mode}_{PATH_TYPE}")


TASK_TAG = build_task_tag(LAYOUT_MODE)
TOTAL_TIMESTEPS = 1_000_00000
SAVE_DIR = "models"
BEST_ROOT_DIR = os.path.join(SAVE_DIR, "best")
TASK_BEST_DIR = os.path.join(BEST_ROOT_DIR, TASK_TAG)
EVAL_FREQ = 80000
N_EVAL_EPISODES = 1
PATIENCE_EVALS = 50
LOG_INTERVAL = 20
SEED = 0
TB_LOG_DIR = "tb_logs"
SAVE_EVAL_GIF = bool(getattr(TrainingSettingConfig, "SAVE_EVAL_GIF", False))
SAVE_EVAL_PREVIEW = bool(getattr(TrainingSettingConfig, "SAVE_EVAL_PREVIEW", False))
SAVE_BEST_PREVIEW = bool(getattr(TrainingSettingConfig, "SAVE_BEST_PREVIEW", True))
INFLOW_WARMUP_ENABLE = bool(getattr(TrainingSettingConfig, "INFLOW_WARMUP_ENABLE", True))
INFLOW_WARMUP_RATIO = float(getattr(TrainingSettingConfig, "INFLOW_WARMUP_RATIO", 0.20))
AUTO_STAGE_ENABLE = bool(getattr(TrainingSettingConfig, "AUTO_STAGE_ENABLE", False))
AUTO_STAGE_EVALS_PER_PHASE = int(getattr(TrainingSettingConfig, "AUTO_STAGE_EVALS_PER_PHASE", 3))
AUTO_STAGE_SNAPSHOT_TARGET = str(getattr(TrainingSettingConfig, "AUTO_STAGE_SNAPSHOT_TARGET", "")).strip().upper()

PPO_CFG = dict(
    policy="MlpPolicy",
    verbose=1,
    n_steps=128,
    batch_size=64,
    learning_rate=3e-4,
    ent_coef=0.001,
    gamma=0.99,
)


def parse_train_args():
    parser = argparse.ArgumentParser(description="Train one-shot layout optimization")
    parser.add_argument("--layout-mode", default=LAYOUT_MODE)
    parser.add_argument("--path-type", default=PATH_TYPE)
    parser.add_argument("--source-port", default=str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")))
    parser.add_argument("--target-port", default=str(getattr(LogicBoxConfig, "TARGET_PORT", "R1")))
    parser.add_argument("--logic-route-mode", default=str(getattr(LogicBoxConfig, "ROUTE_MODE", "single")))
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--n-eval-episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--patience-evals", type=int, default=PATIENCE_EVALS)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=LOG_INTERVAL,
        help="SB3 train log print interval (larger -> less frequent console output).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument(
        "--init-model",
        default="",
        help="Optional PPO checkpoint (.zip) to continue from.",
    )
    parser.add_argument(
        "--init-vecnorm",
        default="",
        help="Optional VecNormalize stats (.pkl) to continue from.",
    )
    parser.add_argument(
        "--reset-num-timesteps",
        action="store_true",
        help="When loading --init-model, reset SB3 timestep counter instead of continuing.",
    )
    parser.add_argument(
        "--bootstrap-fixed-layout-from-init",
        action="store_true",
        help=(
            "For logic_box mode: infer one-shot layout from --init-model and freeze it as "
            "FIXED_LAYOUT_X/Y before training."
        ),
    )
    parser.add_argument(
        "--auto-stage-enable",
        action="store_true",
        default=AUTO_STAGE_ENABLE,
        help=(
            "Enable auto stage scheduler (single run alternates free<->fixed layout every "
            "N eval checkpoints)."
        ),
    )
    parser.add_argument(
        "--auto-stage-evals-per-phase",
        type=int,
        default=AUTO_STAGE_EVALS_PER_PHASE,
        help="Auto stage scheduler: number of eval checkpoints per phase before switching.",
    )
    parser.add_argument(
        "--auto-stage-snapshot-target",
        default=AUTO_STAGE_SNAPSHOT_TARGET,
        help=(
            "Optional target port (e.g. R1) used when snapshotting centers before switching to fixed stage. "
            "Empty uses environment sampled target."
        ),
    )
    return parser.parse_args()


def bootstrap_logic_fixed_layout_from_model(model_path: str, vecnorm_path: str, layout_mode: str):
    """
    Infer one-shot cylinder centers from a model in free-layout mode,
    then freeze them into LogicBoxConfig.FIXED_LAYOUT_X/Y.
    """
    if len(str(model_path).strip()) == 0 or (not os.path.isfile(model_path)):
        raise FileNotFoundError(f"bootstrap model not found: {model_path}")

    prev_fixed = bool(getattr(LogicBoxConfig, "FIXED_LAYOUT_ENABLE", False))
    vec_env = None
    try:
        # Temporarily unfreeze to let the model output x/y layout.
        LogicBoxConfig.FIXED_LAYOUT_ENABLE = False

        ctx = TaskContext()
        fe = FluidEnv(
            start_pos=np.array([-0.025, 0.06], dtype=np.float32),
            layout_mode=layout_mode,
        )
        env = FollowerEnv(fluid_env=fe, ctx=ctx, logic_seed_profile="eval")
        vec_env = DummyVecEnv([lambda: env])
        if len(str(vecnorm_path).strip()) > 0 and os.path.isfile(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        model = PPO.load(model_path, env=vec_env, device=DEVICE)
        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, info = vec_env.step(action)
        info0 = info[0]
        x = np.asarray(info0.get("layout_x", []), dtype=np.float32).reshape(-1)
        y = np.asarray(info0.get("layout_y", []), dtype=np.float32).reshape(-1)
        if x.size == 0 or y.size == 0 or x.size != y.size:
            raise RuntimeError("failed to infer layout_x/layout_y from init model")
        LogicBoxConfig.FIXED_LAYOUT_X = x.copy()
        LogicBoxConfig.FIXED_LAYOUT_Y = y.copy()
        LogicBoxConfig.FIXED_LAYOUT_ENABLE = True
        print(f"[BootstrapLayout] frozen centers from init model | n={x.size}")
        return x, y
    finally:
        if vec_env is not None:
            try:
                vec_env.close()
            except Exception:
                pass
        # Keep fixed mode enabled only if bootstrap succeeded and set above.
        if not bool(getattr(LogicBoxConfig, "FIXED_LAYOUT_ENABLE", False)):
            LogicBoxConfig.FIXED_LAYOUT_ENABLE = prev_fixed


def _build_compact_eval_title(reward_value: float, info0: dict, base_mode: str) -> str:
    title = f"One-Shot Eval | R:{float(reward_value):.2f}"
    if base_mode == "logic_box_layout":
        lfit = info0.get("logic_path_fit_error", None)
        lcov = info0.get("logic_path_cover_penalty", None)
        lmis = info0.get("logic_miss_ratio", None)
        lws = info0.get("logic_wrong_side_ratio", None)
        lout = info0.get("logic_outlet_error", None)
        dead_cnt = info0.get("inactive_count", None)
        total_cnt = info0.get("total_count", None)
        killed_eps = info0.get("killed_eps", None)
        ltgt = info0.get("logic_episode_target_port", info0.get("logic_target_port", None))
        if ltgt is not None:
            title += f" | tgt:{str(ltgt).upper()}"
        if dead_cnt is not None and total_cnt is not None:
            title += f" | dead:{int(dead_cnt)}/{int(total_cnt)}"
        if killed_eps is not None:
            title += f" | eps:{float(killed_eps):.1e}"
        if lfit is not None:
            title += f" | lfit:{float(lfit):.3f}"
        if lcov is not None:
            title += f" | lcov:{float(lcov):.2f}"
        if lmis is not None:
            title += f" | lmis:{float(lmis):.2f}"
        if lws is not None:
            title += f" | lws:{float(lws):.2f}"
        if lout is not None:
            title += f" | lout:{float(lout):.2f}"
        return title

    if base_mode == "gate3_layout":
        gerr = info0.get("gate_lane_error", None)
        gmis = info0.get("gate_lane_miss_ratio", None)
        if gerr is not None:
            title += f" | gerr:{float(gerr):.3f}"
        if gmis is not None:
            title += f" | gmis:{float(gmis):.2f}"
        return title

    mean_dist = info0.get("mean_path_dist", None)
    final_dist = info0.get("final_dist", None)
    if mean_dist is not None:
        title += f" | mean:{float(mean_dist):.4f}"
    if final_dist is not None:
        title += f" | final:{float(final_dist):.4f}"
    return title


def render_evaluation_run(
    model_path,
    vecnorm_path,
    output_dir,
    filename_prefix,
    layout_mode=LAYOUT_MODE,
    forced_target_port=None,
):
    print("[Visualization] Generating evaluation image...")

    ctx = TaskContext()
    fe = FluidEnv(
        start_pos=np.array([-0.025, 0.06], dtype=np.float32),
        layout_mode=layout_mode,
    )
    follower_env = FollowerEnv(fluid_env=fe, ctx=ctx, logic_seed_profile="eval")
    if forced_target_port is not None and hasattr(follower_env, "set_logic_forced_target_port"):
        follower_env.set_logic_forced_target_port(str(forced_target_port).upper())
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
    flow_tangent_pen = info0.get("flow_tangent_penalty", None)
    flow_normal_pen = info0.get("flow_normal_penalty", None)
    inflow_reg_pen = info0.get("inflow_reg_penalty", None)
    inflow_u = info0.get("inflow_u", None)
    inflow_v = info0.get("inflow_v", None)
    global_omega = info0.get("global_omega", None)
    gate_lane_err = info0.get("gate_lane_error", None)
    gate_lane_miss = info0.get("gate_lane_miss_ratio", None)
    gate_lane_collision = info0.get("gate_lane_collision_ratio", None)
    gate_lane_order = info0.get("gate_lane_order_penalty", None)
    gate_lane_target = info0.get("gate_lane_target_y", None)
    gate_dir_pen = info0.get("gate_dir_penalty", None)
    gate_rev_ratio = info0.get("gate_reverse_ratio", None)
    logic_miss = info0.get("logic_miss_ratio", None)
    logic_collision = info0.get("logic_collision_ratio", None)
    logic_wrong_side = info0.get("logic_wrong_side_ratio", None)
    logic_outlet_err = info0.get("logic_outlet_error", None)
    logic_forward_pen = info0.get("logic_forward_penalty", None)
    logic_path_fit = info0.get("logic_path_fit_error", None)
    logic_path_cover = info0.get("logic_path_cover_penalty", None)
    logic_src = info0.get("logic_source_port", None)
    logic_tgt = info0.get("logic_target_port", None)
    logic_route_mode = str(info0.get("logic_route_mode", getattr(LogicBoxConfig, "ROUTE_MODE", "single")))
    logic_route_pairs = info0.get("logic_route_pairs", [])
    logic_exits = info0.get("logic_exits", [])
    inactive_count = info0.get("inactive_count", None)
    total_count = info0.get("total_count", None)
    zero_radius_ratio = info0.get("zero_radius_ratio", None)
    killed_eps = info0.get("killed_eps", None)

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
            "fixed_count": int(info0.get("fixed_count", 0)),
        },
        "inflow": {
            "u": float(info0.get("inflow_u", 0.0)),
            "v": float(info0.get("inflow_v", 0.0)),
        },
    }
    base_mode, _ = parse_layout_mode(layout_mode)
    if base_mode == "logic_box_layout":
        (bx0, bx1), (by0, by1) = get_logic_box_ranges()
        ports = {}
        for i, yy in enumerate(np.asarray(LogicBoxConfig.LEFT_PORT_Y, dtype=np.float32)):
            ports[f"L{i}"] = {"side": "left", "xy": [float(bx0), float(yy)]}
        for i, yy in enumerate(np.asarray(LogicBoxConfig.RIGHT_PORT_Y, dtype=np.float32)):
            ports[f"R{i}"] = {"side": "right", "xy": [float(bx1), float(yy)]}
        ports["T0"] = {"side": "top", "xy": [float(LogicBoxConfig.TOP_PORT_X), float(by1)]}
        ports["B0"] = {"side": "bottom", "xy": [float(LogicBoxConfig.BOTTOM_PORT_X), float(by0)]}
        seed_offsets = np.asarray(
            getattr(LogicBoxConfig, "EVAL_SOURCE_SEED_DY", getattr(LogicBoxConfig, "SOURCE_SEED_DY", [0.0])),
            dtype=np.float32,
        )
        if seed_offsets.size == 0:
            seed_offsets = np.array([0.0], dtype=np.float32)
        seed_points = []
        route_pairs = []
        if isinstance(logic_route_pairs, (list, tuple)) and len(logic_route_pairs) > 0:
            route_pairs = [
                [str(p[0]).upper(), str(p[1]).upper()]
                for p in logic_route_pairs
                if isinstance(p, (list, tuple)) and len(p) == 2
            ]
        if len(route_pairs) > 0:
            for src_name, _ in route_pairs:
                src_info = ports.get(str(src_name).upper(), None)
                if src_info is None:
                    continue
                seed_x = float(src_info["xy"][0])
                if str(src_info.get("side", "")) == "left":
                    seed_x = float(bx0) + max(
                        1e-4, float(getattr(LogicBoxConfig, "SOURCE_SEED_X_INSET", 1e-4))
                    )
                for dy in seed_offsets:
                    seed_points.append([float(seed_x), float(src_info["xy"][1] + float(dy))])
        else:
            src_name = str(logic_src) if logic_src is not None else str(LogicBoxConfig.SOURCE_PORT)
            src_name = src_name.upper()
            if src_name not in ports:
                src_name = "L1"
            src_info = ports[src_name]
            seed_x = float(src_info["xy"][0])
            if str(src_info.get("side", "")) == "left":
                seed_x = float(bx0) + max(
                    1e-4, float(getattr(LogicBoxConfig, "SOURCE_SEED_X_INSET", 1e-4))
                )
            seed_points = [[float(seed_x), float(src_info["xy"][1] + float(dy))] for dy in seed_offsets]
        scene_final["logic_box"] = {
            "x_range": [float(bx0), float(bx1)],
            "y_range": [float(by0), float(by1)],
            "source_port": str(logic_src) if logic_src is not None else str(LogicBoxConfig.SOURCE_PORT),
            "target_port": str(logic_tgt) if logic_tgt is not None else str(LogicBoxConfig.TARGET_PORT),
            "route_mode": str(logic_route_mode),
            "route_pairs": route_pairs,
            "show_box": bool(getattr(LogicBoxConfig, "SHOW_BOX", True)),
            "ports": ports,
            "seed_points": seed_points,
            "show_fixed_centers": bool(getattr(LogicBoxConfig, "FIXED_LAYOUT_ENABLE", False)),
            "fixed_centers": [
                [float(xx), float(yy)]
                for xx, yy in zip(info0["layout_x"], info0["layout_y"])
            ],
            "seed_streamlines": list(logic_exits) if isinstance(logic_exits, (list, tuple)) else [],
        }

    print(
        f"[Visualization] reward={reward[0]:.2f}, path_len={len(path_history)}, "
        f"mean_path_dist={mean_path_dist}, final_dist={final_dist}, "
        f"path_block_penalty={block_pen}, flow_tangent={flow_tangent_pen}, "
        f"flow_normal={flow_normal_pen}, gate_err={gate_lane_err}, "
        f"gate_miss={gate_lane_miss}, gate_col={gate_lane_collision}, "
        f"gate_ord={gate_lane_order}, gate_dir={gate_dir_pen}, gate_rev={gate_rev_ratio}, "
        f"gate_target={gate_lane_target}, "
        f"logic_miss={logic_miss}, logic_col={logic_collision}, logic_wrong={logic_wrong_side}, "
        f"logic_out={logic_outlet_err}, logic_fwd={logic_forward_pen}, "
        f"logic_fit={logic_path_fit}, logic_cov={logic_path_cover}, "
        f"inactive={inactive_count}/{total_count}, killed_eps={killed_eps}, zero_ratio={zero_radius_ratio}, "
        f"inflow_reg={inflow_reg_pen}, "
        f"inflow=({inflow_u},{inflow_v}), omega={global_omega}"
    )

    title = _build_compact_eval_title(
        reward_value=float(reward[0]),
        info0=info0,
        base_mode=base_mode,
    )
    final_frame = renderer.render(
        scene=scene_final,
        follower_path=path_history,
        target_pos=np.array(target_path[-1], dtype=np.float32),
        target_path=target_path,
        title=title,
        draw_flow=True,
    )

    task_tag = build_task_tag(layout_mode)
    out_base = os.path.normpath(output_dir)
    if os.path.basename(out_base) == task_tag:
        run_group_dir = output_dir
    else:
        run_group_dir = os.path.join(output_dir, task_tag)
    png_dir = os.path.join(run_group_dir, "frames")
    os.makedirs(png_dir, exist_ok=True)

    png_path = os.path.join(png_dir, f"{filename_prefix}final.png")

    imageio.imwrite(png_path, final_frame)
    print(f"[Visualization] Final frame saved to {png_path}")

    if SAVE_EVAL_GIF:
        gif_dir = os.path.join(run_group_dir, "gifs")
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{filename_prefix}trajectory.gif")
        imageio.mimsave(gif_path, [final_frame] * 8, duration=0.08, loop=0)
        print(f"[Visualization] GIF saved to {gif_path}")

    renderer.close()

class EvalCallbackWithEarlyStop(EvalCallback):
    def __init__(self, **kwargs):
        self.patience = kwargs.pop("patience", 10)
        self.enable_inflow_warmup = bool(kwargs.pop("enable_inflow_warmup", False))
        self.inflow_warmup_ratio = float(kwargs.pop("inflow_warmup_ratio", 0.0))
        self.total_timesteps_target = int(kwargs.pop("total_timesteps_target", 1))
        self.enable_auto_stage = bool(kwargs.pop("enable_auto_stage", False))
        self.auto_stage_evals_per_phase = max(1, int(kwargs.pop("auto_stage_evals_per_phase", 3)))
        self.auto_stage_snapshot_target = str(kwargs.pop("auto_stage_snapshot_target", "")).strip().upper()
        self._inflow_unfrozen = (not self.enable_inflow_warmup)
        super().__init__(**kwargs)
        self.no_improve_count = 0
        self._best_mean_reward = -np.inf
        self._auto_stage_eval_count = 0
        self._auto_stage_switch_count = 0

    @staticmethod
    def _set_inflow_control(vec_env, enabled: bool):
        try:
            vec_env.env_method("set_inflow_control_enabled", bool(enabled))
            return
        except Exception:
            pass
        try:
            vec_env.set_attr("inflow_control_enabled", bool(enabled))
        except Exception:
            pass

    def _maybe_update_inflow_control(self):
        if not self.enable_inflow_warmup:
            return
        warmup_steps = int(max(0, min(1.0, self.inflow_warmup_ratio)) * self.total_timesteps_target)
        should_unfreeze = self.num_timesteps >= warmup_steps
        if should_unfreeze == self._inflow_unfrozen:
            return
        self._inflow_unfrozen = should_unfreeze
        self._set_inflow_control(self.training_env, should_unfreeze)
        self._set_inflow_control(self.eval_env, should_unfreeze)
        state = "unfrozen" if should_unfreeze else "frozen"
        print(f"[InflowWarmup] inflow control {state} at step {self.num_timesteps}/{self.total_timesteps_target}")

    @staticmethod
    def _set_logic_fixed_layout(vec_env, enabled: bool, fixed_x=None, fixed_y=None):
        try:
            vec_env.env_method(
                "set_logic_fixed_layout",
                bool(enabled),
                fixed_x=fixed_x,
                fixed_y=fixed_y,
            )
        except Exception:
            try:
                vec_env.env_method("set_logic_fixed_layout", bool(enabled))
            except Exception:
                pass

    @staticmethod
    def _get_logic_fixed_layout(vec_env) -> bool:
        try:
            out = vec_env.env_method("get_logic_fixed_layout")
            if isinstance(out, (list, tuple)) and len(out) > 0:
                return bool(out[0])
        except Exception:
            pass
        return bool(getattr(LogicBoxConfig, "FIXED_LAYOUT_ENABLE", False))

    def _infer_layout_centers_from_current_policy(self):
        forced = str(self.auto_stage_snapshot_target).strip().upper()
        if len(forced) > 0:
            try:
                self.eval_env.env_method("set_logic_forced_target_port", forced)
            except Exception:
                pass
        try:
            obs = self.eval_env.reset()
            action, _ = self.model.predict(obs, deterministic=True)
            _, _, _, info = self.eval_env.step(action)
            info0 = info[0] if isinstance(info, (list, tuple)) else info
            x = np.asarray(info0.get("layout_x", []), dtype=np.float32).reshape(-1)
            y = np.asarray(info0.get("layout_y", []), dtype=np.float32).reshape(-1)
            if x.size == 0 or y.size == 0 or x.size != y.size:
                raise RuntimeError("failed to snapshot layout centers from current policy")
            return x, y
        finally:
            if len(forced) > 0:
                try:
                    self.eval_env.env_method("set_logic_forced_target_port", None)
                except Exception:
                    pass

    def _maybe_auto_switch_logic_stage(self) -> bool:
        if not self.enable_auto_stage:
            return False
        base_mode, _ = parse_layout_mode(LAYOUT_MODE)
        if base_mode != "logic_box_layout":
            return False

        self._auto_stage_eval_count += 1
        if self._auto_stage_eval_count < self.auto_stage_evals_per_phase:
            return False
        self._auto_stage_eval_count = 0

        cur_fixed = self._get_logic_fixed_layout(self.training_env)
        next_fixed = not bool(cur_fixed)
        if next_fixed:
            x, y = self._infer_layout_centers_from_current_policy()
            self._set_logic_fixed_layout(self.training_env, True, fixed_x=x, fixed_y=y)
            self._set_logic_fixed_layout(self.eval_env, True, fixed_x=x, fixed_y=y)
            LogicBoxConfig.FIXED_LAYOUT_X = x.copy()
            LogicBoxConfig.FIXED_LAYOUT_Y = y.copy()
            LogicBoxConfig.FIXED_LAYOUT_ENABLE = True
            self._auto_stage_switch_count += 1
            print(
                f"[AutoStage] switch#{self._auto_stage_switch_count}: FREE -> FIXED "
                f"(snapshot n={x.size}, target={self.auto_stage_snapshot_target or 'env'}) "
                f"at step {self.num_timesteps}"
            )
        else:
            self._set_logic_fixed_layout(self.training_env, False)
            self._set_logic_fixed_layout(self.eval_env, False)
            LogicBoxConfig.FIXED_LAYOUT_ENABLE = False
            self._auto_stage_switch_count += 1
            print(
                f"[AutoStage] switch#{self._auto_stage_switch_count}: FIXED -> FREE "
                f"at step {self.num_timesteps}"
            )
        return True

    def _render_single_multi_target_eval_triplet(self):
        targets = [str(t).upper() for t in logic_target_port_set()]
        if len(targets) == 0:
            return

        eval_model_stem = os.path.join(self.best_model_save_path, "_eval_current_model")
        eval_model_path = f"{eval_model_stem}.zip"
        eval_vn_path = os.path.join(self.best_model_save_path, "_eval_current_vecnormalize.pkl")

        # Snapshot current policy/normalizer so rendering matches current eval step,
        # not only the historical best checkpoint.
        self.model.save(eval_model_stem)
        self.training_env.save(eval_vn_path)

        for tgt in targets:
            render_evaluation_run(
                model_path=eval_model_path,
                vecnorm_path=eval_vn_path,
                output_dir=self.best_model_save_path,
                filename_prefix=f"eval_{TASK_TAG}_t{self.num_timesteps}_{tgt}_",
                layout_mode=LAYOUT_MODE,
                forced_target_port=str(tgt).upper(),
            )

    def _on_step(self) -> bool:
        self._maybe_update_inflow_control()
        continue_training = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            route_mode = str(getattr(LogicBoxConfig, "ROUTE_MODE", "single")).strip().lower()
            single_multi_target_mode = route_mode in {
                "single_multi_target",
                "single_source_multi_target",
                "one_to_many",
                "one_to_three",
            }
            if single_multi_target_mode and SAVE_EVAL_PREVIEW:
                try:
                    self._render_single_multi_target_eval_triplet()
                except Exception as exc:
                    print(f"[Visualization] eval triplet render skipped: {exc}")

            if self.best_mean_reward > self._best_mean_reward:
                self._best_mean_reward = self.best_mean_reward
                self.no_improve_count = 0

                vn_path = os.path.join(self.best_model_save_path, "vecnormalize_best.pkl")
                self.training_env.save(vn_path)
                if SAVE_BEST_PREVIEW:
                    if single_multi_target_mode:
                        targets = [str(t).upper() for t in logic_target_port_set()]
                        for tgt in targets:
                            try:
                                render_evaluation_run(
                                    model_path=os.path.join(self.best_model_save_path, "best_model.zip"),
                                    vecnorm_path=vn_path,
                                    output_dir=self.best_model_save_path,
                                    filename_prefix=f"best_{TASK_TAG}_t{self.num_timesteps}_{tgt}_",
                                    layout_mode=LAYOUT_MODE,
                                    forced_target_port=str(tgt).upper(),
                                )
                            except Exception as exc:
                                print(f"[Visualization] best render skipped for {tgt}: {exc}")
                    else:
                        try:
                            render_evaluation_run(
                                model_path=os.path.join(self.best_model_save_path, "best_model.zip"),
                                vecnorm_path=vn_path,
                                output_dir=self.best_model_save_path,
                                filename_prefix=f"best_{TASK_TAG}_t{self.num_timesteps}_",
                                layout_mode=LAYOUT_MODE,
                            )
                        except Exception as exc:
                            print(f"[Visualization] best render skipped: {exc}")
            else:
                self.no_improve_count += 1

            if self.enable_auto_stage:
                try:
                    switched = self._maybe_auto_switch_logic_stage()
                    if switched:
                        # Avoid premature early-stop right after a deliberate stage transition.
                        self.no_improve_count = 0
                except Exception as exc:
                    print(f"[AutoStage] stage switch skipped: {exc}")

            if self.no_improve_count >= self.patience:
                return False

        return continue_training


if __name__ == "__main__":
    args = parse_train_args()

    LAYOUT_MODE = str(args.layout_mode)
    PATH_TYPE = str(args.path_type)
    TOTAL_TIMESTEPS = int(args.total_timesteps)
    EVAL_FREQ = int(args.eval_freq)
    N_EVAL_EPISODES = int(args.n_eval_episodes)
    PATIENCE_EVALS = int(args.patience_evals)
    LOG_INTERVAL = int(args.log_interval)
    SEED = int(args.seed)
    WORKERS = int(args.workers)
    INIT_MODEL = str(args.init_model).strip()
    INIT_VECNORM = str(args.init_vecnorm).strip()
    RESET_NUM_TIMESTEPS = bool(args.reset_num_timesteps)
    BOOTSTRAP_FIXED_LAYOUT = bool(args.bootstrap_fixed_layout_from_init)
    AUTO_STAGE_ENABLE = bool(args.auto_stage_enable)
    AUTO_STAGE_EVALS_PER_PHASE = max(1, int(args.auto_stage_evals_per_phase))
    AUTO_STAGE_SNAPSHOT_TARGET = str(args.auto_stage_snapshot_target).strip().upper()

    LayoutModeConfig.LAYOUT_MODE = LAYOUT_MODE
    TrainingSettingConfig.PATH_TYPE = PATH_TYPE
    base_mode, _ = parse_layout_mode(LAYOUT_MODE)
    if base_mode == "logic_box_layout":
        LogicBoxConfig.ROUTE_MODE = str(args.logic_route_mode).strip().lower()
        LogicBoxConfig.SOURCE_PORT = str(args.source_port).upper()
        LogicBoxConfig.TARGET_PORT = str(args.target_port).upper()
        if AUTO_STAGE_ENABLE and (not bool(getattr(LogicBoxConfig, "KEEP_XY_ACTION_WHEN_FIXED", False))):
            LogicBoxConfig.KEEP_XY_ACTION_WHEN_FIXED = True
            print("[TrainConfig] auto-enable KEEP_XY_ACTION_WHEN_FIXED=True for auto stage switching.")
        if len(INIT_MODEL) > 0 and bool(getattr(LogicBoxConfig, "FIXED_LAYOUT_ENABLE", False)):
            # Auto enable stage-switch compatible action shape when loading a prior model.
            if not bool(getattr(LogicBoxConfig, "KEEP_XY_ACTION_WHEN_FIXED", False)):
                LogicBoxConfig.KEEP_XY_ACTION_WHEN_FIXED = True
                print("[TrainConfig] auto-enable KEEP_XY_ACTION_WHEN_FIXED=True for resumed fixed-layout training.")

    if len(INIT_MODEL) > 0 and (not os.path.isfile(INIT_MODEL)):
        raise FileNotFoundError(f"Init model not found: {INIT_MODEL}")
    if len(INIT_VECNORM) > 0 and (not os.path.isfile(INIT_VECNORM)):
        raise FileNotFoundError(f"Init vecnorm not found: {INIT_VECNORM}")

    TASK_TAG = build_task_tag(LAYOUT_MODE)
    TASK_BEST_DIR = os.path.join(BEST_ROOT_DIR, TASK_TAG)

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(BEST_ROOT_DIR, exist_ok=True)
    best_model_dir = TASK_BEST_DIR
    os.makedirs(best_model_dir, exist_ok=True)

    if BOOTSTRAP_FIXED_LAYOUT:
        if base_mode != "logic_box_layout":
            raise ValueError("--bootstrap-fixed-layout-from-init is only valid for logic_box layout.")
        if len(INIT_MODEL) == 0:
            raise ValueError("--bootstrap-fixed-layout-from-init requires --init-model.")
        bx, by = bootstrap_logic_fixed_layout_from_model(
            model_path=INIT_MODEL,
            vecnorm_path=INIT_VECNORM,
            layout_mode=LAYOUT_MODE,
        )
        boot_path = os.path.join(best_model_dir, "bootstrap_fixed_layout.npz")
        np.savez(boot_path, x=bx, y=by)
        print(f"[BootstrapLayout] saved centers to {boot_path}")

    print(
        "[TrainConfig] "
        f"mode={LAYOUT_MODE} path={PATH_TYPE} src={getattr(LogicBoxConfig, 'SOURCE_PORT', '')} "
        f"tgt={getattr(LogicBoxConfig, 'TARGET_PORT', '')} "
        f"route_mode={getattr(LogicBoxConfig, 'ROUTE_MODE', 'single')} "
        f"steps={TOTAL_TIMESTEPS} eval_freq={EVAL_FREQ} log_interval={LOG_INTERVAL} "
        f"workers={WORKERS} seed={SEED} "
        f"init_model={'yes' if len(INIT_MODEL)>0 else 'no'} "
        f"init_vecnorm={'yes' if len(INIT_VECNORM)>0 else 'no'} "
        f"reset_ts={RESET_NUM_TIMESTEPS} "
        f"auto_stage={AUTO_STAGE_ENABLE} auto_stage_evals={AUTO_STAGE_EVALS_PER_PHASE} "
        f"auto_stage_tgt={AUTO_STAGE_SNAPSHOT_TARGET or 'env'} "
        f"save_eval_preview={SAVE_EVAL_PREVIEW} save_best_preview={SAVE_BEST_PREVIEW} "
        f"tag={TASK_TAG}"
    )

    reseed_everything(SEED)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    def make_env_fn(seed=None, layout_mode=LAYOUT_MODE, logic_seed_profile="train"):
        def _init():
            ctx = TaskContext()
            fe = FluidEnv(
                start_pos=np.array([-0.025, 0.06], dtype=np.float32),
                layout_mode=layout_mode,
            )
            env = FollowerEnv(fluid_env=fe, ctx=ctx, logic_seed_profile=logic_seed_profile)
            return env
        return _init

    env_fns = [
        make_env_fn(seed=i, layout_mode=LAYOUT_MODE, logic_seed_profile="train")
        for i in range(WORKERS)
    ]
    train_env_raw = DummyVecEnv(env_fns)
    if len(INIT_VECNORM) > 0:
        vec_env = VecNormalize.load(INIT_VECNORM, train_env_raw)
        vec_env.training = True
        vec_env.norm_reward = False
        print(f"[Init] Loaded train VecNormalize: {INIT_VECNORM}")
    else:
        vec_env = VecNormalize(train_env_raw, norm_obs=True, norm_reward=False)

    eval_env_raw = DummyVecEnv(
        [make_env_fn(seed=999, layout_mode=LAYOUT_MODE, logic_seed_profile="eval")]
    )
    if len(INIT_VECNORM) > 0:
        eval_env = VecNormalize.load(INIT_VECNORM, eval_env_raw)
    else:
        eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False)
    eval_env.training = False
    eval_env.norm_reward = False

    _, use_inflow = parse_layout_mode(LAYOUT_MODE)
    use_inflow_warmup = bool(INFLOW_WARMUP_ENABLE and use_inflow)
    if use_inflow_warmup:
        try:
            vec_env.env_method("set_inflow_control_enabled", False)
            eval_env.env_method("set_inflow_control_enabled", False)
            print(
                f"[InflowWarmup] enabled | frozen_ratio={max(0.0, min(1.0, INFLOW_WARMUP_RATIO)):.2f} "
                f"(inflow fixed to target during warmup)"
            )
        except Exception as exc:
            print(f"[InflowWarmup] setup skipped: {exc}")
            use_inflow_warmup = False

    if len(INIT_MODEL) > 0:
        model = PPO.load(INIT_MODEL, env=vec_env, device=DEVICE)
        model.tensorboard_log = TB_LOG_DIR
        print(f"[Init] Loaded model: {INIT_MODEL}")
    else:
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
        log_path=os.path.join(SAVE_DIR, "eval", TASK_TAG),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        patience=PATIENCE_EVALS,
        enable_inflow_warmup=use_inflow_warmup,
        inflow_warmup_ratio=INFLOW_WARMUP_RATIO,
        total_timesteps_target=TOTAL_TIMESTEPS,
        enable_auto_stage=(AUTO_STAGE_ENABLE and base_mode == "logic_box_layout"),
        auto_stage_evals_per_phase=AUTO_STAGE_EVALS_PER_PHASE,
        auto_stage_snapshot_target=AUTO_STAGE_SNAPSHOT_TARGET,
        verbose=1,
    )

    print(f"Start training one-shot layout optimization | mode={LAYOUT_MODE}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_cb],
        log_interval=max(1, LOG_INTERVAL),
        tb_log_name=TASK_TAG,
        reset_num_timesteps=(True if len(INIT_MODEL) == 0 else bool(RESET_NUM_TIMESTEPS)),
    )
