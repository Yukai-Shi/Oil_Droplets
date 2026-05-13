import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import (
    LayoutModeConfig,
    LogicBoxConfig,
    StokesCylinderConfig,
    TrainingSettingConfig,
)
from envs.FluidEnv import FluidEnv
from envs.SharedXYPolicy import SharedGeometryActorCriticPolicy, SharedXYActorCriticPolicy  # noqa: F401
from envs.Wrapper import FollowerEnv, TaskContext
from test import default_model_paths


DEVICE = "cpu"
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    DEVICE = "mps"


def _safe_alias(text: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(text))
    safe = safe.strip("._")
    return safe or "run"


def _split_targets(raw: str):
    targets = [t.strip().upper() for t in str(raw).split(",") if t.strip()]
    if len(targets) == 0:
        raise ValueError("--targets must contain at least one target port.")
    return targets


def _run(cmd, dry_run: bool = False):
    print("[Run] " + " ".join(str(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _logic_task_paths(layout_mode: str, source: str, targets, alias: str, n_cylinders=None):
    LogicBoxConfig.ROUTE_MODE = "single_multi_target"
    LogicBoxConfig.SOURCE_PORT = str(source).upper()
    LogicBoxConfig.TARGET_PORT = str(targets[0]).upper()
    LogicBoxConfig.TARGET_PORT_SET = [str(t).upper() for t in targets]
    TrainingSettingConfig.PATH_TYPE = "logic_route"
    TrainingSettingConfig.RUN_ALIAS = str(alias)
    if n_cylinders is not None:
        StokesCylinderConfig.NUM_CYLINDERS = int(n_cylinders)
    return default_model_paths(layout_mode)


def _extract_geometry(model_path: str, vecnorm_path: str, source: str, target: str, targets, n_cylinders=None):
    LayoutModeConfig.LAYOUT_MODE = "logic_box_layout_inflow_u_fixed"
    LogicBoxConfig.ROUTE_MODE = "single_multi_target"
    LogicBoxConfig.SOURCE_PORT = str(source).upper()
    LogicBoxConfig.TARGET_PORT = str(target).upper()
    LogicBoxConfig.TARGET_PORT_SET = [str(t).upper() for t in targets]
    LogicBoxConfig.FIXED_LAYOUT_ENABLE = False
    LogicBoxConfig.FIXED_GEOMETRY_ENABLE = False
    LogicBoxConfig.KEEP_XY_ACTION_WHEN_FIXED = True
    if n_cylinders is not None:
        StokesCylinderConfig.NUM_CYLINDERS = int(n_cylinders)

    ctx = TaskContext()
    fe = FluidEnv(layout_mode="logic_box_layout_inflow_u_fixed")
    env = FollowerEnv(fluid_env=fe, ctx=ctx, logic_seed_profile="eval")
    if hasattr(env, "set_logic_forced_target_port"):
        env.set_logic_forced_target_port(str(target).upper())
    vec_env = DummyVecEnv([lambda: env])
    if vecnorm_path and os.path.isfile(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env, device=DEVICE)
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, info = vec_env.step(action)
    info0 = info[0] if isinstance(info, (list, tuple)) else info
    vec_env.close()

    x = np.asarray(info0.get("layout_x", []), dtype=np.float32).reshape(-1)
    y = np.asarray(info0.get("layout_y", []), dtype=np.float32).reshape(-1)
    r = np.asarray(info0.get("layout_r", []), dtype=np.float32).reshape(-1)
    if x.size <= 0 or x.size != y.size or x.size != r.size:
        raise RuntimeError(
            f"Failed to extract geometry from stage-A model: x={x.size}, y={y.size}, r={r.size}"
        )
    return x, y, r, info0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage router training: first find x/y/r geometry, then lock it and train "
            "real-time dynamic omega control."
        )
    )
    parser.add_argument("--source-port", default=str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")))
    parser.add_argument("--targets", default="R0,R1,R2", help="Comma-separated target ports.")
    parser.add_argument("--num-cylinders", type=int, default=None)
    parser.add_argument("--stage-a-steps", type=int, default=600000)
    parser.add_argument("--stage-b-steps", type=int, default=800000)
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-prefix", default="find_geom_then_omega_rt")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--learning-rate-a", type=float, default=None)
    parser.add_argument("--learning-rate-b", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-stage-a", action="store_true")
    parser.add_argument(
        "--stage-a-model",
        default="",
        help="Use an existing stage-A best_model.zip when --skip-stage-a is set.",
    )
    parser.add_argument(
        "--stage-a-vecnorm",
        default="",
        help="Use an existing stage-A vecnormalize_best.pkl when --skip-stage-a is set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    targets = _split_targets(args.targets)
    source = str(args.source_port).upper()
    prefix = _safe_alias(args.run_prefix)
    stage_a_alias = f"{prefix}_A_find_geom"
    stage_b_alias = f"{prefix}_B_omega_rt"
    bundle_dir = Path("models") / "best" / f"{prefix}_find_geom_then_omega_rt_bundle"
    geom_path = bundle_dir / "fixed_geometry_from_stage_a.npz"
    summary_path = bundle_dir / "summary.json"
    target_arg = ",".join(targets)

    bundle_dir.mkdir(parents=True, exist_ok=True)

    stage_a_model, stage_a_vecnorm = _logic_task_paths(
        layout_mode="logic_box_layout_inflow_u_fixed",
        source=source,
        targets=targets,
        alias=stage_a_alias,
        n_cylinders=args.num_cylinders,
    )

    if args.skip_stage_a:
        if len(str(args.stage_a_model).strip()) > 0:
            stage_a_model = str(args.stage_a_model)
        if len(str(args.stage_a_vecnorm).strip()) > 0:
            stage_a_vecnorm = str(args.stage_a_vecnorm)
    else:
        cmd_a = [
            args.python_exe,
            "train.py",
            "--layout-mode",
            "logic_box_layout_inflow_u_fixed",
            "--path-type",
            "logic_route",
            "--logic-route-mode",
            "single_multi_target",
            "--source-port",
            source,
            "--target-port",
            targets[0],
            "--target-port-set",
            target_arg,
            "--total-timesteps",
            str(int(args.stage_a_steps)),
            "--eval-freq",
            str(int(args.eval_freq)),
            "--workers",
            str(int(args.workers)),
            "--seed",
            str(int(args.seed)),
            "--run-alias",
            stage_a_alias,
            "--shared-geometry-one-stage-enable",
        ]
        if args.num_cylinders is not None:
            cmd_a += ["--num-cylinders", str(int(args.num_cylinders))]
        if args.learning_rate_a is not None:
            cmd_a += ["--learning-rate", str(float(args.learning_rate_a))]
        _run(cmd_a, dry_run=bool(args.dry_run))

    if args.dry_run:
        stage_b_model, stage_b_vecnorm = _logic_task_paths(
            layout_mode="logic_box_layout_inflow_u_fixed_omega_rt",
            source=source,
            targets=targets,
            alias=stage_b_alias,
            n_cylinders=args.num_cylinders,
        )
        print(f"[DryRun] stage_a_model={stage_a_model}")
        print(f"[DryRun] geometry_npz={geom_path}")
        print(f"[DryRun] stage_b_model={stage_b_model}")
        return

    if not os.path.isfile(stage_a_model):
        raise FileNotFoundError(f"Stage-A best model not found: {stage_a_model}")
    if not os.path.isfile(stage_a_vecnorm):
        raise FileNotFoundError(f"Stage-A VecNormalize not found: {stage_a_vecnorm}")

    x, y, r, info0 = _extract_geometry(
        model_path=stage_a_model,
        vecnorm_path=stage_a_vecnorm,
        source=source,
        target=targets[0],
        targets=targets,
        n_cylinders=args.num_cylinders,
    )
    np.savez(
        geom_path,
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        r=r.astype(np.float32),
        source_port=source,
        target_ports=np.asarray(targets),
        stage_a_model=str(stage_a_model),
        stage_a_vecnorm=str(stage_a_vecnorm),
    )
    print(f"[Geometry] saved {geom_path} | N={x.size}")

    stage_b_model, stage_b_vecnorm = _logic_task_paths(
        layout_mode="logic_box_layout_inflow_u_fixed_omega_rt",
        source=source,
        targets=targets,
        alias=stage_b_alias,
        n_cylinders=int(x.size),
    )
    cmd_b = [
        args.python_exe,
        "train.py",
        "--layout-mode",
        "logic_box_layout_inflow_u_fixed_omega_rt",
        "--path-type",
        "logic_route",
        "--logic-route-mode",
        "single_multi_target",
        "--source-port",
        source,
        "--target-port",
        targets[0],
        "--target-port-set",
        target_arg,
        "--fixed-geometry-npz",
        str(geom_path),
        "--total-timesteps",
        str(int(args.stage_b_steps)),
        "--eval-freq",
        str(int(args.eval_freq)),
        "--workers",
        str(int(args.workers)),
        "--seed",
        str(int(args.seed)),
        "--run-alias",
        stage_b_alias,
    ]
    if args.learning_rate_b is not None:
        cmd_b += ["--learning-rate", str(float(args.learning_rate_b))]
    _run(cmd_b, dry_run=False)

    final_dir = bundle_dir / "final_dynamic_omega"
    final_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for src_path, name in [(stage_b_model, "best_model.zip"), (stage_b_vecnorm, "vecnormalize_best.pkl")]:
        if os.path.isfile(src_path):
            dst = final_dir / name
            shutil.copy2(src_path, dst)
            copied[name] = str(dst)

    summary = {
        "source_port": source,
        "target_ports": targets,
        "stage_a_alias": stage_a_alias,
        "stage_b_alias": stage_b_alias,
        "stage_a_model": str(stage_a_model),
        "stage_a_vecnorm": str(stage_a_vecnorm),
        "stage_b_model": str(stage_b_model),
        "stage_b_vecnorm": str(stage_b_vecnorm),
        "geometry_npz": str(geom_path),
        "bundle_dir": str(bundle_dir),
        "final_copies": copied,
        "extracted_geometry_reward": float(info0.get("episode_return", 0.0)),
        "num_cylinders": int(x.size),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Summary] saved {summary_path}")
    print(f"[Final] model={stage_b_model}")
    print(f"[Final] vecnorm={stage_b_vecnorm}")


if __name__ == "__main__":
    main()
