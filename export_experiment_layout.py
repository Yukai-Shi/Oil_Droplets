import argparse
import csv
import json
import math
import os
import pickle
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import (
    LayoutModeConfig,
    LogicBoxConfig,
    RenderSettingConfig,
    StokesCylinderConfig,
    TrainingSettingConfig,
    get_logic_box_ranges,
    is_dynamic_omega_design_mode,
    is_dynamic_omega_mode,
)
from envs.FluidEnv import FluidEnv
from envs.SharedXYPolicy import SharedGeometryActorCriticPolicy, SharedXYActorCriticPolicy  # noqa: F401
from envs.Wrapper import FollowerEnv, TaskContext
from test import default_model_paths


DEVICE = "cpu"
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    DEVICE = "mps"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Export trained cylinder layout and controls for experiment comparison."
    )
    parser.add_argument("--layout-mode", default=LayoutModeConfig.LAYOUT_MODE)
    parser.add_argument("--logic-route-mode", default=str(getattr(LogicBoxConfig, "ROUTE_MODE", "multi_map")))
    parser.add_argument("--source-port", default=str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")))
    parser.add_argument("--target-port", default=str(getattr(LogicBoxConfig, "TARGET_PORT", "R1")))
    parser.add_argument(
        "--target-port-set",
        default="",
        help="Comma-separated target set for single_multi_target exports.",
    )
    parser.add_argument(
        "--logic-route-set-idx",
        type=int,
        default=None,
        help="Force route set index for multi_map_switch models.",
    )
    parser.add_argument("--num-cylinders", type=int, default=None)
    parser.add_argument("--run-alias", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--vecnorm", default=None)
    parser.add_argument(
        "--platform-width-cm",
        type=float,
        default=1.4,
        help="Experimental platform/channel width in cm.",
    )
    parser.add_argument(
        "--scale-ref",
        choices=["logic_box_y", "render_y"],
        default="logic_box_y",
        help="Simulation width used to map to platform-width-cm.",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix without extension. Default: beside model as experiment_layout.",
    )
    return parser.parse_args()


def _apply_overrides(args):
    LayoutModeConfig.LAYOUT_MODE = str(args.layout_mode)
    if is_dynamic_omega_mode(args.layout_mode) and is_dynamic_omega_design_mode(args.layout_mode):
        LogicBoxConfig.FIXED_LAYOUT_ENABLE = False
        LogicBoxConfig.FIXED_GEOMETRY_ENABLE = False
        LogicBoxConfig.KEEP_XY_ACTION_WHEN_FIXED = True
    elif is_dynamic_omega_mode(args.layout_mode):
        LogicBoxConfig.FIXED_LAYOUT_ENABLE = True
        LogicBoxConfig.FIXED_GEOMETRY_ENABLE = True
        LogicBoxConfig.KEEP_XY_ACTION_WHEN_FIXED = False
    LogicBoxConfig.ROUTE_MODE = str(args.logic_route_mode).strip().lower()
    LogicBoxConfig.SOURCE_PORT = str(args.source_port).upper()
    LogicBoxConfig.TARGET_PORT = str(args.target_port).upper()
    if args.target_port_set:
        tgts = [t.strip().upper() for t in str(args.target_port_set).split(",") if t.strip()]
        if tgts:
            LogicBoxConfig.TARGET_PORT_SET = tgts
    if args.num_cylinders is not None:
        StokesCylinderConfig.NUM_CYLINDERS = int(max(1, args.num_cylinders))
    if args.run_alias is not None:
        TrainingSettingConfig.RUN_ALIAS = str(args.run_alias).strip()


def _scale_reference_width(args):
    if args.scale_ref == "render_y":
        y0, y1 = RenderSettingConfig.Y_LIM
    else:
        _, (y0, y1) = get_logic_box_ranges()
    sim_width = float(y1) - float(y0)
    if sim_width <= 0.0:
        raise ValueError(f"Invalid simulation width for {args.scale_ref}: {sim_width}")
    return sim_width, float(args.platform_width_cm) / sim_width


def _infer_logic_box_n_from_vecnorm(vecnorm_path: Optional[str]) -> Optional[int]:
    if not vecnorm_path or not os.path.isfile(vecnorm_path):
        return None
    try:
        with open(vecnorm_path, "rb") as f:
            obj = pickle.load(f)
        obs_space = getattr(obj, "observation_space", None)
        shape = getattr(obs_space, "shape", None)
        if shape is None or len(shape) != 1:
            return None
        obs_dim = int(shape[0])
        # Static logic-box obs = particle xy (2) + x/y/r (3N) + target code (2).
        # Dynamic omega obs adds current omega, so obs_dim = 3N + 5.
        if obs_dim >= 4 and (obs_dim - 4) % 3 == 0:
            return int((obs_dim - 4) // 3)
        if obs_dim >= 5 and (obs_dim - 5) % 3 == 0:
            return int((obs_dim - 5) // 3)
    except Exception:
        return None
    return None


def _load_and_evaluate(args, model_path: str, vecnorm_path: Optional[str]):
    ctx = TaskContext()
    fe = FluidEnv(
        start_pos=np.array([-0.025, 0.06], dtype=np.float32),
        layout_mode=args.layout_mode,
    )
    env = FollowerEnv(fluid_env=fe, ctx=ctx, logic_seed_profile="eval")
    if args.logic_route_set_idx is not None and hasattr(env, "set_logic_forced_route_set_idx"):
        env.set_logic_forced_route_set_idx(int(args.logic_route_set_idx))

    vec_env = DummyVecEnv([lambda: env])
    if vecnorm_path and os.path.isfile(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env, device=DEVICE)
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    action = np.asarray(action).reshape(-1).astype(np.float32)

    _, reward, _, _, info = env.step(action)
    vec_env.close()
    return float(reward), info


def _default_outputs(model_path: str, out_prefix: Optional[str]):
    if out_prefix:
        prefix = Path(out_prefix)
    else:
        prefix = Path(model_path).resolve().parent / "experiment_layout"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    layout_csv = prefix.with_suffix(".csv")
    controls_csv = prefix.with_name(f"{prefix.name}_controls.csv")
    json_path = prefix.with_suffix(".json")
    return layout_csv, controls_csv, json_path


def main():
    args = _parse_args()
    # Apply mode/alias first so default paths are built from the requested task.
    _apply_overrides(args)

    default_model, default_vecnorm = default_model_paths(args.layout_mode)
    model_path = args.model or default_model
    vecnorm_path = args.vecnorm or default_vecnorm
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if vecnorm_path and not os.path.isfile(vecnorm_path):
        vecnorm_path = None

    inferred_n = _infer_logic_box_n_from_vecnorm(vecnorm_path)
    if args.num_cylinders is None and inferred_n is not None:
        StokesCylinderConfig.NUM_CYLINDERS = int(inferred_n)
        print(f"[Infer] NUM_CYLINDERS={StokesCylinderConfig.NUM_CYLINDERS} from VecNormalize obs shape.")

    reward, info = _load_and_evaluate(args, model_path, vecnorm_path)
    sim_width, scale_cm_per_sim = _scale_reference_width(args)
    (box_x0, box_x1), (box_y0, box_y1) = get_logic_box_ranges()

    xs = np.asarray(info["layout_x"], dtype=np.float64)
    ys = np.asarray(info["layout_y"], dtype=np.float64)
    rs = np.asarray(info["layout_r"], dtype=np.float64)
    omega = float(info.get("global_omega", 0.0))
    inflow_u = float(info.get("inflow_u", 0.0))
    inflow_v = float(info.get("inflow_v", 0.0))
    inflow_speed = float(math.hypot(inflow_u, inflow_v))

    omega_abs = abs(omega)
    rows = []
    cylinder_rows = []
    for i, (x, y, r) in enumerate(zip(xs, ys, rs)):
        diameter_cm = 2.0 * float(r) * scale_cm_per_sim
        diameter_mm = diameter_cm * 10.0
        x_cm_from_left = (float(x) - box_x0) * scale_cm_per_sim
        y_cm_from_bottom = (float(y) - box_y0) * scale_cm_per_sim
        surface_speed = omega_abs * float(r)
        u_over_surface = float(inflow_speed / surface_speed) if surface_speed > 1e-12 else None
        cylinder_rows.append(
            {
                "idx": int(i),
                "x_mm_from_box_left": float(x_cm_from_left * 10.0),
                "y_mm_from_box_bottom": float(y_cm_from_bottom * 10.0),
                "diameter_mm": float(diameter_mm),
            }
        )
        rows.append(
            {
                **cylinder_rows[-1],
                "x_sim": float(x),
                "y_sim": float(y),
                "r_sim": float(r),
                "diameter_sim": float(2.0 * r),
                "x_cm_centered": float(x * scale_cm_per_sim),
                "y_cm_centered": float(y * scale_cm_per_sim),
                "x_cm_from_box_left": float(x_cm_from_left),
                "y_cm_from_box_bottom": float(y_cm_from_bottom),
                "radius_cm": float(r * scale_cm_per_sim),
                "diameter_cm": float(diameter_cm),
                "u_over_abs_omega_r": u_over_surface,
            }
        )

    ratio_values = [row["u_over_abs_omega_r"] for row in rows if row["u_over_abs_omega_r"] is not None]
    mean_inflow_surface_ratio = float(np.nanmean(ratio_values)) if ratio_values else None
    controls = {
        "omega_rad_s": float(omega),
        "omega_hz": float(omega / (2.0 * math.pi)),
        "omega_rpm": float(omega * 60.0 / (2.0 * math.pi)),
        "omega_direction": int(np.sign(omega)),
        "inflow_u_sim": float(inflow_u),
        "inflow_v_sim": float(inflow_v),
        "inflow_speed_sim": float(inflow_speed),
        "inflow_speed_mm_s_if_sim_s": float(inflow_speed * scale_cm_per_sim * 10.0),
        "mean_inflow_to_surface_speed": mean_inflow_surface_ratio,
        "scale_mm_per_sim": float(scale_cm_per_sim * 10.0),
        "platform_width_mm": float(args.platform_width_cm * 10.0),
        "logic_box_width_x_mm": float((box_x1 - box_x0) * scale_cm_per_sim * 10.0),
        "logic_box_width_y_mm": float((box_y1 - box_y0) * scale_cm_per_sim * 10.0),
    }

    csv_path, controls_csv_path, json_path = _default_outputs(model_path, args.out_prefix)
    fieldnames = list(cylinder_rows[0].keys()) if cylinder_rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cylinder_rows)

    with open(controls_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(controls.keys()))
        writer.writeheader()
        writer.writerow(controls)

    meta = {
        "model_path": str(model_path),
        "vecnorm_path": str(vecnorm_path) if vecnorm_path else None,
        "layout_mode": str(args.layout_mode),
        "route_mode": str(LogicBoxConfig.ROUTE_MODE),
        "source_port": str(LogicBoxConfig.SOURCE_PORT),
        "target_port": str(LogicBoxConfig.TARGET_PORT),
        "target_port_set": list(getattr(LogicBoxConfig, "TARGET_PORT_SET", [])),
        "logic_route_set_idx": args.logic_route_set_idx,
        "reward": float(reward),
        "logic_route_pairs": info.get("logic_route_pairs", []),
        "logic_route_set_pairs": info.get("logic_route_set_pairs", []),
        "scale_ref": str(args.scale_ref),
        "platform_width_cm": float(args.platform_width_cm),
        "sim_width": float(sim_width),
        "scale_cm_per_sim": float(scale_cm_per_sim),
        "logic_box_sim": {
            "x0": float(box_x0),
            "x1": float(box_x1),
            "y0": float(box_y0),
            "y1": float(box_y1),
        },
        "logic_box_cm": {
            "width_x_cm": float((box_x1 - box_x0) * scale_cm_per_sim),
            "width_y_cm": float((box_y1 - box_y0) * scale_cm_per_sim),
        },
        "controls": controls,
        "global_omega_rad_s": float(omega),
        "global_omega_hz": controls["omega_hz"],
        "global_omega_rpm": controls["omega_rpm"],
        "inflow": {
            "u_sim": float(inflow_u),
            "v_sim": float(inflow_v),
            "speed_sim": float(inflow_speed),
            "speed_cm_s_if_sim_s": float(inflow_speed * scale_cm_per_sim),
            "mean_u_over_abs_omega_r": mean_inflow_surface_ratio,
        },
        "cylinders": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Export] cylinders_csv={csv_path}")
    print(f"[Export] controls_csv={controls_csv_path}")
    print(f"[Export] json={json_path}")
    print(
        "[Scale] "
        f"{args.scale_ref}: sim_width={sim_width:.6g} -> {args.platform_width_cm:.6g} cm, "
        f"scale={scale_cm_per_sim:.6g} cm/sim"
    )
    print(
        "[Control] "
        f"omega={omega:.6g} rad/s ({omega / (2.0 * math.pi):.6g} Hz, "
        f"{omega * 60.0 / (2.0 * math.pi):.6g} rpm), "
        f"inflow=({inflow_u:.6g},{inflow_v:.6g}) sim"
    )


if __name__ == "__main__":
    main()
