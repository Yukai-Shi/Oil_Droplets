import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from config import LogicBoxConfig, get_logic_multi_route_sets


def parse_layout_mode(layout_mode: str):
    mode = str(layout_mode)
    if mode.endswith("_inflow_u_fixed"):
        return mode[:-15], True
    if mode.endswith("_inflow_u"):
        return mode[:-9], True
    if mode.endswith("_inflow"):
        return mode[:-7], True
    return mode, False


def _logic_bounds_suffix() -> str:
    mode = str(getattr(LogicBoxConfig, "BOUNDS_MODE", "local_box")).strip().lower()
    if mode in {"global", "global_field", "full_field", "render_field", "world"}:
        return "bnd_global"
    return "bnd_local"


def _append_run_alias(tag: str, alias: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(alias))
    safe = safe.strip("._")
    if len(safe) == 0:
        return tag
    return f"{tag}__{safe}"


def build_task_tag(layout_mode: str, path_type: str, route_mode: str, run_alias: str) -> str:
    base_mode, _ = parse_layout_mode(layout_mode)
    if base_mode == "logic_box_layout":
        bounds_suffix = _logic_bounds_suffix()
        rm = str(route_mode).strip().lower()
        if rm in {"multi_map_switch", "multi_switch", "mapping_switch"}:
            route_sets = get_logic_multi_route_sets()
            first_pairs = route_sets[0] if len(route_sets) > 0 else [("L0", "T0"), ("L1", "R1"), ("L2", "B0")]
            first_tag = "_".join([f"{str(s).upper()}to{str(t).upper()}" for s, t in first_pairs])
            base = f"{layout_mode}_{path_type}_multisw{len(route_sets)}_{first_tag}__{bounds_suffix}"
            return _append_run_alias(base, run_alias)
        if rm in {"multi", "multi_map", "multi_route", "mapping"}:
            pairs = getattr(LogicBoxConfig, "MULTI_ROUTE_PAIRS", [("L0", "T0"), ("L1", "R1"), ("L2", "B0")])
            pair_tag = "_".join([f"{str(s).upper()}to{str(t).upper()}" for s, t in pairs])
            base = f"{layout_mode}_{path_type}_multi_{pair_tag}__{bounds_suffix}"
            return _append_run_alias(base, run_alias)
        src = str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")).upper()
        tgt = str(getattr(LogicBoxConfig, "TARGET_PORT", "R1")).upper()
        base = f"{layout_mode}_{path_type}_{src}_to_{tgt}__{bounds_suffix}"
        return _append_run_alias(base, run_alias)
    return _append_run_alias(f"{layout_mode}_{path_type}", run_alias)


def parse_csv_ints(raw: str, expected_len: int) -> List[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if len(x.strip()) > 0]
    if len(vals) == 1:
        vals = vals * expected_len
    if len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} ints, got {len(vals)} from: {raw}")
    return vals


def parse_csv_floats(raw: str, expected_len: int) -> List[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if len(x.strip()) > 0]
    if len(vals) == 1:
        vals = vals * expected_len
    if len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} floats, got {len(vals)} from: {raw}")
    return vals


def run_cmd(cmd: List[str], dry_run: bool = False):
    print("[Run] " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def ensure_file(path: Path, title: str):
    if not path.is_file():
        raise FileNotFoundError(f"{title} not found: {path}")


def copy_tree_if_exists(src: Path, dst: Path):
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Alternating A/B training for logic multi_map_switch:\n"
            "A stage: free x/y/r/omega search; B stage: bootstrap + fixed-layout refine."
        )
    )
    ap.add_argument("--layout-mode", default="logic_box_layout_inflow_u_fixed")
    ap.add_argument("--path-type", default="logic_route")
    ap.add_argument("--route-mode", default="multi_map_switch")
    ap.add_argument(
        "--stage-cycles",
        default="15,15,4,8",
        help="Eval-cycles for [A0,B0,A1,B1]. One value broadcasts.",
    )
    ap.add_argument(
        "--stage-lrs",
        default="1e-4,5e-5,5e-5,3e-5",
        help="Learning rates for [A0,B0,A1,B1]. One value broadcasts.",
    )
    ap.add_argument("--eval-freq", type=int, default=20000, help="Evaluation frequency in env timesteps.")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-eval-episodes", type=int, default=1)
    ap.add_argument("--patience-evals", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--run-prefix", default="multiswitch_ab")
    ap.add_argument(
        "--output-root",
        default="models/ab_runs",
        help="All staged outputs are copied into a single folder under this root.",
    )
    ap.add_argument("--init-model", default="", help="Optional init best_model.zip for A0.")
    ap.add_argument("--init-vecnorm", default="", help="Optional init vecnormalize for A0.")
    ap.add_argument("--num-cylinders", type=int, default=None)
    ap.add_argument("--logic-min-active-r", type=float, default=None)
    ap.add_argument("--logic-max-r", type=float, default=None)
    ap.add_argument(
        "--logic-forbid-elimination",
        dest="logic_forbid_elimination",
        action="store_true",
    )
    ap.add_argument(
        "--logic-allow-elimination",
        dest="logic_forbid_elimination",
        action="store_false",
    )
    ap.set_defaults(logic_forbid_elimination=None)
    ap.add_argument("--run-tests", action="store_true", help="After final stage, run test.py for each route-set.")
    ap.add_argument("--test-rollout-steps", type=int, default=600)
    ap.add_argument("--test-fps", type=int, default=20)
    ap.add_argument("--test-frame-stride", type=int, default=1)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    stage_names = ["A0", "B0", "A1", "B1"]
    stage_cycles = parse_csv_ints(args.stage_cycles, expected_len=4)
    stage_lrs = parse_csv_floats(args.stage_lrs, expected_len=4)
    workers = max(1, int(args.workers))
    eval_freq_ts = int(args.eval_freq)
    eval_freq_calls = max(1, int(round(eval_freq_ts / workers)))
    forbid = args.logic_forbid_elimination

    route_sets = get_logic_multi_route_sets()
    route_sets_n = len(route_sets)
    if str(args.route_mode).strip().lower() not in {"multi_map_switch", "multi_switch", "mapping_switch"}:
        print(f"[Warn] route_mode={args.route_mode} is not multi_map_switch family.")

    bundle_name = f"{args.run_prefix}__{str(args.layout_mode)}__{str(args.path_type)}"
    bundle_dir = Path(args.output_root) / bundle_name
    if args.clean and bundle_dir.exists() and (not args.dry_run):
        shutil.rmtree(bundle_dir, ignore_errors=True)
    if not args.dry_run:
        bundle_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[ABConfig] "
        f"layout={args.layout_mode} path={args.path_type} route_mode={args.route_mode} "
        f"route_sets={route_sets_n} cycles={stage_cycles} lrs={stage_lrs} "
        f"eval_freq_ts={eval_freq_ts} eval_freq_calls={eval_freq_calls} workers={workers} "
        f"bundle={bundle_dir}"
    )

    prev_model = str(args.init_model).strip()
    prev_vecnorm = str(args.init_vecnorm).strip()
    if len(prev_model) > 0 and (not args.dry_run):
        ensure_file(Path(prev_model), "init model")
        if len(prev_vecnorm) > 0:
            ensure_file(Path(prev_vecnorm), "init vecnorm")

    records = []

    for idx, stage in enumerate(stage_names):
        cycles = int(stage_cycles[idx])
        lr = float(stage_lrs[idx])
        if cycles <= 0:
            print(f"[Skip] {stage} cycles<=0")
            continue

        is_a_stage = stage.startswith("A")
        stage_alias = f"{args.run_prefix}_{stage}_{'free' if is_a_stage else 'fixed'}"
        stage_seed = int(args.seed_base) + idx
        stage_steps = int(cycles) * eval_freq_ts
        task_tag = build_task_tag(
            layout_mode=str(args.layout_mode),
            path_type=str(args.path_type),
            route_mode=str(args.route_mode),
            run_alias=stage_alias,
        )
        best_dir = Path("models") / "best" / task_tag
        eval_dir = Path("models") / "eval" / task_tag

        print(
            "\n" + "=" * 96 + "\n"
            f"[Stage {stage}] type={'FREE' if is_a_stage else 'FIXED'} "
            f"cycles={cycles} steps={stage_steps} lr={lr:.6g} seed={stage_seed}\n"
            f"[OutTag] {task_tag}"
        )

        if args.clean and (not args.dry_run):
            shutil.rmtree(best_dir, ignore_errors=True)
            shutil.rmtree(eval_dir, ignore_errors=True)

        cmd = [
            str(args.python_exe),
            "train.py",
            "--layout-mode", str(args.layout_mode),
            "--path-type", str(args.path_type),
            "--logic-route-mode", str(args.route_mode),
            "--total-timesteps", str(stage_steps),
            "--eval-freq", str(eval_freq_calls),
            "--n-eval-episodes", str(int(args.n_eval_episodes)),
            "--patience-evals", str(int(args.patience_evals)),
            "--seed", str(stage_seed),
            "--workers", str(workers),
            "--run-alias", stage_alias,
            "--learning-rate", f"{lr:.8f}",
        ]

        if is_a_stage:
            cmd.append("--shared-xy-one-stage-enable")

        if len(prev_model) > 0:
            cmd.extend(["--init-model", prev_model])
            if len(prev_vecnorm) > 0:
                cmd.extend(["--init-vecnorm", prev_vecnorm])

        if (not is_a_stage) and len(prev_model) > 0:
            cmd.append("--bootstrap-fixed-layout-from-init")

        if args.num_cylinders is not None:
            cmd.extend(["--num-cylinders", str(int(args.num_cylinders))])
        if args.logic_min_active_r is not None:
            cmd.extend(["--logic-min-active-r", f"{float(args.logic_min_active_r):.6f}"])
        if args.logic_max_r is not None:
            cmd.extend(["--logic-max-r", f"{float(args.logic_max_r):.6f}"])
        if forbid is True:
            cmd.append("--logic-forbid-elimination")
        elif forbid is False:
            cmd.append("--logic-allow-elimination")

        run_cmd(cmd, dry_run=bool(args.dry_run))

        stage_model = best_dir / "best_model.zip"
        stage_vecnorm = best_dir / "vecnormalize_best.pkl"
        if not args.dry_run:
            ensure_file(stage_model, f"{stage} best_model")
            ensure_file(stage_vecnorm, f"{stage} vecnormalize")

            # Collect everything into one bundle directory.
            stage_bundle = bundle_dir / f"{idx+1:02d}_{stage}"
            stage_bundle.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage_model, stage_bundle / "best_model.zip")
            shutil.copy2(stage_vecnorm, stage_bundle / "vecnormalize_best.pkl")
            copy_tree_if_exists(best_dir / "frames", stage_bundle / "frames")
            copy_tree_if_exists(best_dir / "gifs", stage_bundle / "gifs")
            copy_tree_if_exists(best_dir / "_sb3_eval_mean", stage_bundle / "_sb3_eval_mean")
            copy_tree_if_exists(eval_dir, stage_bundle / "eval_logs")

        prev_model = str(stage_model)
        prev_vecnorm = str(stage_vecnorm)
        records.append(
            {
                "stage": stage,
                "type": "free" if is_a_stage else "fixed",
                "cycles": cycles,
                "timesteps": stage_steps,
                "learning_rate": lr,
                "seed": stage_seed,
                "run_alias": stage_alias,
                "task_tag": task_tag,
                "best_dir": str(best_dir),
                "eval_dir": str(eval_dir),
                "model": str(stage_model),
                "vecnorm": str(stage_vecnorm),
            }
        )

    if len(records) == 0:
        raise RuntimeError("No stage executed. Check --stage-cycles.")

    final_model = Path(prev_model)
    final_vecnorm = Path(prev_vecnorm)

    if not args.dry_run:
        shutil.copy2(final_model, bundle_dir / "final_model.zip")
        shutil.copy2(final_vecnorm, bundle_dir / "final_vecnormalize.pkl")

    if args.run_tests:
        route_sets = get_logic_multi_route_sets()
        rollout_dir = bundle_dir / "rollout"
        if not args.dry_run:
            rollout_dir.mkdir(parents=True, exist_ok=True)
        for route_set_idx in range(len(route_sets)):
            out_gif = rollout_dir / f"final_route_set_{route_set_idx}.gif"
            test_cmd = [
                str(args.python_exe),
                "test.py",
                "--layout-mode", str(args.layout_mode),
                "--model", str(final_model),
                "--vecnorm", str(final_vecnorm),
                "--logic-route-set-idx", str(route_set_idx),
                "--rollout-steps", str(int(args.test_rollout_steps)),
                "--frame-stride", str(int(args.test_frame_stride)),
                "--fps", str(int(args.test_fps)),
                "--out", str(out_gif),
            ]
            run_cmd(test_cmd, dry_run=bool(args.dry_run))

    summary = {
        "layout_mode": str(args.layout_mode),
        "path_type": str(args.path_type),
        "route_mode": str(args.route_mode),
        "route_sets_count": int(route_sets_n),
        "eval_freq_timesteps": int(eval_freq_ts),
        "eval_freq_calls": int(eval_freq_calls),
        "workers": int(workers),
        "records": records,
        "final_model": str(final_model),
        "final_vecnorm": str(final_vecnorm),
        "bundle_dir": str(bundle_dir),
    }
    if not args.dry_run:
        with open(bundle_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Done] A/B alternating training finished.")
    print(f"[Bundle] {bundle_dir}")
    print(f"[FinalModel] {final_model}")
    print(f"[FinalVecNorm] {final_vecnorm}")


if __name__ == "__main__":
    main()
