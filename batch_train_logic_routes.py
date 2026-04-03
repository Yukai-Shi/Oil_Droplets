import argparse
import os
import shutil
import subprocess
import sys
from typing import List

from config import LogicBoxConfig


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


def build_task_tag(layout_mode: str, path_type: str, source_port: str, target_port: str) -> str:
    base_mode, _ = parse_layout_mode(layout_mode)
    if base_mode == "logic_box_layout":
        return f"{layout_mode}_{path_type}_{source_port}_to_{target_port}__{_logic_bounds_suffix()}"
    return f"{layout_mode}_{path_type}"


def run_cmd(cmd: List[str], dry_run: bool = False):
    pretty = " ".join(cmd)
    print(f"[Run] {pretty}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch train logic-box routes and render rollout GIFs."
    )
    parser.add_argument("--source-port", default="L0")
    parser.add_argument("--targets", default="T0,R0,R1,R2,B0")
    parser.add_argument("--layout-mode", default="logic_box_layout_inflow")
    parser.add_argument("--path-type", default="logic_route")
    parser.add_argument("--eval-cycles", type=int, default=3)
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--n-eval-episodes", type=int, default=1)
    parser.add_argument("--patience-evals", type=int, default=50)
    parser.add_argument("--rollout-steps", type=int, default=600)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing models/eval folders for each tag before training.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source = str(args.source_port).upper()
    targets = [t.strip().upper() for t in str(args.targets).split(",") if t.strip()]
    workers = max(1, int(args.workers))
    eval_freq_timesteps = int(args.eval_freq)
    # SB3 EvalCallback eval_freq is measured in callback calls (n_calls),
    # while user-facing config here is in timesteps.
    eval_freq_calls = max(1, int(round(eval_freq_timesteps / workers)))
    total_timesteps = int(args.eval_cycles) * eval_freq_timesteps

    if len(targets) == 0:
        raise ValueError("No target ports provided.")

    print(
        "[BatchConfig] "
        f"source={source} targets={targets} layout={args.layout_mode} path={args.path_type} "
        f"eval_cycles={args.eval_cycles} eval_freq_timesteps={eval_freq_timesteps} "
        f"eval_freq_calls={eval_freq_calls} workers={workers} total_steps={total_timesteps}"
    )

    for idx, target in enumerate(targets):
        seed = int(args.seed_base) + idx
        tag = build_task_tag(
            layout_mode=str(args.layout_mode),
            path_type=str(args.path_type),
            source_port=source,
            target_port=target,
        )
        best_dir = os.path.join("models", "best", tag)
        eval_dir = os.path.join("models", "eval", tag)
        model_path = os.path.join(best_dir, "best_model.zip")
        vecnorm_path = os.path.join(best_dir, "vecnormalize_best.pkl")
        out_gif = os.path.join(best_dir, "rollout", f"long_rollout_{tag}.gif")

        print("\n" + "=" * 88)
        print(f"[Route] {source} -> {target} | tag={tag} | seed={seed}")

        if args.clean:
            print(f"[Clean] remove {best_dir}")
            print(f"[Clean] remove {eval_dir}")
            if not args.dry_run:
                shutil.rmtree(best_dir, ignore_errors=True)
                shutil.rmtree(eval_dir, ignore_errors=True)

        train_cmd = [
            str(args.python_exe),
            "train.py",
            "--layout-mode",
            str(args.layout_mode),
            "--path-type",
            str(args.path_type),
            "--source-port",
            source,
            "--target-port",
            target,
            "--total-timesteps",
            str(total_timesteps),
            "--eval-freq",
            str(eval_freq_calls),
            "--n-eval-episodes",
            str(args.n_eval_episodes),
            "--patience-evals",
            str(args.patience_evals),
            "--seed",
            str(seed),
            "--workers",
            str(workers),
        ]
        run_cmd(train_cmd, dry_run=bool(args.dry_run))

        if (not args.dry_run) and (not os.path.isfile(model_path)):
            print(
                f"[Warn] Missing model after training: {model_path}. "
                "Likely no eval checkpoint was triggered. Skip this route."
            )
            continue

        test_cmd = [
            str(args.python_exe),
            "test.py",
            "--layout-mode",
            str(args.layout_mode),
            "--model",
            model_path,
            "--vecnorm",
            vecnorm_path,
            "--rollout-steps",
            str(args.rollout_steps),
            "--frame-stride",
            str(args.frame_stride),
            "--fps",
            str(args.fps),
            "--out",
            out_gif,
        ]
        run_cmd(test_cmd, dry_run=bool(args.dry_run))

        print(f"[Done] {source} -> {target} | gif={out_gif}")

    print("\n[BatchDone] all routes finished.")


if __name__ == "__main__":
    main()
