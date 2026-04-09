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


def _append_run_alias(tag: str, alias: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(alias))
    safe = safe.strip("._")
    if len(safe) == 0:
        return tag
    return f"{tag}__{safe}"


def build_task_tag(
    layout_mode: str,
    path_type: str,
    source_port: str,
    target_ports: List[str],
    run_alias: str,
) -> str:
    base_mode, _ = parse_layout_mode(layout_mode)
    if base_mode == "logic_box_layout":
        tgt_tag = "".join([str(t).upper() for t in target_ports])
        base = f"{layout_mode}_{path_type}_{source_port}_to_{tgt_tag}__{_logic_bounds_suffix()}"
        return _append_run_alias(base, run_alias)
    return _append_run_alias(f"{layout_mode}_{path_type}", run_alias)


def run_cmd(cmd: List[str], dry_run: bool = False):
    print("[Run] " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _parse_targets(raw: str) -> List[str]:
    return [t.strip().upper() for t in str(raw).split(",") if len(t.strip()) > 0]


def _parse_stage_targets(raw: str) -> List[List[str]]:
    stages = []
    for chunk in str(raw).split(";"):
        c = chunk.strip()
        if len(c) == 0:
            continue
        tgts = _parse_targets(c)
        if len(tgts) == 0:
            continue
        stages.append(tgts)
    return stages


def _parse_int_list(raw: str) -> List[int]:
    out = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if len(t) == 0:
            continue
        out.append(int(t))
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Three-stage curriculum runner for one-to-three logic task. "
            "Each stage auto-loads previous stage best_model/vecnormalize."
        )
    )
    parser.add_argument("--layout-mode", default="logic_box_layout_inflow_u_fixed")
    parser.add_argument("--path-type", default="logic_route")
    parser.add_argument("--source-port", default="L1")
    parser.add_argument(
        "--stage-targets",
        default="T0;T0,R0;T0,R0,B0",
        help='Semicolon-separated stage target sets, e.g. "T0;T0,R0;T0,R0,B0".',
    )
    parser.add_argument(
        "--eval-cycles",
        default="3,3,3",
        help='Per-stage eval cycles. One value broadcasts to all stages, e.g. "3" or "3,3,4".',
    )
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--n-eval-episodes", type=int, default=1)
    parser.add_argument("--patience-evals", type=int, default=50)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-prefix", default="one2three_curriculum")
    parser.add_argument("--num-cylinders", type=int, default=None)
    parser.add_argument(
        "--active-cyl-stages",
        default="",
        help=(
            "Optional per-stage ACTIVE_CYL_LIMIT list, e.g. '3,5,7'. "
            "One value broadcasts to all stages. Empty disables stage-wise active gating."
        ),
    )
    parser.add_argument(
        "--logic-inactive-r",
        type=float,
        default=None,
        help="Optional INACTIVE_LOCK_R used with --active-cyl-stages (e.g. 0.0 or 0.002).",
    )
    parser.add_argument("--logic-min-active-r", type=float, default=None)
    parser.add_argument("--logic-max-r", type=float, default=None)
    parser.add_argument(
        "--logic-forbid-elimination",
        dest="logic_forbid_elimination",
        action="store_true",
        help="Force FORBID_ELIMINATION=True for this run.",
    )
    parser.add_argument(
        "--logic-allow-elimination",
        dest="logic_forbid_elimination",
        action="store_false",
        help="Force FORBID_ELIMINATION=False for this run.",
    )
    parser.set_defaults(logic_forbid_elimination=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    source = str(args.source_port).strip().upper()
    stages = _parse_stage_targets(args.stage_targets)
    if len(stages) == 0:
        raise ValueError("No valid stage targets parsed from --stage-targets.")
    if len(stages) != 3:
        print(f"[Warn] expected 3 stages, got {len(stages)}. Continue anyway.")

    cycles = _parse_int_list(args.eval_cycles)
    if len(cycles) == 0:
        raise ValueError("No valid --eval-cycles provided.")
    if len(cycles) == 1:
        cycles = cycles * len(stages)
    if len(cycles) != len(stages):
        raise ValueError(
            f"--eval-cycles count mismatch: got {len(cycles)} values for {len(stages)} stages."
        )

    workers = max(1, int(args.workers))
    eval_freq_timesteps = int(args.eval_freq)
    eval_freq_calls = max(1, int(round(eval_freq_timesteps / workers)))
    forbid = args.logic_forbid_elimination

    active_k_stages = []
    active_raw = str(args.active_cyl_stages).strip()
    if len(active_raw) > 0:
        active_k_stages = _parse_int_list(active_raw)
        if len(active_k_stages) == 0:
            raise ValueError("No valid --active-cyl-stages parsed.")
        if len(active_k_stages) == 1:
            active_k_stages = active_k_stages * len(stages)
        if len(active_k_stages) != len(stages):
            raise ValueError(
                f"--active-cyl-stages count mismatch: got {len(active_k_stages)} values for {len(stages)} stages."
            )

    print(
        "[CurriculumConfig] "
        f"source={source} stages={stages} eval_cycles={cycles} "
        f"eval_freq_ts={eval_freq_timesteps} eval_freq_calls={eval_freq_calls} "
        f"workers={workers} run_prefix={args.run_prefix} "
        f"active_k_stages={active_k_stages if len(active_k_stages) > 0 else 'disabled'} "
        f"inactive_r={args.logic_inactive_r}"
    )

    prev_model = ""
    prev_vecnorm = ""
    stage_records = []

    for idx, targets in enumerate(stages):
        stage_id = idx + 1
        stage_seed = int(args.seed_base) + idx
        stage_cycles = int(cycles[idx])
        total_timesteps = int(stage_cycles) * eval_freq_timesteps
        active_k = None
        if len(active_k_stages) > 0:
            active_k = int(active_k_stages[idx])
        tgt_tag = "".join(targets)
        stage_alias = f"{args.run_prefix}_s{stage_id}_{tgt_tag}"

        tag = build_task_tag(
            layout_mode=str(args.layout_mode),
            path_type=str(args.path_type),
            source_port=source,
            target_ports=targets,
            run_alias=stage_alias,
        )
        best_dir = os.path.join("models", "best", tag)
        eval_dir = os.path.join("models", "eval", tag)
        model_path = os.path.join(best_dir, "best_model.zip")
        vecnorm_path = os.path.join(best_dir, "vecnormalize_best.pkl")
        stage_records.append(
            {
                "stage": stage_id,
                "targets": list(targets),
                "tag": tag,
                "best_dir": best_dir,
                "model": model_path,
                "vecnorm": vecnorm_path,
            }
        )

        print("\n" + "=" * 96)
        print(
            f"[Stage {stage_id}] targets={targets} seed={stage_seed} "
            f"cycles={stage_cycles} steps={total_timesteps} "
            f"active_k={active_k if active_k is not None else 'all'}"
        )
        print(f"[OutDir] {best_dir}")

        if args.clean:
            print(f"[Clean] {best_dir}")
            print(f"[Clean] {eval_dir}")
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
            "--logic-route-mode",
            "single_multi_target",
            "--source-port",
            source,
            "--target-port",
            targets[0],
            "--target-port-set",
            ",".join(targets),
            "--total-timesteps",
            str(total_timesteps),
            "--eval-freq",
            str(eval_freq_calls),
            "--n-eval-episodes",
            str(int(args.n_eval_episodes)),
            "--patience-evals",
            str(int(args.patience_evals)),
            "--seed",
            str(stage_seed),
            "--workers",
            str(workers),
            "--run-alias",
            stage_alias,
        ]

        if len(prev_model) > 0:
            train_cmd.extend(["--init-model", prev_model, "--init-vecnorm", prev_vecnorm])
            print(f"[Resume] init_model={prev_model}")
            print(f"[Resume] init_vecnorm={prev_vecnorm}")

        if args.num_cylinders is not None:
            train_cmd.extend(["--num-cylinders", str(int(args.num_cylinders))])
        if active_k is not None:
            train_cmd.extend(["--logic-active-cyl-limit", str(int(active_k))])
            if args.logic_inactive_r is not None:
                train_cmd.extend(["--logic-inactive-r", f"{float(args.logic_inactive_r):.6f}"])
        if args.logic_min_active_r is not None:
            train_cmd.extend(["--logic-min-active-r", f"{float(args.logic_min_active_r):.6f}"])
        if args.logic_max_r is not None:
            train_cmd.extend(["--logic-max-r", f"{float(args.logic_max_r):.6f}"])
        if forbid is True:
            train_cmd.append("--logic-forbid-elimination")
        elif forbid is False:
            train_cmd.append("--logic-allow-elimination")

        run_cmd(train_cmd, dry_run=bool(args.dry_run))

        if (not args.dry_run) and (not os.path.isfile(model_path)):
            raise FileNotFoundError(f"[Stage {stage_id}] Missing model: {model_path}")
        if (not args.dry_run) and (not os.path.isfile(vecnorm_path)):
            raise FileNotFoundError(f"[Stage {stage_id}] Missing vecnormalize: {vecnorm_path}")

        prev_model = model_path
        prev_vecnorm = vecnorm_path
        print(f"[Stage {stage_id}] done | model={model_path}")

    print("\n[CurriculumDone] all stages finished.")
    if len(stage_records) > 0:
        last = stage_records[-1]
        print(f"[Final] tag={last['tag']}")
        print(f"[Final] model={last['model']}")
        print(f"[Final] vecnorm={last['vecnorm']}")


if __name__ == "__main__":
    main()
