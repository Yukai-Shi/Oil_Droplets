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


def _append_run_alias(tag: str, alias: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(alias))
    safe = safe.strip("._")
    if len(safe) == 0:
        return tag
    return f"{tag}__{safe}"


def _logic_bounds_suffix() -> str:
    mode = str(getattr(LogicBoxConfig, "BOUNDS_MODE", "local_box")).strip().lower()
    if mode in {"global", "global_field", "full_field", "render_field", "world"}:
        return "bnd_global"
    return "bnd_local"


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


def _parse_int_list(raw: str) -> List[int]:
    out = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if len(t) == 0:
            continue
        out.append(int(t))
    return out


def _parse_float_list(raw: str) -> List[float]:
    out = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if len(t) == 0:
            continue
        out.append(float(t))
    return out


def _rmin_tag(v: float) -> str:
    return f"{float(v):.4f}".replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep logic-box training over NUM_CYLINDERS and MIN_ACTIVE_R."
    )
    parser.add_argument("--layout-mode", default="logic_box_layout_inflow_u_fixed")
    parser.add_argument("--path-type", default="logic_route")
    parser.add_argument("--source-port", default="L1")
    parser.add_argument("--targets", default="T0,R0,B0")
    parser.add_argument("--num-cylinders-list", default="4,6,8,10")
    parser.add_argument("--min-active-r-list", default="0.0030,0.0035,0.0040")
    parser.add_argument(
        "--use-config-min-active-r",
        action="store_true",
        default=False,
        help="Do not sweep r_min; use LogicBoxConfig.MIN_ACTIVE_R from config.py.",
    )
    parser.add_argument("--logic-max-r", type=float, default=None)
    parser.add_argument(
        "--forbid-elimination",
        action="store_true",
        default=True,
        help="Use --forbid-elimination / --allow-elimination to toggle.",
    )
    parser.add_argument(
        "--allow-elimination",
        action="store_true",
        default=False,
        help="If set, override and allow elimination.",
    )
    parser.add_argument("--run-prefix", default="one2three_mean")
    parser.add_argument("--eval-cycles", type=int, default=3)
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--n-eval-episodes", type=int, default=1)
    parser.add_argument("--patience-evals", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source = str(args.source_port).strip().upper()
    targets = [t.strip().upper() for t in str(args.targets).split(",") if len(t.strip()) > 0]
    if len(targets) == 0:
        raise ValueError("No targets provided.")
    num_list = _parse_int_list(args.num_cylinders_list)
    if bool(args.use_config_min_active_r):
        rmin_list = [None]
    else:
        rmin_list = _parse_float_list(args.min_active_r_list)
    if len(num_list) == 0:
        raise ValueError("Empty sweep list: num-cylinders-list.")
    if (not bool(args.use_config_min_active_r)) and len(rmin_list) == 0:
        raise ValueError("Empty sweep list: min-active-r-list.")

    workers = max(1, int(args.workers))
    eval_freq_timesteps = int(args.eval_freq)
    eval_freq_calls = max(1, int(round(eval_freq_timesteps / workers)))
    total_timesteps = int(args.eval_cycles) * eval_freq_timesteps
    forbid = bool(args.forbid_elimination) and (not bool(args.allow_elimination))

    print(
        "[SweepConfig] "
        f"source={source} targets={targets} layout={args.layout_mode} path={args.path_type} "
        f"N={num_list} rmin={'config.MIN_ACTIVE_R' if bool(args.use_config_min_active_r) else rmin_list} "
        f"forbid_elim={forbid} "
        f"eval_cycles={args.eval_cycles} eval_freq_ts={eval_freq_timesteps} "
        f"eval_freq_calls={eval_freq_calls} workers={workers} total_steps={total_timesteps}"
    )

    case_id = 0
    for n in num_list:
        for rmin in rmin_list:
            case_id += 1
            seed = int(args.seed_base) + case_id - 1
            if rmin is None:
                alias = f"{args.run_prefix}_N{int(n)}_cfgRmin"
            else:
                alias = f"{args.run_prefix}_N{int(n)}_rmin{_rmin_tag(rmin)}"
            tag = build_task_tag(
                layout_mode=str(args.layout_mode),
                path_type=str(args.path_type),
                source_port=source,
                target_ports=targets,
                run_alias=alias,
            )
            best_dir = os.path.join("models", "best", tag)
            eval_dir = os.path.join("models", "eval", tag)

            print("\n" + "=" * 92)
            if rmin is None:
                print(
                    f"[Case {case_id}] alias={alias} | N={int(n)} "
                    f"rmin=config({float(getattr(LogicBoxConfig, 'MIN_ACTIVE_R', 0.0)):.4f}) "
                    f"seed={seed}"
                )
            else:
                print(
                    f"[Case {case_id}] alias={alias} | N={int(n)} rmin={float(rmin):.4f} "
                    f"seed={seed}"
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
                "--num-cylinders",
                str(int(n)),
                "--run-alias",
                alias,
                "--total-timesteps",
                str(total_timesteps),
                "--eval-freq",
                str(eval_freq_calls),
                "--n-eval-episodes",
                str(int(args.n_eval_episodes)),
                "--patience-evals",
                str(int(args.patience_evals)),
                "--seed",
                str(seed),
                "--workers",
                str(workers),
            ]
            if rmin is not None:
                train_cmd.extend(["--logic-min-active-r", f"{float(rmin):.6f}"])
            if args.logic_max_r is not None:
                train_cmd.extend(["--logic-max-r", f"{float(args.logic_max_r):.6f}"])
            if forbid:
                train_cmd.append("--logic-forbid-elimination")
            else:
                train_cmd.append("--logic-allow-elimination")

            run_cmd(train_cmd, dry_run=bool(args.dry_run))

    print("\n[SweepDone] all combinations finished.")


if __name__ == "__main__":
    main()