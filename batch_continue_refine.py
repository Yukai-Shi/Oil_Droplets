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


def build_task_tag(layout_mode: str, path_type: str, source_port: str, targets: List[str], run_alias: str) -> str:
    base_mode, _ = parse_layout_mode(layout_mode)
    if base_mode == "logic_box_layout":
        tgt_tag = "".join([str(t).upper() for t in targets])
        base = f"{layout_mode}_{path_type}_{source_port}_to_{tgt_tag}__{_logic_bounds_suffix()}"
        return _append_run_alias(base, run_alias)
    return _append_run_alias(f"{layout_mode}_{path_type}", run_alias)


def parse_targets(raw: str) -> List[str]:
    tgts = [t.strip().upper() for t in str(raw).split(",") if len(t.strip()) > 0]
    if len(tgts) == 0:
        raise ValueError("No valid targets parsed.")
    return tgts


def run_cmd(cmd: List[str], dry_run: bool = False):
    print("[Run] " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def must_exist(path: str, title: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{title} not found: {path}")


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Continue training from an existing model with a robust two-phase schedule:\n"
            "Phase1 fixed-layout refine (r/omega), then optional Phase2 further refine.\n"
            "Default skips Phase2 so you can validate Phase1 performance first."
        )
    )
    ap.add_argument("--layout-mode", default="logic_box_layout_inflow_u_fixed")
    ap.add_argument("--path-type", default="logic_route")
    ap.add_argument("--source-port", default="L1")
    ap.add_argument("--targets", default="T0,R0,R2")
    ap.add_argument("--init-model", required=True, help="Input best_model.zip")
    ap.add_argument("--init-vecnorm", default="", help="Optional input vecnormalize_best.pkl")
    ap.add_argument("--phase1-cycles", type=int, default=3, help="Fixed-layout refine eval cycles.")
    ap.add_argument("--phase2-cycles", type=int, default=0, help="Phase2 refine eval cycles. Set 0 to skip Phase2.")
    ap.add_argument("--phase1-lr", type=float, default=1e-4, help="Phase1 learning rate.")
    ap.add_argument("--phase2-lr", type=float, default=5e-5, help="Phase2 learning rate.")
    ap.add_argument(
        "--phase2-free-layout",
        action="store_true",
        help=(
            "Unfreeze layout in Phase2 (adds --shared-xy-one-stage-enable). "
            "Warning: if resumed policy is target-conditioned MlpPolicy, x/y may become target-dependent."
        ),
    )
    ap.add_argument("--eval-freq", type=int, default=20000, help="Eval freq in environment timesteps.")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--n-eval-episodes", type=int, default=1)
    ap.add_argument("--patience-evals", type=int, default=50)
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--run-prefix", default="continue_refine")
    ap.add_argument("--num-cylinders", type=int, default=None)
    ap.add_argument("--logic-min-active-r", type=float, default=None)
    ap.add_argument("--logic-max-r", type=float, default=None)
    ap.add_argument(
        "--logic-forbid-elimination",
        dest="logic_forbid_elimination",
        action="store_true",
        help="Force FORBID_ELIMINATION=True for both phases.",
    )
    ap.add_argument(
        "--logic-allow-elimination",
        dest="logic_forbid_elimination",
        action="store_false",
        help="Force FORBID_ELIMINATION=False for both phases.",
    )
    ap.set_defaults(logic_forbid_elimination=None)
    ap.add_argument("--clean", action="store_true", help="Delete output dirs before run.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    targets = parse_targets(args.targets)
    source = str(args.source_port).strip().upper()
    py = str(args.python_exe)
    workers = max(1, int(args.workers))
    eval_freq_ts = int(args.eval_freq)
    eval_freq_calls = max(1, int(round(eval_freq_ts / workers)))
    forbid = args.logic_forbid_elimination

    init_model = os.path.normpath(str(args.init_model))
    init_vecnorm = os.path.normpath(str(args.init_vecnorm)) if len(str(args.init_vecnorm).strip()) > 0 else ""
    if not args.dry_run:
        must_exist(init_model, "init model")
        if len(init_vecnorm) > 0:
            must_exist(init_vecnorm, "init vecnorm")

    print(
        "[ContinueConfig] "
        f"layout={args.layout_mode} path={args.path_type} src={source} tgts={targets} "
        f"phase1_cycles={args.phase1_cycles} phase2_cycles={args.phase2_cycles} "
        f"phase1_lr={float(args.phase1_lr):.6g} phase2_lr={float(args.phase2_lr):.6g} "
        f"phase2_free_layout={bool(args.phase2_free_layout)} "
        f"eval_freq_ts={eval_freq_ts} eval_freq_calls={eval_freq_calls} workers={workers}"
    )

    # Phase 1: fixed-layout refine from current model (r/omega mainly).
    p1_alias = f"{args.run_prefix}_p1_fixed"
    p1_tag = build_task_tag(
        layout_mode=str(args.layout_mode),
        path_type=str(args.path_type),
        source_port=source,
        targets=targets,
        run_alias=p1_alias,
    )
    p1_best_dir = os.path.join("models", "best", p1_tag)
    p1_eval_dir = os.path.join("models", "eval", p1_tag)
    if args.clean and not args.dry_run:
        shutil.rmtree(p1_best_dir, ignore_errors=True)
        shutil.rmtree(p1_eval_dir, ignore_errors=True)

    p1_steps = max(1, int(args.phase1_cycles)) * eval_freq_ts
    p1_cmd = [
        py,
        "train.py",
        "--layout-mode", str(args.layout_mode),
        "--path-type", str(args.path_type),
        "--logic-route-mode", "single_multi_target",
        "--source-port", source,
        "--target-port", targets[0],
        "--target-port-set", ",".join(targets),
        "--total-timesteps", str(p1_steps),
        "--eval-freq", str(eval_freq_calls),
        "--n-eval-episodes", str(int(args.n_eval_episodes)),
        "--patience-evals", str(int(args.patience_evals)),
        "--seed", str(int(args.seed_base)),
        "--workers", str(workers),
        "--run-alias", p1_alias,
        "--init-model", init_model,
        "--bootstrap-fixed-layout-from-init",
        "--learning-rate", f"{float(args.phase1_lr):.8f}",
    ]
    if len(init_vecnorm) > 0:
        p1_cmd.extend(["--init-vecnorm", init_vecnorm])
    if args.num_cylinders is not None:
        p1_cmd.extend(["--num-cylinders", str(int(args.num_cylinders))])
    if args.logic_min_active_r is not None:
        p1_cmd.extend(["--logic-min-active-r", f"{float(args.logic_min_active_r):.6f}"])
    if args.logic_max_r is not None:
        p1_cmd.extend(["--logic-max-r", f"{float(args.logic_max_r):.6f}"])
    if forbid is True:
        p1_cmd.append("--logic-forbid-elimination")
    elif forbid is False:
        p1_cmd.append("--logic-allow-elimination")
    run_cmd(p1_cmd, dry_run=bool(args.dry_run))

    p1_model = os.path.join(p1_best_dir, "best_model.zip")
    p1_vecnorm = os.path.join(p1_best_dir, "vecnormalize_best.pkl")
    if not args.dry_run:
        must_exist(p1_model, "phase1 best_model")
        must_exist(p1_vecnorm, "phase1 vecnormalize")

    phase2_cycles = max(0, int(args.phase2_cycles))
    if phase2_cycles <= 0:
        print("\n[Done] phase2 skipped (phase2-cycles <= 0).")
        print(f"[Phase1] {p1_best_dir}")
        print(f"[FinalModel] {p1_model}")
        print(f"[FinalVecNorm] {p1_vecnorm}")
        return

    # Phase 2: default keep fixed layout; optional free-layout unfreeze.
    p2_alias = f"{args.run_prefix}_{'p2_free' if bool(args.phase2_free_layout) else 'p2_fixed'}"
    p2_tag = build_task_tag(
        layout_mode=str(args.layout_mode),
        path_type=str(args.path_type),
        source_port=source,
        targets=targets,
        run_alias=p2_alias,
    )
    p2_best_dir = os.path.join("models", "best", p2_tag)
    p2_eval_dir = os.path.join("models", "eval", p2_tag)
    if args.clean and not args.dry_run:
        shutil.rmtree(p2_best_dir, ignore_errors=True)
        shutil.rmtree(p2_eval_dir, ignore_errors=True)

    p2_steps = int(phase2_cycles) * eval_freq_ts
    p2_cmd = [
        py,
        "train.py",
        "--layout-mode", str(args.layout_mode),
        "--path-type", str(args.path_type),
        "--logic-route-mode", "single_multi_target",
        "--source-port", source,
        "--target-port", targets[0],
        "--target-port-set", ",".join(targets),
        "--total-timesteps", str(p2_steps),
        "--eval-freq", str(eval_freq_calls),
        "--n-eval-episodes", str(int(args.n_eval_episodes)),
        "--patience-evals", str(int(args.patience_evals)),
        "--seed", str(int(args.seed_base) + 1),
        "--workers", str(workers),
        "--run-alias", p2_alias,
        "--init-model", p1_model,
        "--init-vecnorm", p1_vecnorm,
        "--learning-rate", f"{float(args.phase2_lr):.8f}",
    ]
    if bool(args.phase2_free_layout):
        p2_cmd.append("--shared-xy-one-stage-enable")
    if args.num_cylinders is not None:
        p2_cmd.extend(["--num-cylinders", str(int(args.num_cylinders))])
    if args.logic_min_active_r is not None:
        p2_cmd.extend(["--logic-min-active-r", f"{float(args.logic_min_active_r):.6f}"])
    if args.logic_max_r is not None:
        p2_cmd.extend(["--logic-max-r", f"{float(args.logic_max_r):.6f}"])
    if forbid is True:
        p2_cmd.append("--logic-forbid-elimination")
    elif forbid is False:
        p2_cmd.append("--logic-allow-elimination")
    run_cmd(p2_cmd, dry_run=bool(args.dry_run))

    p2_model = os.path.join(p2_best_dir, "best_model.zip")
    p2_vecnorm = os.path.join(p2_best_dir, "vecnormalize_best.pkl")
    if not args.dry_run:
        must_exist(p2_model, "phase2 best_model")
        must_exist(p2_vecnorm, "phase2 vecnormalize")

    print("\n[Done] continue refinement finished.")
    print(f"[Phase1] {p1_best_dir}")
    print(f"[Phase2] {p2_best_dir}")
    print(f"[FinalModel] {p2_model}")
    print(f"[FinalVecNorm] {p2_vecnorm}")


if __name__ == "__main__":
    main()
