import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import (
    GlobalOmegaControlConfig,
    InflowConfig,
    LayoutModeConfig,
    LogicBoxConfig,
    StokesCylinderConfig,
    TrainingSettingConfig,
)


def parse_layout_mode(layout_mode: str):
    mode = str(layout_mode)
    if mode.endswith("_inflow_u_fixed"):
        return mode[:-15], True, True
    if mode.endswith("_inflow_u"):
        return mode[:-9], True, True
    if mode.endswith("_inflow"):
        return mode[:-7], True, False
    return mode, False, False


def _safe_name(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(text))


def infer_dims():
    layout_mode = str(LayoutModeConfig.LAYOUT_MODE)
    base_mode, use_inflow, u_fixed = parse_layout_mode(layout_mode)
    n_cyl = int(StokesCylinderConfig.NUM_CYLINDERS)

    if base_mode == "fixed_grid_3x3":
        layout_dim = 9
    elif base_mode == "logic_box_layout":
        keep_xy_when_fixed = bool(getattr(LogicBoxConfig, "KEEP_XY_ACTION_WHEN_FIXED", False))
        fixed = bool(getattr(LogicBoxConfig, "FIXED_LAYOUT_ENABLE", False))
        layout_dim = n_cyl if (fixed and (not keep_xy_when_fixed)) else (3 * n_cyl)
    else:
        layout_dim = 3 * n_cyl

    inflow_dim = 0
    if use_inflow and (not u_fixed):
        if bool(getattr(InflowConfig, "OPTIMIZE_U", True)):
            inflow_dim += 1
        if bool(getattr(InflowConfig, "OPTIMIZE_V", True)):
            inflow_dim += 1

    omega_dim = 1 if bool(getattr(GlobalOmegaControlConfig, "OPTIMIZE_OMEGA", False)) else 0
    action_dim = layout_dim + inflow_dim + omega_dim

    use_shared_xy = (
        base_mode == "logic_box_layout"
        and bool(getattr(TrainingSettingConfig, "SHARED_XY_ONE_STAGE_ENABLE", False))
        and layout_dim == 3 * n_cyl
    )
    return {
        "layout_mode": layout_mode,
        "base_mode": base_mode,
        "n_cyl": n_cyl,
        "layout_dim": layout_dim,
        "inflow_dim": inflow_dim,
        "omega_dim": omega_dim,
        "action_dim": action_dim,
        "use_shared_xy": use_shared_xy,
    }


def add_box(ax, xy, w, h, text, fc="#ffffff", ec="#1f2937", lw=1.4, fontsize=10):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2.0,
        xy[1] + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        wrap=True,
    )
    return box


def add_arrow(ax, p1, p2, color="#334155", lw=1.4):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arr)


def reward_terms_text(base_mode: str):
    if base_mode != "logic_box_layout":
        return "R = -layout -cluster -block -flow + (task-specific terms)"

    mode = str(getattr(LogicBoxConfig, "REWARD_MODE", "hybrid")).strip().lower()
    port_terms = "miss + collision + wrong-side + outlet-pos + forward"
    stream_terms = "path-fit + path-cover"
    if mode in {"port_only", "port"}:
        body = f"port terms: {port_terms}"
    elif mode in {"streamline_only", "flow_only", "streamline"}:
        body = f"streamline terms: {stream_terms}"
    else:
        body = f"hybrid: {port_terms} + {stream_terms}"
    return (
        f"{body}\n"
        "always: overlap/boundary/empty-layout penalties\n"
        "plus inflow regularization (deadzone + target baseline)"
    )


def draw_shared_xy(ax, dims):
    n_cyl = int(dims["n_cyl"])
    tail_dim = int(dims["inflow_dim"] + dims["omega_dim"])

    add_box(
        ax,
        (0.03, 0.80),
        0.22,
        0.13,
        (
            "Task Context\n"
            f"mode={dims['layout_mode']}\n"
            "one-shot episode"
        ),
        fc="#f8fafc",
    )
    add_box(
        ax,
        (0.30, 0.80),
        0.22,
        0.13,
        "Observation\nobs = fluid_state + [dx, dy]\n(last 2 dims are target-conditioned)",
        fc="#eef2ff",
    )
    add_box(
        ax,
        (0.57, 0.80),
        0.22,
        0.13,
        "Shared Feature Extractor\nActorCriticPolicy MLP features",
        fc="#ecfeff",
    )

    add_box(
        ax,
        (0.03, 0.58),
        0.22,
        0.11,
        "Critic Branch\n2 x FC(256, tanh)\nvalue -> V(s)",
        fc="#fef9c3",
    )
    add_box(
        ax,
        (0.30, 0.58),
        0.22,
        0.11,
        "Masked Obs for XY\nset obs[-2:] = 0\n(target-agnostic x/y)",
        fc="#e0e7ff",
    )
    add_box(
        ax,
        (0.57, 0.58),
        0.18,
        0.11,
        f"XY Actor Head\n2 x FC(256, tanh)\nout: 2N={2 * n_cyl}",
        fc="#dcfce7",
    )
    add_box(
        ax,
        (0.78, 0.58),
        0.18,
        0.11,
        (
            "Control Actor Head\n"
            "2 x FC(256, tanh)\n"
            f"out: N+tail={n_cyl}+{tail_dim}"
        ),
        fc="#dbeafe",
    )

    add_box(
        ax,
        (0.30, 0.37),
        0.30,
        0.13,
        (
            "Action Assembly\n"
            "interleave to layout: [x0,y0,r0,...,xN,yN,rN]\n"
            f"append tail(inflow/omega), total dim={dims['action_dim']}"
        ),
        fc="#fee2e2",
    )
    add_box(
        ax,
        (0.64, 0.37),
        0.32,
        0.13,
        (
            "Decode + One-Shot Apply\n"
            "[-1,1] -> physical bounds (x/y/r, inflow, global omega)\n"
            "trace seed streamlines in current flow field"
        ),
        fc="#ffedd5",
    )

    add_box(
        ax,
        (0.03, 0.17),
        0.45,
        0.15,
        "Reward (logic box)\n" + reward_terms_text(dims["base_mode"]),
        fc="#dbeafe",
        fontsize=9,
    )
    add_box(
        ax,
        (0.53, 0.17),
        0.43,
        0.15,
        (
            "PPO Update\n"
            "n_steps=128, batch=64, lr=3e-4\n"
            "terminated=True after one action"
        ),
        fc="#dcfce7",
    )

    add_arrow(ax, (0.25, 0.865), (0.30, 0.865))
    add_arrow(ax, (0.52, 0.865), (0.57, 0.865))

    add_arrow(ax, (0.68, 0.80), (0.66, 0.69))
    add_arrow(ax, (0.72, 0.80), (0.87, 0.69))
    add_arrow(ax, (0.60, 0.80), (0.14, 0.69))
    add_arrow(ax, (0.41, 0.80), (0.41, 0.69))
    add_arrow(ax, (0.52, 0.635), (0.57, 0.635))

    add_arrow(ax, (0.66, 0.58), (0.45, 0.50))
    add_arrow(ax, (0.87, 0.58), (0.53, 0.50))
    add_arrow(ax, (0.60, 0.435), (0.64, 0.435))
    add_arrow(ax, (0.74, 0.37), (0.26, 0.32))
    add_arrow(ax, (0.48, 0.245), (0.53, 0.245))
    add_arrow(ax, (0.74, 0.32), (0.74, 0.37))
    add_arrow(ax, (0.64, 0.245), (0.14, 0.58))

    ax.text(
        0.80,
        0.43,
        "one-shot env step",
        fontsize=8,
        color="#64748b",
    )
    ax.text(
        0.66,
        0.27,
        "policy gradient update",
        fontsize=8,
        color="#64748b",
    )


def draw_standard(ax, dims):
    add_box(
        ax,
        (0.03, 0.77),
        0.22,
        0.14,
        f"Task Context\nmode={dims['layout_mode']}\none-shot episode",
        fc="#f8fafc",
    )
    add_box(
        ax,
        (0.30, 0.77),
        0.22,
        0.14,
        "Observation\nobs = fluid_state + [dx, dy]",
        fc="#eef2ff",
    )
    add_box(
        ax,
        (0.57, 0.77),
        0.22,
        0.14,
        "PPO MlpPolicy\nactor-critic shared encoder",
        fc="#ecfeff",
    )
    add_box(
        ax,
        (0.03, 0.52),
        0.22,
        0.12,
        "Actor Head\nmu + log_std -> Gaussian",
        fc="#dcfce7",
    )
    add_box(
        ax,
        (0.30, 0.52),
        0.22,
        0.12,
        "Critic Head\nV(s)",
        fc="#fef9c3",
    )
    add_box(
        ax,
        (0.57, 0.52),
        0.39,
        0.12,
        (
            "Action Decode + One-Shot Apply\n"
            f"action dim={dims['action_dim']} -> physical layout/inflow/omega"
        ),
        fc="#ffedd5",
    )
    add_box(
        ax,
        (0.03, 0.25),
        0.45,
        0.15,
        "Reward\n" + reward_terms_text(dims["base_mode"]),
        fc="#dbeafe",
        fontsize=9,
    )
    add_box(
        ax,
        (0.53, 0.25),
        0.43,
        0.15,
        "PPO Update\nterminated=True after one action",
        fc="#dcfce7",
    )

    add_arrow(ax, (0.25, 0.84), (0.30, 0.84))
    add_arrow(ax, (0.52, 0.84), (0.57, 0.84))
    add_arrow(ax, (0.68, 0.77), (0.14, 0.64))
    add_arrow(ax, (0.68, 0.77), (0.41, 0.64))
    add_arrow(ax, (0.68, 0.77), (0.76, 0.64))
    add_arrow(ax, (0.23, 0.58), (0.57, 0.58))
    add_arrow(ax, (0.76, 0.52), (0.26, 0.40))
    add_arrow(ax, (0.48, 0.325), (0.53, 0.325))
    add_arrow(ax, (0.75, 0.40), (0.68, 0.77))


def main():
    dims = infer_dims()
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    policy_name = "SharedXYActorCriticPolicy" if dims["use_shared_xy"] else "PPO MlpPolicy"
    ax.text(
        0.5,
        0.965,
        "Current DRL Network Architecture (One-Shot Optimization)",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.5,
        0.93,
        (
            f"policy={policy_name} | mode={dims['layout_mode']} | "
            f"action_dim={dims['action_dim']} (layout={dims['layout_dim']}, "
            f"inflow={dims['inflow_dim']}, omega={dims['omega_dim']})"
        ),
        ha="center",
        va="center",
        fontsize=9.5,
        color="#334155",
    )

    if dims["use_shared_xy"]:
        draw_shared_xy(ax, dims)
    else:
        draw_standard(ax, dims)

    ax.text(
        0.5,
        0.05,
        "Train loop: one action -> decode/apply -> streamline metrics -> reward -> PPO update",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#334155",
    )

    out_dir = "previews"
    os.makedirs(out_dir, exist_ok=True)
    out_main = os.path.join(out_dir, "network_architecture_current.png")
    out_mode = os.path.join(out_dir, f"network_architecture_{_safe_name(dims['layout_mode'])}.png")
    out_legacy = os.path.join(out_dir, "algorithm_model_diagram.png")
    fig.savefig(out_main, bbox_inches="tight")
    fig.savefig(out_mode, bbox_inches="tight")
    fig.savefig(out_legacy, bbox_inches="tight")
    plt.close(fig)

    print(out_main)
    if out_mode != out_main:
        print(out_mode)
    if out_legacy != out_main and out_legacy != out_mode:
        print(out_legacy)


if __name__ == "__main__":
    main()
