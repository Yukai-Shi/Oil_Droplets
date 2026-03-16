import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, xy, w, h, text, fc="#ffffff", ec="#1f2937", lw=1.5, fontsize=11):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        wrap=True,
    )
    return box


def add_arrow(ax, p1, p2, color="#374151", lw=1.5):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arr)


def main():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "DRL-FlowMatch-OS: One-Shot Flowline Matching",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )

    # Top row
    add_box(
        ax,
        (0.04, 0.74),
        0.19,
        0.14,
        "Input\nPreset trajectory\nmode: free_layout / fixed_grid_3x3",
        fc="#f8fafc",
    )
    add_box(
        ax,
        (0.29, 0.74),
        0.19,
        0.14,
        "Observation\nFluid state + dx,dy\n(one-shot episode reset)",
        fc="#eef2ff",
    )
    add_box(
        ax,
        (0.54, 0.74),
        0.19,
        0.14,
        "PPO Policy\n(MlpPolicy)\noutputs one action",
        fc="#ecfeff",
    )
    add_box(
        ax,
        (0.79, 0.74),
        0.17,
        0.14,
        "Action Decode\n[-1,1] -> physical\nx,y,r or r-only",
        fc="#f0fdf4",
    )

    # Middle row
    add_box(
        ax,
        (0.09, 0.46),
        0.24,
        0.17,
        "Apply Layout (One-Shot)\nSet cylinder positions/radii\nω fixed and same direction",
        fc="#fff7ed",
    )
    add_box(
        ax,
        (0.39, 0.46),
        0.24,
        0.17,
        "Flowline Metrics Along Path\n- direction mismatch\n- speed deficit\n- reverse-flow ratio",
        fc="#fff1f2",
    )
    add_box(
        ax,
        (0.69, 0.46),
        0.24,
        0.17,
        "Regularization Terms\n- overlap/boundary\n- path blocking\n- cluster (free mode)",
        fc="#fefce8",
    )

    # Bottom row
    add_box(
        ax,
        (0.20, 0.19),
        0.25,
        0.17,
        "Reward\nR = - w_flow * flow penalties\n    - w_reg * regularization",
        fc="#e0f2fe",
    )
    add_box(
        ax,
        (0.55, 0.19),
        0.25,
        0.17,
        "PPO Update\ncollect one-shot transitions\noptimize policy parameters",
        fc="#dcfce7",
    )

    # arrows top
    add_arrow(ax, (0.23, 0.81), (0.29, 0.81))
    add_arrow(ax, (0.48, 0.81), (0.54, 0.81))
    add_arrow(ax, (0.73, 0.81), (0.79, 0.81))

    # arrows down
    add_arrow(ax, (0.875, 0.74), (0.21, 0.63))
    add_arrow(ax, (0.21, 0.46), (0.39, 0.54))
    add_arrow(ax, (0.63, 0.54), (0.69, 0.54))
    add_arrow(ax, (0.51, 0.46), (0.325, 0.36))
    add_arrow(ax, (0.81, 0.46), (0.325, 0.36))

    # reward -> update -> policy loop
    add_arrow(ax, (0.45, 0.275), (0.55, 0.275))
    add_arrow(ax, (0.675, 0.36), (0.635, 0.74))
    ax.text(
        0.69,
        0.58,
        "backprop through RL objective",
        fontsize=9,
        color="#475569",
        rotation=77,
    )

    ax.text(
        0.5,
        0.06,
        "Training: one action per episode (static optimization)\n"
        "Evaluation: optional long particle rollout under fixed optimized layout",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
    )

    out_dir = "previews"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "algorithm_model_diagram.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
