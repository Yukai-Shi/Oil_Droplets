import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import InflowConfig, RenderSettingConfig
from utils.calc import calculate_point_velocity, calculate_velocity_grid, is_legal


STROKES = ("soft_s", "soft_c", "soft_u", "soft_j")


def fixed_grid_3x3(pitch: float = 0.06, center=(0.0, 0.0)):
    cx, cy = float(center[0]), float(center[1])
    xs = np.array([cx - pitch, cx, cx + pitch], dtype=np.float32)
    ys = np.array([cy + pitch, cy, cy - pitch], dtype=np.float32)
    gx, gy = [], []
    for y in ys:
        for x in xs:
            gx.append(float(x))
            gy.append(float(y))
    return np.asarray(gx, dtype=np.float32), np.asarray(gy, dtype=np.float32)


def target_stroke(stroke: str, n: int = 240):
    s = np.linspace(0.0, 1.0, int(n), dtype=np.float32)
    stroke = str(stroke).lower()
    if stroke == "soft_s":
        x0, x1 = -0.105, 0.105
        y0, y1 = 0.045, -0.045
        xs = x0 + (x1 - x0) * s
        ys = y0 + (y1 - y0) * s + 0.026 * np.sin(2.0 * np.pi * s)
        start = np.array([x0, y0], dtype=np.float32)
    elif stroke == "soft_c":
        theta = np.linspace(2.20, -2.20, int(n), dtype=np.float32)
        xs = -0.005 + 0.075 * np.cos(theta)
        ys = 0.000 + 0.065 * np.sin(theta)
        start = np.array([float(xs[0]), float(ys[0])], dtype=np.float32)
    elif stroke == "soft_u":
        theta = np.linspace(np.pi, 2.0 * np.pi, int(n), dtype=np.float32)
        xs = 0.000 + 0.080 * np.cos(theta)
        ys = 0.035 + 0.075 * np.sin(theta)
        start = np.array([float(xs[0]), float(ys[0])], dtype=np.float32)
    elif stroke == "soft_j":
        xs = -0.090 + 0.165 * s
        ys = 0.060 - 0.130 * (s**1.7)
        ys += 0.026 * np.sin(np.pi * s) * (s > 0.35)
        start = np.array([float(xs[0]), float(ys[0])], dtype=np.float32)
    else:
        raise ValueError(f"Unknown stroke: {stroke}. Choose from {', '.join(STROKES)}")
    return np.stack([xs, ys], axis=1).astype(np.float32), start


def min_clearance_to_cylinders(points, cyl_x, cyl_y, radii):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    centers = np.stack([cyl_x, cyl_y], axis=1).astype(np.float32)
    d = np.sqrt(np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2))
    clearance = d - np.asarray(radii, dtype=np.float32)[None, :]
    return float(np.min(clearance))


def make_target_safe(points, start, cyl_x, cyl_y, radii, clearance=0.010, iters=40):
    """Push target stroke points outside cylinder clearance corridors."""
    safe = np.asarray(points, dtype=np.float32).copy()
    centers = np.stack([cyl_x, cyl_y], axis=1).astype(np.float32)
    rr = np.asarray(radii, dtype=np.float32) + float(clearance)
    for _ in range(int(iters)):
        moved = False
        for i in range(safe.shape[0]):
            p = safe[i]
            for c, r_safe in zip(centers, rr):
                v = p - c
                dist = float(np.linalg.norm(v))
                if dist < float(r_safe):
                    if dist < 1e-8:
                        v = np.array([1.0, 0.0], dtype=np.float32)
                        dist = 1.0
                    safe[i] = c + v / dist * float(r_safe)
                    moved = True
        # Light smoothing, while preserving endpoints well enough for seed intent.
        if safe.shape[0] > 4:
            old = safe.copy()
            safe[1:-1] = 0.18 * old[:-2] + 0.64 * old[1:-1] + 0.18 * old[2:]
        if not moved:
            break
    # Final projection after smoothing so the returned target is truly safe.
    for i in range(safe.shape[0]):
        p = safe[i]
        for c, r_safe in zip(centers, rr):
            v = p - c
            dist = float(np.linalg.norm(v))
            if dist < float(r_safe):
                if dist < 1e-8:
                    v = np.array([1.0, 0.0], dtype=np.float32)
                    dist = 1.0
                safe[i] = c + v / dist * float(r_safe)
    return safe.astype(np.float32), safe[0].astype(np.float32)


def segment_hits_cylinder(p0, p1, centers, radii, clearance=0.0005):
    p0 = np.asarray(p0, dtype=np.float32).reshape(2)
    p1 = np.asarray(p1, dtype=np.float32).reshape(2)
    seg = p1 - p0
    denom = float(np.dot(seg, seg))
    for c, r in zip(centers, radii):
        rr = float(r) + float(clearance)
        if rr <= 0.0:
            continue
        if denom <= 1e-14:
            d2 = float(np.sum((p0 - c) ** 2))
        else:
            t = float(np.dot(c - p0, seg) / denom)
            t = min(1.0, max(0.0, t))
            closest = p0 + t * seg
            d2 = float(np.sum((closest - c) ** 2))
        if d2 <= rr * rr:
            return True
    return False


def trace_streamline(
    start,
    cyl_x,
    cyl_y,
    radii,
    omega_rad_s,
    inflow_u,
    inflow_v,
    dt=0.006,
    steps=3600,
):
    centers = np.stack([cyl_x, cyl_y], axis=1).astype(np.float32)
    omegas = np.full(len(cyl_x), float(omega_rad_s), dtype=np.float32)
    pos = np.asarray(start, dtype=np.float32).copy()
    path = [pos.copy()]
    for _ in range(int(steps)):
        vx, vy = calculate_point_velocity(
            float(pos[0]), float(pos[1]), cyl_x, cyl_y, radii, omegas
        )
        vel = np.array([float(vx) + float(inflow_u), float(vy) + float(inflow_v)], dtype=np.float32)
        nxt = pos + vel * float(dt)
        if segment_hits_cylinder(pos, nxt, centers, radii) or not is_legal(
            nxt,
            centers=centers,
            radii=radii,
            xlim=RenderSettingConfig.X_LIM,
            ylim=RenderSettingConfig.Y_LIM,
        ):
            break
        pos = nxt
        path.append(pos.copy())
        if pos[0] > 0.14 or abs(pos[1]) > 0.14:
            break
    return np.asarray(path, dtype=np.float32)


def path_score(path, target):
    if len(path) < 5:
        return 1.0
    p = np.asarray(path, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    d2 = np.sum((p[:, None, :] - t[None, :, :]) ** 2, axis=2)
    path_to_target = float(np.mean(np.sqrt(np.min(d2, axis=1))))
    target_to_path = float(np.mean(np.sqrt(np.min(d2, axis=0))))
    coverage = min(1.0, float(len(path)) / max(1.0, float(len(target))))
    return path_to_target + target_to_path + 0.06 * (1.0 - coverage)


def render_case(
    output_path,
    stroke,
    target,
    start,
    path,
    cyl_x,
    cyl_y,
    radii,
    omega_rad_s,
    inflow_u,
    inflow_v,
    min_target_clearance=None,
    grid_name="3x3",
):
    xlim = RenderSettingConfig.X_LIM
    ylim = RenderSettingConfig.Y_LIM
    width, height = 900, 900
    margin = 70

    def xy_to_px(x, y):
        px = margin + (float(x) - xlim[0]) / max(xlim[1] - xlim[0], 1e-8) * (width - 2 * margin)
        py = height - margin - (float(y) - ylim[0]) / max(ylim[1] - ylim[0], 1e-8) * (height - 2 * margin)
        return int(round(px)), int(round(py))

    def data_len_to_px(v):
        return float(v) / max(xlim[1] - xlim[0], 1e-8) * (width - 2 * margin)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Plot area and light grid.
    x0p, y0p = xy_to_px(xlim[0], ylim[0])
    x1p, y1p = xy_to_px(xlim[1], ylim[1])
    left, right = min(x0p, x1p), max(x0p, x1p)
    top, bottom = min(y0p, y1p), max(y0p, y1p)
    draw.rectangle([left, top, right, bottom], outline=(40, 40, 40), width=2)
    for val in np.linspace(xlim[0], xlim[1], 9):
        px, _ = xy_to_px(val, 0.0)
        draw.line([px, top, px, bottom], fill=(225, 225, 225), width=1)
    for val in np.linspace(ylim[0], ylim[1], 9):
        _, py = xy_to_px(0.0, val)
        draw.line([left, py, right, py], fill=(225, 225, 225), width=1)

    grid = np.linspace(xlim[0], xlim[1], 27)
    X, Y = np.meshgrid(grid, grid)
    omegas = np.full(len(cyl_x), float(omega_rad_s), dtype=np.float32)
    U, V = calculate_velocity_grid(X, Y, cyl_x, cyl_y, radii, omegas)
    U = U + float(inflow_u)
    V = V + float(inflow_v)
    speed = np.sqrt(U * U + V * V)
    max_speed = max(float(np.max(speed)), 1e-8)
    for xx, yy, uu, vv, ss in zip(X.reshape(-1), Y.reshape(-1), U.reshape(-1), V.reshape(-1), speed.reshape(-1)):
        if not (left <= xy_to_px(xx, yy)[0] <= right):
            continue
        px, py = xy_to_px(xx, yy)
        scale = 24.0 / max_speed
        dx = float(uu) * scale
        dy = -float(vv) * scale
        color = (160, 195, 225) if ss < 0.55 * max_speed else (55, 130, 190)
        draw.line([px, py, int(px + dx), int(py + dy)], fill=color, width=1)

    points_per_data = data_len_to_px(1.0)
    for i, (x, y, r) in enumerate(zip(cyl_x, cyl_y, radii)):
        px, py = xy_to_px(x, y)
        rr = max(3, int(round(float(r) * points_per_data)))
        draw.ellipse([px - rr, py - rr, px + rr, py + rr], fill=(165, 215, 165), outline=(50, 120, 60), width=2)
        draw.text((px - 3, py - 5), str(i), fill=(20, 50, 20), font=font)

    target_pts = [xy_to_px(x, y) for x, y in target]
    for k in range(len(target_pts) - 1):
        if k % 8 < 5:
            draw.line([target_pts[k], target_pts[k + 1]], fill=(21, 128, 61), width=4)
    if len(path) > 1:
        path_pts = [xy_to_px(x, y) for x, y in path]
        draw.line(path_pts, fill=(245, 158, 11), width=5)
        ex, ey = path_pts[-1]
        draw.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=(245, 158, 11))
    sx, sy = xy_to_px(start[0], start[1])
    draw.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], fill=(22, 163, 74))

    hz = float(omega_rad_s) / (2.0 * np.pi)
    clr_txt = "" if min_target_clearance is None else f" | target clr={float(min_target_clearance):.3f}"
    title = f"{grid_name} stroke preview | {stroke} | f={hz:.2f} Hz | inflow=({inflow_u:.3f},{inflow_v:.3f}){clr_txt}"
    draw.text((left, 22), title, fill=(20, 20, 20), font=font)
    draw.line([right - 220, bottom + 30, right - 180, bottom + 30], fill=(21, 128, 61), width=4)
    draw.text((right - 172, bottom + 24), "target stroke", fill=(20, 20, 20), font=font)
    draw.line([right - 220, bottom + 52, right - 180, bottom + 52], fill=(245, 158, 11), width=5)
    draw.text((right - 172, bottom + 46), "traced streamline", fill=(20, 20, 20), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def run_one(args, stroke):
    target, start = target_stroke(stroke)
    cyl_x, cyl_y = fixed_grid_3x3(pitch=float(args.pitch))
    radii = np.full(len(cyl_x), float(args.radius), dtype=np.float32)
    if bool(args.safe_target):
        target, start = make_target_safe(
            target,
            start,
            cyl_x,
            cyl_y,
            radii,
            clearance=float(args.target_clearance),
            iters=int(args.safe_target_iters),
        )
    target_clearance = min_clearance_to_cylinders(target, cyl_x, cyl_y, radii)
    inflow_u = float(args.inflow_u)
    inflow_v = float(args.inflow_v)

    if args.sweep:
        best = None
        for hz in np.linspace(float(args.omega_min_hz), float(args.omega_max_hz), int(args.sweep_count)):
            path = trace_streamline(
                start, cyl_x, cyl_y, radii, 2.0 * np.pi * float(hz), inflow_u, inflow_v,
                dt=float(args.dt), steps=int(args.steps)
            )
            score = path_score(path, target)
            if best is None or score < best[0]:
                best = (score, float(hz), path)
        score, hz, path = best
    else:
        hz = float(args.omega_hz)
        path = trace_streamline(
            start, cyl_x, cyl_y, radii, 2.0 * np.pi * hz, inflow_u, inflow_v,
            dt=float(args.dt), steps=int(args.steps)
        )
        score = path_score(path, target)

    out_dir = Path(args.out_dir)
    out_path = out_dir / f"stroke_3x3_{stroke}_f{hz:+.2f}Hz.png"
    render_case(
        out_path, stroke, target, start, path, cyl_x, cyl_y, radii,
        2.0 * np.pi * hz, inflow_u, inflow_v,
        min_target_clearance=target_clearance,
    )
    print(
        f"[Saved] {out_path} | score={score:.4f} | path_points={len(path)} "
        f"| f={hz:.3f} Hz | target_clearance={target_clearance:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Preview 3x3 fixed-grid streamline stroke shaping.")
    parser.add_argument("--stroke", default="all", help=f"One of {','.join(STROKES)} or all.")
    parser.add_argument("--out-dir", default="models/stroke_previews")
    parser.add_argument("--pitch", type=float, default=0.06)
    parser.add_argument("--radius", type=float, default=0.010)
    parser.add_argument("--target-clearance", type=float, default=0.012)
    parser.add_argument("--safe-target-iters", type=int, default=80)
    parser.add_argument("--no-safe-target", action="store_false", dest="safe_target")
    parser.add_argument("--inflow-u", type=float, default=float(InflowConfig.U_IN))
    parser.add_argument("--inflow-v", type=float, default=0.0)
    parser.add_argument("--omega-hz", type=float, default=0.5)
    parser.add_argument("--omega-min-hz", type=float, default=-1.0)
    parser.add_argument("--omega-max-hz", type=float, default=1.0)
    parser.add_argument("--sweep-count", type=int, default=61)
    parser.add_argument("--dt", type=float, default=0.006)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--no-sweep", action="store_false", dest="sweep")
    parser.set_defaults(sweep=True)
    parser.set_defaults(safe_target=True)
    return parser.parse_args()


def main():
    args = parse_args()
    strokes = list(STROKES) if str(args.stroke).lower() == "all" else [str(args.stroke).lower()]
    for stroke in strokes:
        run_one(args, stroke)


if __name__ == "__main__":
    main()
