import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    FixedGrid3x3Config,
    Gate3LevelConfig,
    GlobalOmegaControlConfig,
    get_logic_box_ranges,
    get_logic_max_radius,
    get_logic_multi_route_pairs,
    get_logic_port_coordinates,
    InflowConfig,
    LogicBoxConfig,
    RenderSettingConfig,
    StokesCylinderConfig,
    TrainingSettingConfig,
)
from envs.FluidEnv import FluidEnv
from utils.calc import calculate_point_velocity, is_legal


def layout_overlap_penalty(x, y, r, margin=0.002):
    penalty = 0.0
    n = len(r)
    for i in range(n):
        if r[i] <= 0:
            continue
        for j in range(i + 1, n):
            if r[j] <= 0:
                continue
            dij = np.hypot(x[i] - x[j], y[i] - y[j])
            overlap = (r[i] + r[j] + margin) - dij
            if overlap > 0:
                penalty += overlap
    return penalty


def layout_boundary_penalty(x, y, r, xlim, ylim, margin=0.002):
    penalty = 0.0
    for xi, yi, ri in zip(x, y, r):
        if ri <= 0:
            continue
        penalty += max(0.0, xlim[0] - (xi - ri - margin))
        penalty += max(0.0, (xi + ri + margin) - xlim[1])
        penalty += max(0.0, ylim[0] - (yi - ri - margin))
        penalty += max(0.0, (yi + ri + margin) - ylim[1])
    return penalty


def path_blocking_penalty(path: np.ndarray, x: np.ndarray, y: np.ndarray, r: np.ndarray, clearance=0.006):
    """
    Penalize cylinders that intrude into a corridor around the target path.
    """
    penalty = 0.0
    for xi, yi, ri in zip(x, y, r):
        if ri <= 0:
            continue
        d = np.linalg.norm(path - np.array([xi, yi], dtype=np.float32), axis=1)
        min_d = float(np.min(d))
        overlap = (float(ri) + float(clearance)) - min_d
        if overlap > 0:
            penalty += overlap
    return penalty


def _crossing_y_at_x(p0: np.ndarray, p1: np.ndarray, target_x: float):
    x0, x1 = float(p0[0]), float(p1[0])
    if (x0 - target_x) * (x1 - target_x) > 0.0:
        return False, None
    if abs(x1 - x0) < 1e-10:
        return False, None
    alpha = (target_x - x0) / (x1 - x0)
    if alpha < 0.0 or alpha > 1.0:
        return False, None
    y_cross = float(p0[1] + alpha * (float(p1[1]) - float(p0[1])))
    return True, y_cross


def _crossing_x_at_y(p0: np.ndarray, p1: np.ndarray, target_y: float):
    y0, y1 = float(p0[1]), float(p1[1])
    if (y0 - target_y) * (y1 - target_y) > 0.0:
        return False, None, None
    if abs(y1 - y0) < 1e-10:
        return False, None, None
    alpha = (target_y - y0) / (y1 - y0)
    if alpha < 0.0 or alpha > 1.0:
        return False, None, None
    x_cross = float(p0[0] + alpha * (float(p1[0]) - float(p0[0])))
    return True, x_cross, float(alpha)


def trace_streamline_until_x(
    seed: np.ndarray,
    target_x: float,
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    omega: float,
    inflow_u: float,
    inflow_v: float,
    dt: float,
    max_steps: int,
):
    pos = np.asarray(seed, dtype=np.float32).copy()
    centers = np.stack([x, y], axis=1).astype(np.float32)
    omegas = np.full(len(x), float(omega), dtype=np.float32)
    mode = str(getattr(Gate3LevelConfig, "TRACE_MODE", "x_march")).lower()

    if mode == "x_march":
        x_step = float(getattr(Gate3LevelConfig, "X_STEP", 0.003))
        max_x_steps = int(getattr(Gate3LevelConfig, "MAX_X_STEPS", 140))
        min_forward_vx = float(getattr(Gate3LevelConfig, "MIN_FORWARD_VX", 1e-4))

        direction = 1.0 if float(target_x) >= float(seed[0]) else -1.0
        for _ in range(max_x_steps):
            remain = direction * (float(target_x) - float(pos[0]))
            if remain <= 1e-8:
                return True, float(pos[1]), False

            vx, vy = calculate_point_velocity(
                float(pos[0]),
                float(pos[1]),
                x,
                y,
                r,
                omegas,
            )
            vx = float(vx) + float(inflow_u)
            vy = float(vy) + float(inflow_v)
            if direction * vx < min_forward_vx:
                return False, None, False

            dx = direction * min(abs(x_step), abs(remain))
            dt_eff = dx / vx
            nxt = np.array(
                [float(pos[0] + dx), float(pos[1] + vy * dt_eff)],
                dtype=np.float32,
            )

            if not is_legal(nxt, centers=centers, radii=r):
                return False, None, True
            pos = nxt

        return False, None, False

    # Legacy time marching.
    for _ in range(int(max_steps)):
        vx, vy = calculate_point_velocity(
            float(pos[0]),
            float(pos[1]),
            x,
            y,
            r,
            omegas,
        )
        nxt = pos + np.array(
            [vx + float(inflow_u), vy + float(inflow_v)],
            dtype=np.float32,
        ) * float(dt)

        crossed, y_cross = _crossing_y_at_x(pos, nxt, float(target_x))
        if crossed:
            return True, float(y_cross), False

        pos = nxt
        if not is_legal(pos, centers=centers, radii=r):
            return False, None, True

    return False, None, False


def gate3_lane_passage_metrics(
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    omega: float,
    inflow_u: float,
    inflow_v: float,
    lane_indices=None,
):
    seeds = Gate3LevelConfig.lane_seed_points()
    target_y = np.asarray(Gate3LevelConfig.LANE_TARGET_Y, dtype=np.float32)
    check_x = float(Gate3LevelConfig.CHECK_X)
    if lane_indices is None:
        lane_indices = [0, 1, 2]
    lane_indices = [int(i) for i in lane_indices]

    y_errors = []
    y_cross_list = []
    miss = 0
    collision = 0

    for i in lane_indices:
        reached, y_cross, hit_cyl = trace_streamline_until_x(
            seed=seeds[i],
            target_x=check_x,
            x=x,
            y=y,
            r=r,
            omega=float(omega),
            inflow_u=float(inflow_u),
            inflow_v=float(inflow_v),
            dt=float(Gate3LevelConfig.TRACE_DT),
            max_steps=int(Gate3LevelConfig.TRACE_STEPS),
        )
        if reached and y_cross is not None:
            y_cross_list.append(float(y_cross))
            y_errors.append(abs(float(y_cross) - float(target_y[i])))
        else:
            miss += 1
            if hit_cyl:
                collision += 1
            y_errors.append(float(Gate3LevelConfig.MISS_Y_PENALTY))
            y_cross_list.append(None)

    # Keep top > mid > bottom ordering at the checkpoint.
    order_pen = 0.0
    if lane_indices == [0, 1, 2] and all(v is not None for v in y_cross_list):
        y_top, y_mid, y_bot = float(y_cross_list[0]), float(y_cross_list[1]), float(y_cross_list[2])
        order_pen += max(0.0, y_mid - y_top)
        order_pen += max(0.0, y_bot - y_mid)

    return {
        "lane_error_mean": float(np.mean(y_errors)),
        "lane_miss_ratio": float(miss / max(1, len(lane_indices))),
        "lane_collision_ratio": float(collision / max(1, len(lane_indices))),
        "lane_order_penalty": float(order_pen),
        "lane_y_cross": y_cross_list,
        "lane_target_y": [float(target_y[i]) for i in lane_indices],
        "lane_indices": lane_indices,
    }


def logic_box_bounds():
    return get_logic_box_ranges()


def logic_box_ports():
    (x0, x1), (y0, y1) = logic_box_bounds()
    p = get_logic_port_coordinates()
    ports = {}
    for i, yy in enumerate(np.asarray(p["left_y"], dtype=np.float32)):
        ports[f"L{i}"] = ("left", np.array([x0, float(yy)], dtype=np.float32))
    for i, yy in enumerate(np.asarray(p["right_y"], dtype=np.float32)):
        ports[f"R{i}"] = ("right", np.array([x1, float(yy)], dtype=np.float32))
    ports["T0"] = ("top", np.array([float(p["top_x"]), y1], dtype=np.float32))
    ports["B0"] = ("bottom", np.array([float(p["bottom_x"]), y0], dtype=np.float32))
    return ports


def _logic_route_mode() -> str:
    return str(getattr(LogicBoxConfig, "ROUTE_MODE", "single")).strip().lower()


def _logic_is_multi_route_mode() -> bool:
    return _logic_route_mode() in {"multi", "multi_map", "multi_route", "mapping"}


def _logic_is_single_multi_target_mode() -> bool:
    return _logic_route_mode() in {
        "single_multi_target",
        "single_source_multi_target",
        "one_to_many",
        "one_to_three",
    }


def logic_seed_offsets(profile: str = "train") -> np.ndarray:
    p = str(profile).strip().lower()
    if p in {"eval", "evaluation", "test", "render", "inference"}:
        raw = getattr(LogicBoxConfig, "EVAL_SOURCE_SEED_DY", None)
    elif p in {"train", "training"}:
        raw = getattr(LogicBoxConfig, "TRAIN_SOURCE_SEED_DY", None)
    else:
        raw = None
    if raw is None:
        raw = getattr(LogicBoxConfig, "SOURCE_SEED_DY", [0.0])
    arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 0:
        arr = np.array([0.0], dtype=np.float32)
    return arr


def logic_target_port_set():
    ports = logic_box_ports()
    raw = getattr(LogicBoxConfig, "TARGET_PORT_SET", ["R0", "R1", "R2"])
    out = []
    if isinstance(raw, (list, tuple)):
        for t in raw:
            tt = str(t).strip().upper()
            if tt in ports and ports[tt][0] != "left":
                out.append(tt)
    if len(out) == 0:
        out = ["R0", "R1", "R2"]
    return out


def _logic_sanitize_pair(src: str, tgt: str):
    ports = logic_box_ports()
    s = str(src).strip().upper()
    t = str(tgt).strip().upper()
    if s not in ports:
        s = "L1"
    if t not in ports:
        t = "R1"
    s_side, _ = ports[s]
    if s_side != "left":
        s = "L1"
    return s, t


def logic_box_active_route_pairs():
    ports = logic_box_ports()
    pairs = []
    for src, tgt in get_logic_multi_route_pairs():
        s, t = _logic_sanitize_pair(src, tgt)
        if s in ports and t in ports:
            pairs.append((s, t))
    if len(pairs) == 0:
        pairs = [("L0", "T0"), ("L1", "R1"), ("L2", "B0")]
    return pairs


def build_logic_route_path(source_port: str, target_port: str, steps: int) -> np.ndarray:
    ports = logic_box_ports()
    src, tgt = _logic_sanitize_pair(source_port, target_port)
    src_side, p0 = ports[src]
    tgt_side, p3 = ports[tgt]
    p0 = p0.astype(np.float32).copy()
    p3 = p3.astype(np.float32).copy()
    (x0_box, x1_box), (y0_box, y1_box) = logic_box_bounds()
    if src_side == "left":
        p0[0] = float(x0_box) + max(
            1e-4, float(getattr(LogicBoxConfig, "SOURCE_SEED_X_INSET", 1e-4))
        )
    if tgt_side == "right":
        p3[0] = float(x1_box)
    elif tgt_side == "top":
        p3[1] = float(y1_box)
    elif tgt_side == "bottom":
        p3[1] = float(y0_box)

    def cubic_bezier(a, b, c, d, u):
        return (
            ((1.0 - u) ** 3)[:, None] * a[None, :]
            + (3.0 * ((1.0 - u) ** 2) * u)[:, None] * b[None, :]
            + (3.0 * (1.0 - u) * (u ** 2))[:, None] * c[None, :]
            + (u ** 3)[:, None] * d[None, :]
        )

    if tgt_side == "right":
        p1 = np.array([p0[0] + 0.30 * (x1_box - x0_box), p0[1]], dtype=np.float32)
        p2 = np.array([p0[0] + 0.80 * (x1_box - x0_box), p3[1]], dtype=np.float32)
    elif tgt_side == "top":
        p1 = np.array([p0[0] + 0.35 * (x1_box - x0_box), p0[1]], dtype=np.float32)
        p2 = np.array([p3[0], p0[1] + 0.75 * (y1_box - p0[1])], dtype=np.float32)
    elif tgt_side == "bottom":
        p1 = np.array([p0[0] + 0.35 * (x1_box - x0_box), p0[1]], dtype=np.float32)
        p2 = np.array([p3[0], p0[1] + 0.75 * (y0_box - p0[1])], dtype=np.float32)
    else:
        p1 = np.array([0.5 * (p0[0] + p3[0]), p0[1]], dtype=np.float32)
        p2 = np.array([0.5 * (p0[0] + p3[0]), p3[1]], dtype=np.float32)

    n_steps = max(2, int(steps))
    path = cubic_bezier(p0, p1, p2, p3, np.linspace(0.0, 1.0, n_steps))
    return path.astype(np.float32)


def _logic_exit_match_error(side: str, coord: float, target_side: str, target_xy: np.ndarray):
    (x0, x1), (y0, y1) = logic_box_bounds()
    sigma = max(float(LogicBoxConfig.OUTLET_SIGMA), 1e-6)

    if side == "left":
        ex, ey = float(x0), float(coord)
    elif side == "right":
        ex, ey = float(x1), float(coord)
    elif side == "top":
        ex, ey = float(coord), float(y1)
    elif side == "bottom":
        ex, ey = float(coord), float(y0)
    else:
        ex, ey = float(target_xy[0]), float(target_xy[1])

    if side != target_side:
        # Dense wrong-side distance so policy gets gradient before exact side match.
        w = max(float(x1) - float(x0), 1e-6)
        h = max(float(y1) - float(y0), 1e-6)
        if target_side == "right":
            drift = max(0.0, float(x1) - ex) / w
        elif target_side == "left":
            drift = max(0.0, ex - float(x0)) / w
        elif target_side == "top":
            drift = max(0.0, float(y1) - ey) / h
        else:
            drift = max(0.0, ey - float(y0)) / h
        return float(1.0 + drift), False
    if side in {"left", "right"}:
        d = abs(float(coord) - float(target_xy[1])) / sigma
    else:
        d = abs(float(coord) - float(target_xy[0])) / sigma
    return float(d), True


def trace_streamline_until_box_exit(
    seed: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    omega: float,
    inflow_u: float,
    inflow_v: float,
):
    (x0, x1), (y0, y1) = logic_box_bounds()
    dt = float(LogicBoxConfig.TRACE_DT)
    max_steps = int(LogicBoxConfig.TRACE_STEPS)
    mode = str(getattr(LogicBoxConfig, "TRACE_MODE", "x_march")).lower()

    # Keep float64 here to avoid x-march stalling near right boundary due
    # float32 precision (can cause false "miss" when x almost reaches x1).
    pos = np.asarray(seed, dtype=np.float64).copy()
    history = [pos.copy().tolist()]
    centers = np.stack([x, y], axis=1).astype(np.float32)
    omegas = np.full(len(x), float(omega), dtype=np.float32)

    init_vx = 0.0
    init_vy = 0.0
    if mode == "x_march":
        x_step = float(getattr(LogicBoxConfig, "X_STEP", 0.003))
        max_x_steps = int(getattr(LogicBoxConfig, "MAX_X_STEPS", max_steps))
        min_forward_vx = float(getattr(LogicBoxConfig, "MIN_FORWARD_VX", 1e-4))
        low_vx_use_fallback = bool(
            getattr(LogicBoxConfig, "LOW_VX_USE_TIMETRACE_FALLBACK", True)
        )
        low_vx_patience = max(1, int(getattr(LogicBoxConfig, "LOW_VX_PATIENCE", 40)))
        low_vx_fallback_dt = float(getattr(LogicBoxConfig, "LOW_VX_FALLBACK_DT", dt))
        x_eps = 1e-6
        low_vx_count = 0

        for k in range(max_x_steps):
            remain = float(x1) - float(pos[0])
            if remain <= x_eps:
                return {
                    "exited": True,
                    "side": "right",
                    "coord": float(pos[1]),
                    "collision": False,
                    "init_vx": init_vx,
                    "init_vy": init_vy,
                    "history": history,
                }

            vx, vy = calculate_point_velocity(
                float(pos[0]),
                float(pos[1]),
                x,
                y,
                r,
                omegas,
            )
            vx = float(vx) + float(inflow_u)
            vy = float(vy) + float(inflow_v)
            if k == 0:
                init_vx, init_vy = float(vx), float(vy)

            use_low_vx_fallback = float(vx) < min_forward_vx
            if use_low_vx_fallback and low_vx_use_fallback:
                low_vx_count += 1
                if low_vx_count > low_vx_patience:
                    return {
                        "exited": False,
                        "side": None,
                        "coord": None,
                        "collision": False,
                        "init_vx": init_vx,
                        "init_vy": init_vy,
                        "history": history,
                    }
                nxt = pos + np.array(
                    [float(vx) * low_vx_fallback_dt, float(vy) * low_vx_fallback_dt],
                    dtype=np.float64,
                )
            else:
                if use_low_vx_fallback:
                    return {
                        "exited": False,
                        "side": None,
                        "coord": None,
                        "collision": False,
                        "init_vx": init_vx,
                        "init_vy": init_vy,
                        "history": history,
                    }

                low_vx_count = 0
                dx = min(float(x_step), max(remain, 0.0))
                if dx <= x_eps:
                    return {
                        "exited": True,
                        "side": "right",
                        "coord": float(pos[1]),
                        "collision": False,
                        "init_vx": init_vx,
                        "init_vy": init_vy,
                        "history": history,
                    }

                dt_eff = dx / float(vx)
                nxt = np.array(
                    [float(pos[0] + dx), float(pos[1] + vy * dt_eff)],
                    dtype=np.float64,
                )
            history.append(nxt.copy().tolist())
            if not is_legal(nxt, centers=centers, radii=r):
                return {
                    "exited": False,
                    "side": None,
                    "coord": None,
                    "collision": True,
                    "init_vx": init_vx,
                    "init_vy": init_vy,
                    "history": history,
                }

            candidates = []
            dx_seg = float(nxt[0] - pos[0])
            dy_seg = float(nxt[1] - pos[1])
            if abs(dx_seg) > 1e-12:
                a_right = (float(x1) - float(pos[0])) / dx_seg
                if 0.0 <= a_right <= 1.0:
                    candidates.append(
                        (
                            float(a_right),
                            "right",
                            float(pos[1] + float(a_right) * dy_seg),
                        )
                    )
                a_left = (float(x0) - float(pos[0])) / dx_seg
                if 0.0 <= a_left <= 1.0:
                    candidates.append(
                        (
                            float(a_left),
                            "left",
                            float(pos[1] + float(a_left) * dy_seg),
                        )
                    )

            crossed_top, x_top, a_top = _crossing_x_at_y(pos, nxt, float(y1))
            if crossed_top:
                candidates.append((float(a_top), "top", float(x_top)))

            crossed_bot, x_bot, a_bot = _crossing_x_at_y(pos, nxt, float(y0))
            if crossed_bot:
                candidates.append((float(a_bot), "bottom", float(x_bot)))

            if len(candidates) > 0:
                candidates.sort(key=lambda t: t[0])
                _, side, coord = candidates[0]
                return {
                    "exited": True,
                    "side": side,
                    "coord": float(coord),
                    "collision": False,
                    "init_vx": init_vx,
                    "init_vy": init_vy,
                    "history": history,
                }

            pos = nxt
    else:
        for k in range(max_steps):
            vx, vy = calculate_point_velocity(
                float(pos[0]),
                float(pos[1]),
                x,
                y,
                r,
                omegas,
            )
            vx = float(vx) + float(inflow_u)
            vy = float(vy) + float(inflow_v)
            if k == 0:
                init_vx, init_vy = float(vx), float(vy)

            pos = pos + np.array([vx * dt, vy * dt], dtype=np.float32)
            history.append(pos.copy().tolist())
            if not is_legal(pos, centers=centers, radii=r):
                return {
                    "exited": False,
                    "side": None,
                    "coord": None,
                    "collision": True,
                    "init_vx": init_vx,
                    "init_vy": init_vy,
                    "history": history,
                }

            if float(pos[0]) >= float(x1):
                return {
                    "exited": True,
                    "side": "right",
                    "coord": float(pos[1]),
                    "collision": False,
                    "init_vx": init_vx,
                    "init_vy": init_vy,
                    "history": history,
                }
            if float(pos[0]) <= float(x0):
                return {
                    "exited": True,
                    "side": "left",
                    "coord": float(pos[1]),
                    "collision": False,
                    "init_vx": init_vx,
                    "init_vy": init_vy,
                    "history": history,
                }
            if float(pos[1]) >= float(y1):
                return {
                    "exited": True,
                    "side": "top",
                    "coord": float(pos[0]),
                    "collision": False,
                    "init_vx": init_vx,
                    "init_vy": init_vy,
                    "history": history,
                }
            if float(pos[1]) <= float(y0):
                return {
                    "exited": True,
                    "side": "bottom",
                    "coord": float(pos[0]),
                    "collision": False,
                    "init_vx": init_vx,
                    "init_vy": init_vy,
                    "history": history,
                }

    return {
        "exited": False,
        "side": None,
        "coord": None,
        "collision": False,
        "init_vx": init_vx,
        "init_vy": init_vy,
        "history": history,
    }


def _streamline_target_fit_metrics(streamline: np.ndarray, target_path: np.ndarray):
    if streamline.shape[0] < 2 or target_path.shape[0] < 2:
        return 1.0, 1.0

    d2 = np.sum((target_path[:, None, :] - streamline[None, :, :]) ** 2, axis=2)
    target_to_stream = float(np.mean(np.sqrt(np.min(d2, axis=1))))
    stream_to_target = float(np.mean(np.sqrt(np.min(d2, axis=0))))
    fit_error = 0.5 * (target_to_stream + stream_to_target)

    nearest_target_idx = np.argmin(d2, axis=0)
    max_idx = int(np.max(nearest_target_idx))
    coverage = float(max_idx) / float(max(1, target_path.shape[0] - 1))
    cover_penalty = 1.0 - coverage
    return float(fit_error), float(cover_penalty)


def logic_box_route_metrics(
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    omega: float,
    inflow_u: float,
    inflow_v: float,
    source_port: str,
    target_port: str,
    target_path=None,
    seed_offsets=None,
):
    ports = logic_box_ports()
    src = str(source_port).upper()
    tgt = str(target_port).upper()
    if src not in ports:
        src = "L1"
    if tgt not in ports:
        tgt = "R1"

    src_side, src_xy = ports[src]
    tgt_side, tgt_xy = ports[tgt]
    (x0, _), _ = logic_box_bounds()

    if src_side != "left":
        # This mode currently assumes left-side inlets.
        src = "L1"
        src_side, src_xy = ports[src]

    if seed_offsets is None:
        seed_offsets = np.asarray(getattr(LogicBoxConfig, "SOURCE_SEED_DY", [0.0]), dtype=np.float32)
    else:
        seed_offsets = np.asarray(seed_offsets, dtype=np.float32)
    if seed_offsets.size == 0:
        seed_offsets = np.array([0.0], dtype=np.float32)

    miss = 0
    collision = 0
    wrong_side = 0
    pos_err = []
    forward_pen = []
    inlet_block_pen = []
    path_fit_err = []
    path_cover_pen = []
    exits = []

    x_inset = max(1e-4, float(getattr(LogicBoxConfig, "SOURCE_SEED_X_INSET", 1e-4)))
    min_forward_vx = float(getattr(LogicBoxConfig, "MIN_FORWARD_VX", 1e-4))
    raw_reseed = np.asarray(
        getattr(LogicBoxConfig, "SOURCE_RESEED_X_OFFSETS", [0.0]),
        dtype=np.float32,
    ).reshape(-1)
    if raw_reseed.size == 0:
        raw_reseed = np.array([0.0], dtype=np.float32)
    if not np.any(np.abs(raw_reseed) < 1e-9):
        raw_reseed = np.concatenate([np.array([0.0], dtype=np.float32), raw_reseed], axis=0)
    reseed_offsets = []
    for vv in raw_reseed.tolist():
        v = max(0.0, float(vv))
        if not any(abs(v - u) < 1e-8 for u in reseed_offsets):
            reseed_offsets.append(v)

    for dy in seed_offsets:
        best = None
        for offset_x in reseed_offsets:
            seed_try = np.array(
                [float(x0) + max(1e-4, x_inset + float(offset_x)), float(src_xy[1] + dy)],
                dtype=np.float32,
            )
            tr_try = trace_streamline_until_box_exit(
                seed=seed_try,
                x=x,
                y=y,
                r=r,
                omega=float(omega),
                inflow_u=float(inflow_u),
                inflow_v=float(inflow_v),
            )
            hist_try = np.asarray(tr_try.get("history", []), dtype=np.float32)
            if hist_try.ndim == 2 and hist_try.shape[0] > 0 and hist_try.shape[1] >= 2:
                x_prog_try = float(np.max(hist_try[:, 0]) - float(seed_try[0]))
            else:
                x_prog_try = 0.0

            score = float(x_prog_try)
            if bool(tr_try.get("exited", False)):
                score += 2.0
            if bool(tr_try.get("exited", False)) and str(tr_try.get("side", "")).lower() == str(tgt_side).lower():
                score += 2.0
            if bool(tr_try.get("collision", False)):
                score -= 2.0
            if float(tr_try.get("init_vx", 0.0)) >= min_forward_vx:
                score += 0.5

            if (best is None) or (score > float(best["score"])):
                best = {
                    "score": float(score),
                    "seed": seed_try.copy(),
                    "offset_x": float(offset_x),
                    "trace": tr_try,
                    "x_progress": float(x_prog_try),
                }
            # Early stop on best-case outcome: exited from target side.
            if bool(tr_try.get("exited", False)) and str(tr_try.get("side", "")).lower() == str(tgt_side).lower():
                break

        if best is None:
            continue

        seed = np.asarray(best["seed"], dtype=np.float32)
        tr = dict(best["trace"])
        used_offset_x = float(best["offset_x"])
        x_progress = float(best["x_progress"])
        exits.append(
            {
                "seed": [float(seed[0]), float(seed[1])],
                "seed_inset_x": float((float(seed[0]) - float(x0))),
                "seed_extra_inset_x": float(used_offset_x),
                "side": tr["side"],
                "coord": tr["coord"],
                "exited": bool(tr["exited"]),
                "collision": bool(tr["collision"]),
                "init_vx": float(tr["init_vx"]),
                "init_vy": float(tr["init_vy"]),
                "stalled": bool((not tr["exited"]) and (not tr["collision"])),
                "x_progress": float(x_progress),
                "history": tr.get("history", []),
            }
        )
        # Source-neighborhood blockage: evaluate short segment from seed into the field.
        clear_target = max(float(getattr(LogicBoxConfig, "INLET_CLEARANCE", 0.012)), 1e-6)
        probe_dx = max(0.006, 0.75 * max(1e-4, x_inset + used_offset_x))
        probe_pts = np.stack(
            [
                np.linspace(float(seed[0]), float(seed[0] + probe_dx), num=4, dtype=np.float32),
                np.full(4, float(seed[1]), dtype=np.float32),
            ],
            axis=1,
        )
        if len(r) > 0:
            min_clear = 1e9
            for pxy in probe_pts:
                d = np.hypot(x - float(pxy[0]), y - float(pxy[1])) - r
                min_clear = min(min_clear, float(np.min(d)))
            inlet_pen = max(0.0, (clear_target - min_clear) / clear_target)
        else:
            inlet_pen = 0.0
        inlet_block_pen.append(float(inlet_pen))

        fwd = max(0.0, (float(LogicBoxConfig.MIN_FORWARD_VX) - float(tr["init_vx"])))
        forward_pen.append(float(fwd))
        if target_path is not None:
            streamline = np.asarray(tr.get("history", []), dtype=np.float32)
            fit_err, cover_pen = _streamline_target_fit_metrics(
                streamline=streamline,
                target_path=np.asarray(target_path, dtype=np.float32),
            )
            path_fit_err.append(float(fit_err))
            path_cover_pen.append(float(cover_pen))

        if tr["collision"]:
            collision += 1
            miss += 1
            pos_err.append(1.0)
            continue
        if not tr["exited"]:
            miss += 1
            pos_err.append(1.0)
            continue

        d, same_side = _logic_exit_match_error(
            side=str(tr["side"]),
            coord=float(tr["coord"]),
            target_side=str(tgt_side),
            target_xy=tgt_xy,
        )
        if not same_side:
            wrong_side += 1
            pos_err.append(float(d))
        else:
            pos_err.append(float(d))

    n = max(1, len(seed_offsets))
    return {
        "source_port": src,
        "target_port": tgt,
        "target_side": str(tgt_side),
        "target_xy": [float(tgt_xy[0]), float(tgt_xy[1])],
        "miss_ratio": float(miss / n),
        "collision_ratio": float(collision / n),
        "wrong_side_ratio": float(wrong_side / n),
        "outlet_pos_error": float(np.mean(pos_err)),
        "forward_penalty": float(np.mean(forward_pen)),
        "inlet_block_penalty": float(np.mean(inlet_block_pen)) if len(inlet_block_pen) > 0 else 0.0,
        "path_fit_error": float(np.mean(path_fit_err)) if len(path_fit_err) > 0 else 0.0,
        "path_cover_penalty": float(np.mean(path_cover_pen)) if len(path_cover_pen) > 0 else 0.0,
        "exits": exits,
    }


def logic_box_multi_route_metrics(
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    omega: float,
    inflow_u: float,
    inflow_v: float,
    route_pairs,
    target_paths=None,
    seed_offsets=None,
):
    pairs = list(route_pairs) if isinstance(route_pairs, (list, tuple)) else []
    if len(pairs) == 0:
        pairs = logic_box_active_route_pairs()

    route_details = []
    all_exits = []
    miss_list = []
    col_list = []
    wrong_list = []
    out_list = []
    fwd_list = []
    inlet_list = []
    fit_list = []
    cov_list = []

    for src, tgt in pairs:
        key = f"{str(src).upper()}->{str(tgt).upper()}"
        tgt_path = None
        if isinstance(target_paths, dict):
            tgt_path = target_paths.get(key, None)
        m = logic_box_route_metrics(
            x=x,
            y=y,
            r=r,
            omega=float(omega),
            inflow_u=float(inflow_u),
            inflow_v=float(inflow_v),
            source_port=str(src),
            target_port=str(tgt),
            target_path=tgt_path,
            seed_offsets=seed_offsets,
        )
        route_details.append(m)
        miss_list.append(float(m.get("miss_ratio", 0.0)))
        col_list.append(float(m.get("collision_ratio", 0.0)))
        wrong_list.append(float(m.get("wrong_side_ratio", 0.0)))
        out_list.append(float(m.get("outlet_pos_error", 0.0)))
        fwd_list.append(float(m.get("forward_penalty", 0.0)))
        inlet_list.append(float(m.get("inlet_block_penalty", 0.0)))
        fit_list.append(float(m.get("path_fit_error", 0.0)))
        cov_list.append(float(m.get("path_cover_penalty", 0.0)))

        for ex in m.get("exits", []):
            rec = dict(ex)
            rec["source_port"] = str(m.get("source_port", src))
            rec["target_port"] = str(m.get("target_port", tgt))
            rec["route"] = key
            all_exits.append(rec)

    n = max(1, len(route_details))
    return {
        "route_mode": "multi_map",
        "route_pairs": [
            [str(m.get("source_port", "")).upper(), str(m.get("target_port", "")).upper()]
            for m in route_details
        ],
        "source_port": "MULTI",
        "target_port": "MULTI",
        "target_side": "multi",
        "target_xy": [0.0, 0.0],
        "miss_ratio": float(np.mean(miss_list)) if len(miss_list) > 0 else 1.0,
        "collision_ratio": float(np.mean(col_list)) if len(col_list) > 0 else 0.0,
        "wrong_side_ratio": float(np.mean(wrong_list)) if len(wrong_list) > 0 else 1.0,
        "outlet_pos_error": float(np.mean(out_list)) if len(out_list) > 0 else 1.0,
        "forward_penalty": float(np.mean(fwd_list)) if len(fwd_list) > 0 else 0.0,
        "inlet_block_penalty": float(np.mean(inlet_list)) if len(inlet_list) > 0 else 0.0,
        "path_fit_error": float(np.mean(fit_list)) if len(fit_list) > 0 else 0.0,
        "path_cover_penalty": float(np.mean(cov_list)) if len(cov_list) > 0 else 0.0,
        "route_details": route_details,
        "exits": all_exits,
        "num_routes": int(n),
    }


def flow_path_alignment_metrics(
    path: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    omega: float,
    inflow_u: float = 0.0,
    inflow_v: float = 0.0,
    samples=24,
    lane_offsets=(0.0, 0.006, -0.006),
):
    """
    Compare local flow vectors with target-path tangents/normals
    in an orientation-agnostic way.
    Returns normalized penalties (lower is better).
    """
    n = int(path.shape[0])
    if n < 3:
        return 0.0, 0.0

    m = min(samples, n - 2)
    idx = np.linspace(1, n - 2, num=m, dtype=np.int32)
    pts_center = path[idx]
    tang = path[idx + 1] - path[idx - 1]
    t_norm = np.linalg.norm(tang, axis=1, keepdims=True)
    t_hat = tang / np.maximum(t_norm, 1e-8)
    n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)

    omegas = np.full(len(x), float(omega), dtype=np.float32)
    cos_vals = []
    normal_vals = []
    for off in lane_offsets:
        pts = pts_center + float(off) * n_hat
        v = np.zeros((m, 2), dtype=np.float32)
        for k, p in enumerate(pts):
            vx, vy = calculate_point_velocity(
                float(p[0]),
                float(p[1]),
                x,
                y,
                r,
                omegas,
            )
            v[k, 0] = vx + float(inflow_u)
            v[k, 1] = vy + float(inflow_v)
        speed = np.linalg.norm(v, axis=1)
        v_hat = v / np.maximum(speed[:, None], 1e-8)
        cos_sim = np.sum(v_hat * t_hat, axis=1)
        normal_proj = np.sum(v_hat * n_hat, axis=1)
        cos_vals.append(np.clip(cos_sim, -1.0, 1.0))
        normal_vals.append(np.abs(normal_proj))

    cos_all = np.concatenate(cos_vals, axis=0)
    normal_all = np.concatenate(normal_vals, axis=0)

    # 0 is perfect; >0 is misalignment.
    tangential_pen = float(np.mean(1.0 - np.abs(cos_all)))
    normal_pen = float(np.mean(normal_all))
    return tangential_pen, normal_pen


def flow_path_direction_metrics(
    path: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    omega: float,
    inflow_u: float = 0.0,
    inflow_v: float = 0.0,
    samples=20,
):
    """
    Signed direction matching along a path.
    0 is best for dir_pen; reverse_ratio in [0,1].
    """
    n = int(path.shape[0])
    if n < 3:
        return 0.0, 0.0

    m = min(samples, n - 2)
    idx = np.linspace(1, n - 2, num=m, dtype=np.int32)
    pts = path[idx]
    tang = path[idx + 1] - path[idx - 1]
    t_norm = np.linalg.norm(tang, axis=1, keepdims=True)
    t_hat = tang / np.maximum(t_norm, 1e-8)

    omegas = np.full(len(x), float(omega), dtype=np.float32)
    v = np.zeros((m, 2), dtype=np.float32)
    for k, p in enumerate(pts):
        vx, vy = calculate_point_velocity(
            float(p[0]),
            float(p[1]),
            x,
            y,
            r,
            omegas,
        )
        v[k, 0] = vx + float(inflow_u)
        v[k, 1] = vy + float(inflow_v)

    speed = np.linalg.norm(v, axis=1)
    v_hat = v / np.maximum(speed[:, None], 1e-8)
    cos_sim = np.clip(np.sum(v_hat * t_hat, axis=1), -1.0, 1.0)
    dir_pen = float(np.mean(1.0 - cos_sim))
    reverse_ratio = float(np.mean((cos_sim < 0.0).astype(np.float32)))
    return dir_pen, reverse_ratio


class TaskContext:
    def __init__(self):
        self.path = []
        self.current_step = 0
        self.particle_pos = np.array([0.0, 0.0], dtype=np.float32)

    def generate_fixed_path(self, path_type="bezier", source_port=None, target_port=None):
        steps = TrainingSettingConfig.EPISODE_LENGTH + 1
        s = np.linspace(0.0, 1.0, steps)
        t = np.linspace(0.0, 2.0 * np.pi, steps)

        gx, gy = FixedGrid3x3Config.grid_coords()
        c_center = np.array([gx[4], gy[4]], dtype=np.float32)
        c1 = np.array([gx[1], gy[1]], dtype=np.float32)
        c5 = np.array([gx[5], gy[5]], dtype=np.float32)

        if path_type == "soft_snake2":
            x0, x1 = -0.025, 0.055
            y0, y1 = 0.060, -0.020
            xs = x0 + (x1 - x0) * s
            ys = y0 + (y1 - y0) * s - 0.010 * np.sin(2.0 * np.pi * s)

        elif path_type == "soft_snake2_easy":
            x0, x1 = -0.025, 0.050
            y0, y1 = 0.060, -0.015
            xs = x0 + (x1 - x0) * s
            ys = y0 + (y1 - y0) * s - 0.006 * np.sin(2.0 * np.pi * s)

        elif path_type == "bezier":
            p0 = np.array([-0.025, 0.060], dtype=np.float32)
            p1 = np.array([0.000, 0.050], dtype=np.float32)
            p2 = np.array([0.055, -0.010], dtype=np.float32)
            xs = (1 - s) ** 2 * p0[0] + 2 * (1 - s) * s * p1[0] + s ** 2 * p2[0]
            ys = (1 - s) ** 2 * p0[1] + 2 * (1 - s) * s * p1[1] + s ** 2 * p2[1]

        elif path_type == "bend2":
            x0, x1 = -0.025, -0.070
            y0, y1 = 0.060, -0.010
            bend = 0.080
            xs = x0 + (x1 - x0) * s
            ys = y0 + (y1 - y0) * s + bend * s * (1.0 - s) * np.sin(2.0 * np.pi * s)

        elif path_type == "bend":
            x0, x1 = -0.025, -0.070
            y0, y1 = 0.060, -0.010
            xs = x0 + (x1 - x0) * s
            ys = y0 + (y1 - y0) * s + 0.060 * s * (1.0 - s)

        elif path_type == "orbit_center":
            rad = 0.045
            xs = c_center[0] + rad * np.cos(t)
            ys = c_center[1] + rad * np.sin(t)

        elif path_type == "orbit_cyl_1":
            rad = 0.035
            xs = c1[0] + rad * np.cos(t)
            ys = c1[1] + rad * np.sin(t)

        elif path_type == "orbit_pair_1_5":
            # Figure-eight around cylinders #1 and #5.
            mid = 0.5 * (c1 + c5)
            ax = 0.060
            ay = 0.040
            xs = mid[0] + ax * np.sin(t)
            ys = mid[1] + ay * np.sin(t) * np.cos(t)
        elif path_type == "square":
            # Use a rounded-square (superellipse) target so the streamline
            # matching objective remains physically realizable.
            center = np.array([0.0, 0.0], dtype=np.float32)
            half = 0.055
            n_exp = 4.0
            tt = np.linspace(0.0, 2.0 * np.pi, steps)
            c = np.cos(tt)
            s2 = np.sin(tt)
            xs = center[0] + half * np.sign(c) * np.abs(c) ** (2.0 / n_exp)
            ys = center[1] + half * np.sign(s2) * np.abs(s2) ** (2.0 / n_exp)
        elif path_type == "square_hard":
            center = np.array([0.0, 0.0], dtype=np.float32)
            half = 0.05
            p0 = center + np.array([-half, +half], dtype=np.float32)
            p1 = center + np.array([+half, +half], dtype=np.float32)
            p2 = center + np.array([+half, -half], dtype=np.float32)
            p3 = center + np.array([-half, -half], dtype=np.float32)
            u = np.linspace(0.0, 4.0, steps)
            xs = np.zeros(steps, dtype=np.float32)
            ys = np.zeros(steps, dtype=np.float32)
            for i, ui in enumerate(u):
                if ui < 1.0:
                    w = ui
                    p = (1.0 - w) * p0 + w * p1
                elif ui < 2.0:
                    w = ui - 1.0
                    p = (1.0 - w) * p1 + w * p2
                elif ui < 3.0:
                    w = ui - 2.0
                    p = (1.0 - w) * p2 + w * p3
                else:
                    w = min(ui - 3.0, 1.0)
                    p = (1.0 - w) * p3 + w * p0
                xs[i] = float(p[0])
                ys[i] = float(p[1])
        elif path_type in {"gate3_top", "gate3_mid", "gate3_bottom"}:
            x0 = float(Gate3LevelConfig.START_X)
            xg = float(np.mean(Gate3LevelConfig.FIXED_X))
            x1 = float(Gate3LevelConfig.CHECK_X + 0.06)
            lane_map = {
                "gate3_top": float(Gate3LevelConfig.LANE_TARGET_Y[0]),
                "gate3_mid": float(Gate3LevelConfig.LANE_TARGET_Y[1]),
                "gate3_bottom": float(Gate3LevelConfig.LANE_TARGET_Y[2]),
            }
            yg = lane_map[path_type]
            y1 = yg
            y0 = float(Gate3LevelConfig.START_Y)

            def cubic_bezier(p0, p1, p2, p3, u):
                return (
                    ((1.0 - u) ** 3)[:, None] * p0[None, :]
                    + (3.0 * ((1.0 - u) ** 2) * u)[:, None] * p1[None, :]
                    + (3.0 * (1.0 - u) * (u ** 2))[:, None] * p2[None, :]
                    + (u ** 3)[:, None] * p3[None, :]
                )

            n1 = max(3, steps // 2 + 1)
            n2 = max(3, steps - n1 + 1)

            p0 = np.array([x0, y0], dtype=np.float32)
            p1 = np.array(
                [x0 + 0.35 * (xg - x0), y0 + 0.15 * (yg - y0)],
                dtype=np.float32,
            )
            p2 = np.array(
                [x0 + 0.80 * (xg - x0), yg],
                dtype=np.float32,
            )
            p3 = np.array([xg, yg], dtype=np.float32)

            q0 = p3
            q1 = np.array([xg + 0.25 * (x1 - xg), yg], dtype=np.float32)
            q2 = np.array([xg + 0.75 * (x1 - xg), y1], dtype=np.float32)
            q3 = np.array([x1, y1], dtype=np.float32)

            seg1 = cubic_bezier(p0, p1, p2, p3, np.linspace(0.0, 1.0, n1))
            seg2 = cubic_bezier(q0, q1, q2, q3, np.linspace(0.0, 1.0, n2))
            path = np.concatenate([seg1[:-1], seg2], axis=0)
            xs = path[:, 0]
            ys = path[:, 1]
        elif path_type in {"logic_route", "logic_multi_route"}:
            if _logic_is_multi_route_mode():
                pairs = logic_box_active_route_pairs()
                # Preview path for visualization: prefer center lane source (L1) if present.
                preview_pair = pairs[0]
                for src_name, tgt_name in pairs:
                    if str(src_name).upper() == "L1":
                        preview_pair = (src_name, tgt_name)
                        break
                path = build_logic_route_path(
                    source_port=str(preview_pair[0]),
                    target_port=str(preview_pair[1]),
                    steps=steps,
                )
            else:
                src_name = (
                    str(source_port).upper()
                    if source_port is not None
                    else str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")).upper()
                )
                tgt_name = (
                    str(target_port).upper()
                    if target_port is not None
                    else str(getattr(LogicBoxConfig, "TARGET_PORT", "R1")).upper()
                )
                path = build_logic_route_path(
                    source_port=src_name,
                    target_port=tgt_name,
                    steps=steps,
                )
            xs = path[:, 0]
            ys = path[:, 1]

        else:
            x0, x1 = -0.025, 0.050
            y0, y1 = 0.060, -0.010
            xs = x0 + (x1 - x0) * s
            ys = y0 + (y1 - y0) * s

        self.path = np.stack([xs, ys], axis=1).astype(np.float32)

    @property
    def goal(self) -> np.ndarray:
        idx = min(self.current_step, len(self.path) - 1)
        return self.path[idx]


class FollowerEnv(gym.Env):
    def __init__(self, fluid_env: FluidEnv, ctx: TaskContext, logic_seed_profile: str = "train"):
        super().__init__()
        self.env = fluid_env
        self.ctx = ctx
        self.layout_mode = self.env.layout_mode
        self.base_layout_mode = self.env.base_layout_mode
        self.logic_seed_profile = str(logic_seed_profile).strip().lower()
        self.logic_episode_source_port = str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")).upper()
        self.logic_episode_target_port = str(getattr(LogicBoxConfig, "TARGET_PORT", "R1")).upper()
        self.logic_target_candidates = list(logic_target_port_set())
        self.logic_forced_target_port = None

        if self.base_layout_mode == "fixed_grid_3x3":
            action_dim = self.env.max_n
        elif self.base_layout_mode == "gate3_layout":
            action_dim = self.env.design_n * 3
        elif self.base_layout_mode == "logic_box_layout":
            keep_xy_when_fixed = bool(
                getattr(LogicBoxConfig, "KEEP_XY_ACTION_WHEN_FIXED", False)
            )
            if bool(getattr(self.env, "logic_fixed_layout", False)) and (not keep_xy_when_fixed):
                action_dim = self.env.max_n
            else:
                action_dim = self.env.max_n * 3
        else:
            action_dim = self.env.max_n * 3
        action_dim += self.env.inflow_action_dim
        action_dim += self.env.omega_action_dim

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        sample_state = self.env.get_state()
        obs_dim = sample_state.shape[0] + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.episode_return = 0.0
        self.prev_dist = None
        self.last_history = []

    def set_inflow_control_enabled(self, enabled: bool):
        if hasattr(self.env, "set_inflow_control_enabled"):
            self.env.set_inflow_control_enabled(bool(enabled))
        else:
            self.env.inflow_control_enabled = bool(enabled)

    def set_logic_seed_profile(self, profile: str):
        self.logic_seed_profile = str(profile).strip().lower()

    def set_logic_forced_target_port(self, target_port=None):
        if target_port is None:
            self.logic_forced_target_port = None
            return
        self.logic_forced_target_port = str(target_port).strip().upper()

    def set_logic_fixed_layout(self, enabled: bool, fixed_x=None, fixed_y=None):
        if hasattr(self.env, "set_logic_fixed_layout"):
            self.env.set_logic_fixed_layout(bool(enabled), fixed_x=fixed_x, fixed_y=fixed_y)

    def get_logic_fixed_layout(self):
        return bool(getattr(self.env, "logic_fixed_layout", False))

    def _logic_seed_offsets(self) -> np.ndarray:
        return logic_seed_offsets(self.logic_seed_profile)

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        r_low, r_high = StokesCylinderConfig.MIN_R, StokesCylinderConfig.MAX_R
        a_full = np.asarray(action, dtype=np.float32).reshape(-1)

        inflow_tail = []
        omega_tail = []
        tail_dim = self.env.inflow_action_dim + self.env.omega_action_dim
        if tail_dim > 0:
            if a_full.size <= tail_dim:
                raise ValueError(
                    f"Action dim too small for layout+tail decode: {a_full.size}"
                )
            a_layout = a_full[:-tail_dim]
            a_tail = a_full[-tail_dim:]
            k = 0
            if self.env.inflow_action_dim > 0:
                if self.env.optimize_inflow_u:
                    u = InflowConfig.U_MIN + 0.5 * (a_tail[k] + 1.0) * (
                        InflowConfig.U_MAX - InflowConfig.U_MIN
                    )
                    inflow_tail.append(float(u))
                    k += 1
                if self.env.optimize_inflow_v:
                    v = InflowConfig.V_MIN + 0.5 * (a_tail[k] + 1.0) * (
                        InflowConfig.V_MAX - InflowConfig.V_MIN
                    )
                    inflow_tail.append(float(v))
                    k += 1
            if self.env.omega_action_dim > 0:
                mode = str(GlobalOmegaControlConfig.MODE).lower()
                if mode == "continuous":
                    om = GlobalOmegaControlConfig.OMEGA_MIN + 0.5 * (a_tail[k] + 1.0) * (
                        GlobalOmegaControlConfig.OMEGA_MAX - GlobalOmegaControlConfig.OMEGA_MIN
                    )
                    omega_tail.append(float(om))
                else:
                    omega_abs = max(abs(StokesCylinderConfig.FIXED_OMEGA), 1e-8)
                    omega_tail.append(float(omega_abs if a_tail[k] >= 0.0 else -omega_abs))
        else:
            a_layout = a_full

        if self.base_layout_mode == "fixed_grid_3x3":
            a = a_layout.reshape(self.env.max_n).copy()
            a = r_low + 0.5 * (a + 1.0) * (r_high - r_low)
            a = np.where(a < StokesCylinderConfig.EXIST_THRESHOLD, 0.0, a)
            if len(inflow_tail) > 0 or len(omega_tail) > 0:
                return np.concatenate(
                    [
                        a.astype(np.float32),
                        np.asarray(inflow_tail, dtype=np.float32),
                        np.asarray(omega_tail, dtype=np.float32),
                    ],
                    axis=0,
                )
            return a.astype(np.float32)

        if self.base_layout_mode == "gate3_layout":
            a = a_layout.reshape(self.env.design_n, 3).copy()
            x_low, x_high = StokesCylinderConfig.X_RANGE
            y_low, y_high = StokesCylinderConfig.Y_RANGE
            a[:, 0] = x_low + 0.5 * (a[:, 0] + 1.0) * (x_high - x_low)
            a[:, 1] = y_low + 0.5 * (a[:, 1] + 1.0) * (y_high - y_low)
            a[:, 2] = r_low + 0.5 * (a[:, 2] + 1.0) * (r_high - r_low)
            a[:, 2] = np.where(
                a[:, 2] < StokesCylinderConfig.EXIST_THRESHOLD,
                0.0,
                a[:, 2],
            )
            layout_vec = a.reshape(-1).astype(np.float32)
            if len(inflow_tail) > 0 or len(omega_tail) > 0:
                return np.concatenate(
                    [
                        layout_vec,
                        np.asarray(inflow_tail, dtype=np.float32),
                        np.asarray(omega_tail, dtype=np.float32),
                    ],
                    axis=0,
                )
            return layout_vec

        if self.base_layout_mode == "logic_box_layout":
            (x_low, x_high), (y_low, y_high) = logic_box_bounds()
            logic_r_high = min(float(r_high), float(get_logic_max_radius()))
            logic_exist_th = float(
                getattr(LogicBoxConfig, "EXIST_THRESHOLD", StokesCylinderConfig.EXIST_THRESHOLD)
            )
            logic_forbid_elimination = bool(getattr(LogicBoxConfig, "FORBID_ELIMINATION", False))
            logic_min_active_r = max(float(r_low), float(getattr(LogicBoxConfig, "MIN_ACTIVE_R", r_low)))
            if bool(getattr(self.env, "logic_fixed_layout", False)):
                # Compatibility:
                # - legacy fixed-layout policies output N radii
                # - stage-switch compatible policies may still output N*3 (x/y/r); use only r
                if int(a_layout.size) == int(self.env.max_n):
                    raw_r = a_layout.reshape(self.env.max_n).copy()
                elif int(a_layout.size) == int(self.env.max_n * 3):
                    raw_r = a_layout.reshape(self.env.max_n, 3)[:, 2].copy()
                else:
                    raise ValueError(
                        f"LogicBox fixed layout expects {self.env.max_n} or {self.env.max_n * 3} dims, got {a_layout.size}"
                    )
                raw_r = r_low + 0.5 * (raw_r + 1.0) * (logic_r_high - r_low)
                if logic_forbid_elimination:
                    raw_r = np.maximum(raw_r, logic_min_active_r)
                else:
                    raw_r = np.where(raw_r < logic_exist_th, 0.0, raw_r)
                layout_vec = raw_r.astype(np.float32)
            else:
                a = a_layout.reshape(self.env.max_n, 3).copy()
                a[:, 0] = x_low + 0.5 * (a[:, 0] + 1.0) * (x_high - x_low)
                a[:, 1] = y_low + 0.5 * (a[:, 1] + 1.0) * (y_high - y_low)
                a[:, 2] = r_low + 0.5 * (a[:, 2] + 1.0) * (logic_r_high - r_low)
                if logic_forbid_elimination:
                    a[:, 2] = np.maximum(a[:, 2], logic_min_active_r)
                else:
                    a[:, 2] = np.where(a[:, 2] < logic_exist_th, 0.0, a[:, 2])
                layout_vec = a.reshape(-1).astype(np.float32)
            if len(inflow_tail) > 0 or len(omega_tail) > 0:
                return np.concatenate(
                    [
                        layout_vec,
                        np.asarray(inflow_tail, dtype=np.float32),
                        np.asarray(omega_tail, dtype=np.float32),
                    ],
                    axis=0,
                )
            return layout_vec

        a = a_layout.reshape(self.env.max_n, 3).copy()
        x_low, x_high = StokesCylinderConfig.X_RANGE
        y_low, y_high = StokesCylinderConfig.Y_RANGE
        a[:, 0] = x_low + 0.5 * (a[:, 0] + 1.0) * (x_high - x_low)
        a[:, 1] = y_low + 0.5 * (a[:, 1] + 1.0) * (y_high - y_low)
        a[:, 2] = r_low + 0.5 * (a[:, 2] + 1.0) * (r_high - r_low)
        a[:, 2] = np.where(a[:, 2] < StokesCylinderConfig.EXIST_THRESHOLD, 0.0, a[:, 2])
        layout_vec = a.reshape(-1).astype(np.float32)
        if len(inflow_tail) > 0 or len(omega_tail) > 0:
            return np.concatenate(
                [
                    layout_vec,
                    np.asarray(inflow_tail, dtype=np.float32),
                    np.asarray(omega_tail, dtype=np.float32),
                ],
                axis=0,
            )
        return layout_vec

    def _get_obs(self) -> np.ndarray:
        raw_state = self.env.get_state()
        dxdy = np.array(
            [
                self.ctx.goal[0] - self.env.particle.pos_x,
                self.ctx.goal[1] - self.env.particle.pos_y,
            ],
            dtype=np.float32,
        )
        if self.base_layout_mode == "logic_box_layout":
            ports = logic_box_ports()
            tgt = str(self.logic_episode_target_port).upper()
            if tgt in ports:
                target_xy = ports[tgt][1]
                dxdy = np.array(
                    [
                        float(target_xy[0]) - float(self.env.particle.pos_x),
                        float(target_xy[1]) - float(self.env.particle.pos_y),
                    ],
                    dtype=np.float32,
                )
        return np.concatenate([raw_state, dxdy]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ctx.current_step = 0
        path_type = TrainingSettingConfig.PATH_TYPE
        if self.base_layout_mode == "gate3_layout" and not str(path_type).startswith("gate3_"):
            path_type = "gate3_mid"
        if self.base_layout_mode == "logic_box_layout" and not str(path_type).startswith("logic_"):
            path_type = "logic_route"
        if self.base_layout_mode == "logic_box_layout":
            ports = logic_box_ports()
            src = str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")).upper()
            if src not in ports or ports[src][0] != "left":
                src = "L1"
            self.logic_target_candidates = list(logic_target_port_set())
            forced_tgt = str(self.logic_forced_target_port).upper() if self.logic_forced_target_port else None
            if (
                _logic_is_single_multi_target_mode()
                and forced_tgt is not None
                and forced_tgt in self.logic_target_candidates
            ):
                tgt = forced_tgt
            elif _logic_is_single_multi_target_mode() and len(self.logic_target_candidates) > 0:
                k = int(self.np_random.integers(0, len(self.logic_target_candidates)))
                tgt = str(self.logic_target_candidates[k]).upper()
            else:
                tgt = str(getattr(LogicBoxConfig, "TARGET_PORT", "R1")).upper()
                if tgt not in ports:
                    tgt = "R1"
            self.logic_episode_source_port = src
            self.logic_episode_target_port = tgt
            if _logic_is_multi_route_mode():
                self.ctx.generate_fixed_path(path_type="logic_multi_route")
            else:
                self.ctx.generate_fixed_path(
                    path_type="logic_route",
                    source_port=src,
                    target_port=tgt,
                )
            # One-shot logic tasks should expose endpoint direction in observation.
            self.ctx.current_step = max(0, len(self.ctx.path) - 1)
        else:
            self.ctx.generate_fixed_path(path_type=path_type)
        ini_pos = np.array(self.ctx.path[0], dtype=np.float32)
        self.env.reset(ini_pos)
        self.ctx.particle_pos = ini_pos.copy()

        self.episode_return = 0.0
        self.last_history = [tuple(ini_pos)]

        dx = self.ctx.goal[0] - self.ctx.particle_pos[0]
        dy = self.ctx.goal[1] - self.ctx.particle_pos[1]
        self.prev_dist = float(np.hypot(dx, dy))

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # One-shot static optimization: reward is computed directly from
        # layout-induced flow quality w.r.t. target trajectory.
        physical_action = self._decode_action(action)
        self.env.apply_layout(physical_action)

        x = self.env.cylinders_x
        y = self.env.cylinders_y
        r = self.env.cylinders_r
        path_np = np.asarray(self.ctx.path, dtype=np.float32)

        gate_lane_error = 0.0
        gate_lane_miss = 0.0
        gate_lane_collision = 0.0
        gate_lane_order = 0.0
        gate_lane_cross = []
        gate_lane_target = []
        gate_dir_pen = 0.0
        gate_reverse_ratio = 0.0
        logic_miss = 0.0
        logic_collision = 0.0
        logic_wrong_side = 0.0
        logic_outlet_error = 0.0
        logic_forward_pen = 0.0
        logic_inlet_block_pen = 0.0
        logic_path_fit = 0.0
        logic_path_cover = 0.0
        logic_source_port = ""
        logic_target_port = ""
        logic_target_side = ""
        logic_target_xy = [0.0, 0.0]
        logic_route_mode = "single"
        logic_route_pairs = []
        logic_route_details = []
        logic_exits = []
        block_pen = 0.0
        flow_tangent_pen = 0.0
        flow_normal_pen = 0.0

        if self.base_layout_mode == "fixed_grid_3x3":
            w_layout = 6.0
            w_block = 80.0
            w_flow_tangent = 24.0
            w_flow_normal = 16.0
            cluster_scale = 0.0
        elif self.base_layout_mode == "gate3_layout":
            w_layout = Gate3LevelConfig.W_LAYOUT
            w_block = 0.0
            w_flow_tangent = 0.0
            w_flow_normal = 0.0
            cluster_scale = Gate3LevelConfig.W_CLUSTER
        elif self.base_layout_mode == "logic_box_layout":
            w_layout = LogicBoxConfig.W_LAYOUT
            w_block = 0.0
            w_flow_tangent = 0.0
            w_flow_normal = 0.0
            cluster_scale = LogicBoxConfig.W_CLUSTER
        else:
            w_layout = 8.0
            w_block = 90.0
            w_flow_tangent = 30.0
            w_flow_normal = 18.0
            cluster_scale = 0.005

        if self.base_layout_mode == "logic_box_layout":
            xlim, ylim = logic_box_bounds()
            overlap_margin = float(getattr(LogicBoxConfig, "OVERLAP_MARGIN", 0.001))
        else:
            xlim = StokesCylinderConfig.X_RANGE
            ylim = StokesCylinderConfig.Y_RANGE
            overlap_margin = 0.001
        overlap_pen = layout_overlap_penalty(x, y, r, margin=overlap_margin)
        bound_pen = layout_boundary_penalty(
            x,
            y,
            r,
            xlim=xlim,
            ylim=ylim,
            margin=0.001,
        )
        layout_penalty = w_layout * (overlap_pen + bound_pen)

        active_count = int(np.sum(r > 0))
        total_count = int(r.size)
        if self.base_layout_mode == "logic_box_layout":
            killed_eps = float(getattr(LogicBoxConfig, "KILLED_EPS", 1e-6))
        else:
            killed_eps = 1e-12
        inactive_count = int(np.sum(r <= float(killed_eps)))
        zero_radius_ratio = float(inactive_count) / float(max(1, total_count))
        empty_layout_pen = 0.0
        if self.base_layout_mode == "logic_box_layout" and active_count <= 0:
            empty_layout_pen = 1.0
        sparse_penalty = 0.0

        cluster_penalty = 0.0
        if cluster_scale > 0.0:
            if self.base_layout_mode == "gate3_layout":
                active_local = np.where(r[self.env.fixed_count :] > 0)[0]
                active_idx = active_local + self.env.fixed_count
            else:
                active_idx = np.where(r > 0)[0]
            for ii in range(len(active_idx)):
                for jj in range(ii + 1, len(active_idx)):
                    i = active_idx[ii]
                    j = active_idx[jj]
                    dij = np.hypot(x[i] - x[j], y[i] - y[j])
                    cluster_penalty += 1.0 / (dij + 1e-3)
            cluster_penalty *= cluster_scale

        if self.base_layout_mode == "gate3_layout":
            lane_mode = str(getattr(Gate3LevelConfig, "LANE_REWARD_MODE", "single")).lower()
            path_type = str(TrainingSettingConfig.PATH_TYPE)
            lane_idx_map = {"gate3_top": 0, "gate3_mid": 1, "gate3_bottom": 2}
            if lane_mode == "all":
                lane_indices = [0, 1, 2]
            else:
                lane_indices = [lane_idx_map.get(path_type, 1)]
            gate_metrics = gate3_lane_passage_metrics(
                x=x,
                y=y,
                r=r,
                omega=self.env.fixed_omega,
                inflow_u=self.env.inflow_u,
                inflow_v=self.env.inflow_v,
                lane_indices=lane_indices,
            )
            gate_lane_error = float(gate_metrics["lane_error_mean"])
            gate_lane_miss = float(gate_metrics["lane_miss_ratio"])
            gate_lane_collision = float(gate_metrics["lane_collision_ratio"])
            gate_lane_order = float(gate_metrics["lane_order_penalty"])
            gate_lane_cross = list(gate_metrics["lane_y_cross"])
            gate_lane_target = list(gate_metrics["lane_target_y"])
            if (
                float(Gate3LevelConfig.W_DIR_SIGN) > 0.0
                or float(Gate3LevelConfig.W_DIR_REVERSE) > 0.0
            ):
                gate_dir_pen, gate_reverse_ratio = flow_path_direction_metrics(
                    path=path_np,
                    x=x,
                    y=y,
                    r=r,
                    omega=self.env.fixed_omega,
                    inflow_u=self.env.inflow_u,
                    inflow_v=self.env.inflow_v,
                    samples=20,
                )
        elif self.base_layout_mode == "logic_box_layout":
            multi_route = _logic_is_multi_route_mode()
            single_multi_target = _logic_is_single_multi_target_mode()
            if multi_route:
                logic_route_mode = "multi_map"
            elif single_multi_target:
                logic_route_mode = "single_multi_target"
            else:
                logic_route_mode = "single"
            seed_offsets = self._logic_seed_offsets()
            if multi_route:
                route_pairs = logic_box_active_route_pairs()
                target_paths = {}
                for src_port, tgt_port in route_pairs:
                    k = f"{str(src_port).upper()}->{str(tgt_port).upper()}"
                    target_paths[k] = build_logic_route_path(
                        source_port=str(src_port),
                        target_port=str(tgt_port),
                        steps=path_np.shape[0],
                    )
                logic_metrics = logic_box_multi_route_metrics(
                    x=x,
                    y=y,
                    r=r,
                    omega=self.env.fixed_omega,
                    inflow_u=self.env.inflow_u,
                    inflow_v=self.env.inflow_v,
                    route_pairs=route_pairs,
                    target_paths=target_paths,
                    seed_offsets=seed_offsets,
                )
            elif single_multi_target:
                logic_metrics = logic_box_route_metrics(
                    x=x,
                    y=y,
                    r=r,
                    omega=self.env.fixed_omega,
                    inflow_u=self.env.inflow_u,
                    inflow_v=self.env.inflow_v,
                    source_port=str(self.logic_episode_source_port),
                    target_port=str(self.logic_episode_target_port),
                    target_path=path_np,
                    seed_offsets=seed_offsets,
                )
            else:
                logic_metrics = logic_box_route_metrics(
                    x=x,
                    y=y,
                    r=r,
                    omega=self.env.fixed_omega,
                    inflow_u=self.env.inflow_u,
                    inflow_v=self.env.inflow_v,
                    source_port=str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")),
                    target_port=str(getattr(LogicBoxConfig, "TARGET_PORT", "R1")),
                    target_path=path_np,
                    seed_offsets=seed_offsets,
                )
            logic_miss = float(logic_metrics["miss_ratio"])
            logic_collision = float(logic_metrics["collision_ratio"])
            logic_wrong_side = float(logic_metrics["wrong_side_ratio"])
            logic_outlet_error = float(logic_metrics["outlet_pos_error"])
            logic_forward_pen = float(logic_metrics["forward_penalty"])
            logic_inlet_block_pen = float(logic_metrics.get("inlet_block_penalty", 0.0))
            logic_path_fit = float(logic_metrics.get("path_fit_error", 0.0))
            logic_path_cover = float(logic_metrics.get("path_cover_penalty", 0.0))
            logic_source_port = str(logic_metrics["source_port"])
            logic_target_port = str(logic_metrics["target_port"])
            logic_target_side = str(logic_metrics["target_side"])
            logic_target_xy = list(logic_metrics["target_xy"])
            logic_route_pairs = list(logic_metrics.get("route_pairs", []))
            if single_multi_target:
                logic_route_pairs = [
                    [str(self.logic_episode_source_port).upper(), str(t).upper()]
                    for t in self.logic_target_candidates
                ]
            logic_route_details = list(logic_metrics.get("route_details", []))
            logic_exits = list(logic_metrics["exits"])
        else:
            block_pen = path_blocking_penalty(path_np, x, y, r, clearance=0.006)
            flow_tangent_pen, flow_normal_pen = flow_path_alignment_metrics(
                path=path_np,
                x=x,
                y=y,
                r=r,
                omega=self.env.fixed_omega,
                inflow_u=self.env.inflow_u,
                inflow_v=self.env.inflow_v,
                samples=48,
            )

        inflow_reg_pen = 0.0
        if self.env.inflow_action_dim > 0:
            deadzone = float(getattr(InflowConfig, "REG_DEADZONE", 0.0))
            if self.env.optimize_inflow_u:
                u_scale = max(abs(InflowConfig.U_MIN), abs(InflowConfig.U_MAX), 1e-8)
                du_phys = abs(float(self.env.inflow_u) - float(InflowConfig.TARGET_U)) - deadzone
                du = max(0.0, du_phys) / u_scale
                inflow_reg_pen += du * du
            if self.env.optimize_inflow_v:
                v_scale = max(abs(InflowConfig.V_MIN), abs(InflowConfig.V_MAX), 1e-8)
                dv_phys = abs(float(self.env.inflow_v) - float(InflowConfig.TARGET_V)) - deadzone
                dv = max(0.0, dv_phys) / v_scale
                inflow_reg_pen += dv * dv

        total_reward = 0.0
        total_reward -= layout_penalty
        total_reward -= cluster_penalty
        if self.base_layout_mode == "logic_box_layout":
            total_reward -= float(getattr(LogicBoxConfig, "W_EMPTY_LAYOUT", 0.0)) * float(
                empty_layout_pen
            )
        if self.base_layout_mode == "gate3_layout":
            total_reward -= Gate3LevelConfig.W_LANE_ERR * gate_lane_error
            total_reward -= Gate3LevelConfig.W_LANE_MISS * gate_lane_miss
            total_reward -= Gate3LevelConfig.W_LANE_COLLISION * gate_lane_collision
            if len(gate_lane_target) == 3:
                total_reward -= Gate3LevelConfig.W_ORDER * gate_lane_order
            total_reward -= Gate3LevelConfig.W_DIR_SIGN * gate_dir_pen
            total_reward -= Gate3LevelConfig.W_DIR_REVERSE * gate_reverse_ratio
        elif self.base_layout_mode == "logic_box_layout":
            mode = str(getattr(LogicBoxConfig, "REWARD_MODE", "hybrid")).strip().lower()
            use_port_terms = mode in {"port_only", "port", "hybrid", "both"}
            use_stream_terms = mode in {"streamline_only", "flow_only", "streamline", "hybrid", "both"}

            if use_port_terms:
                total_reward -= LogicBoxConfig.W_MISS * logic_miss
                total_reward -= LogicBoxConfig.W_COLLISION * logic_collision
                total_reward -= LogicBoxConfig.W_WRONG_SIDE * logic_wrong_side
                total_reward -= LogicBoxConfig.W_OUTLET_POS * logic_outlet_error
                total_reward -= LogicBoxConfig.W_FORWARD * logic_forward_pen
                total_reward -= float(getattr(LogicBoxConfig, "W_INLET_BLOCK", 0.0)) * logic_inlet_block_pen
            if use_stream_terms:
                total_reward -= LogicBoxConfig.W_PATH_FIT * logic_path_fit
                total_reward -= LogicBoxConfig.W_PATH_COVER * logic_path_cover
        else:
            total_reward -= w_block * block_pen
            total_reward -= w_flow_tangent * flow_tangent_pen
            total_reward -= w_flow_normal * flow_normal_pen
        total_reward -= float(InflowConfig.REG_WEIGHT) * float(inflow_reg_pen)

        # Optional rollout only for visualization/diagnostics.
        full_history = [tuple(self.ctx.particle_pos.copy())]
        rollout_terminated = False
        if bool(getattr(TrainingSettingConfig, "RUN_DIAGNOSTIC_ROLLOUT", False)):
            _, rollout_terminated, extra_info = self.env.step(action=None)
            step_history = extra_info.get("history_pos", [])
            if len(step_history) > 0:
                full_history.extend(step_history)
            cur_pos = np.array(
                [self.env.particle.pos_x, self.env.particle.pos_y], dtype=np.float32
            )
            self.ctx.particle_pos = cur_pos.copy()
        else:
            cur_pos = self.ctx.particle_pos.copy()

        final_goal = np.array(self.ctx.path[-1], dtype=np.float32)
        final_dist = float(np.linalg.norm(cur_pos - final_goal))
        if self.base_layout_mode == "logic_box_layout":
            # In logic-box mode, training objective is streamline-route matching,
            # not particle rollout distance.
            mean_path_fit_dist = float(logic_path_fit)
            target_to_traj_mean = float(logic_path_fit)
            traj_to_target_mean = float(logic_path_fit)
            path_coverage = float(np.clip(1.0 - float(logic_path_cover), 0.0, 1.0))
            nearest_path_idx = int(round(path_coverage * float(max(1, len(path_np) - 1))))
            final_dist = float(logic_path_fit)
        else:
            traj_np = np.asarray(full_history, dtype=np.float32)
            d2 = np.sum((path_np[:, None, :] - traj_np[None, :, :]) ** 2, axis=2)
            target_to_traj_mean = float(np.mean(np.sqrt(np.min(d2, axis=1))))
            traj_to_target_mean = float(np.mean(np.sqrt(np.min(d2, axis=0))))
            mean_path_fit_dist = float(np.mean(np.sqrt(np.min(d2, axis=0))))
            nearest_path_idx = int(np.max(np.argmin(d2, axis=0)))
            path_coverage = float(nearest_path_idx) / float(max(1, len(path_np) - 1))

        self.last_history = full_history
        self.episode_return = float(total_reward)

        obs = self._get_obs()
        terminated = True
        truncated = False

        return obs, float(total_reward), terminated, truncated, {
            "episode_return": float(total_reward),
            "history_pos": full_history,
            "target_path": self.ctx.path.tolist(),
            "final_pos": self.ctx.particle_pos.tolist(),
            "layout_x": self.env.cylinders_x.tolist(),
            "layout_y": self.env.cylinders_y.tolist(),
            "layout_r": self.env.cylinders_r.tolist(),
            "layout_omega": np.full(
                self.env.max_n, self.env.fixed_omega, dtype=np.float32
            ).tolist(),
            "fixed_count": int(self.env.fixed_count),
            "terminated_early": bool(rollout_terminated),
            "layout_penalty": float(layout_penalty),
            "overlap_penalty": float(overlap_pen),
            "boundary_penalty": float(bound_pen),
            "sparse_penalty": float(sparse_penalty),
            "cluster_penalty": float(cluster_penalty),
            "active_count": int(active_count),
            "inactive_count": int(inactive_count),
            "total_count": int(total_count),
            "zero_radius_ratio": float(zero_radius_ratio),
            "killed_eps": float(killed_eps),
            "empty_layout_penalty": float(empty_layout_pen),
            "layout_mode": self.layout_mode,
            "inflow_u": float(self.env.inflow_u),
            "inflow_v": float(self.env.inflow_v),
            "global_omega": float(self.env.fixed_omega),
            "mean_path_dist": float(mean_path_fit_dist),
            "final_dist": float(final_dist),
            "path_coverage": float(path_coverage),
            "nearest_path_idx": int(nearest_path_idx),
            "target_to_traj_mean": float(target_to_traj_mean),
            "traj_to_target_mean": float(traj_to_target_mean),
            "path_block_penalty": float(block_pen),
            "far_cylinder_penalty": 0.0,
            "flow_tangent_penalty": float(flow_tangent_pen),
            "flow_normal_penalty": float(flow_normal_pen),
            "inflow_reg_penalty": float(inflow_reg_pen),
            "gate_lane_error": float(gate_lane_error),
            "gate_lane_miss_ratio": float(gate_lane_miss),
            "gate_lane_collision_ratio": float(gate_lane_collision),
            "gate_lane_order_penalty": float(gate_lane_order),
            "gate_lane_y_cross": gate_lane_cross,
            "gate_lane_target_y": gate_lane_target,
            "gate_dir_penalty": float(gate_dir_pen),
            "gate_reverse_ratio": float(gate_reverse_ratio),
            "logic_miss_ratio": float(logic_miss),
            "logic_collision_ratio": float(logic_collision),
            "logic_wrong_side_ratio": float(logic_wrong_side),
            "logic_outlet_error": float(logic_outlet_error),
            "logic_forward_penalty": float(logic_forward_pen),
            "logic_inlet_block_penalty": float(logic_inlet_block_pen),
            "logic_path_fit_error": float(logic_path_fit),
            "logic_path_cover_penalty": float(logic_path_cover),
            "logic_source_port": logic_source_port,
            "logic_target_port": logic_target_port,
            "logic_target_side": logic_target_side,
            "logic_target_xy": logic_target_xy,
            "logic_route_mode": logic_route_mode,
            "logic_route_pairs": logic_route_pairs,
            "logic_route_details": logic_route_details,
            "logic_exits": logic_exits,
            "logic_episode_source_port": str(self.logic_episode_source_port),
            "logic_episode_target_port": str(self.logic_episode_target_port),
            "logic_target_candidates": list(self.logic_target_candidates),
            "logic_reward_mode": str(getattr(LogicBoxConfig, "REWARD_MODE", "hybrid")),
        }

    def get_scene(self) -> dict:
        omegas = np.full(self.env.max_n, self.env.fixed_omega, dtype=np.float32)
        scene = {
            "particle": {
                "x": float(self.ctx.particle_pos[0]),
                "y": float(self.ctx.particle_pos[1]),
            },
            "target": {
                "x": float(self.ctx.goal[0]),
                "y": float(self.ctx.goal[1]),
            },
            "cylinders": {
                "x": self.env.cylinders_x.tolist(),
                "y": self.env.cylinders_y.tolist(),
                "r": self.env.cylinders_r.tolist(),
                "omegas": omegas.tolist(),
                "fixed_count": int(self.env.fixed_count),
            },
            "inflow": {
                "u": float(self.env.inflow_u),
                "v": float(self.env.inflow_v),
            },
            "global_omega": float(self.env.fixed_omega),
        }
        if self.base_layout_mode == "gate3_layout":
            scene["gate"] = {
                "check_x": float(Gate3LevelConfig.CHECK_X),
                "seed_points": Gate3LevelConfig.lane_seed_points().tolist(),
                "target_y": np.asarray(Gate3LevelConfig.LANE_TARGET_Y, dtype=np.float32).tolist(),
                "lane_reward_mode": str(
                    getattr(Gate3LevelConfig, "LANE_REWARD_MODE", "single")
                ),
            }
        if self.base_layout_mode == "logic_box_layout":
            ports = logic_box_ports()
            (bx0, bx1), (by0, by1) = logic_box_bounds()
            seed_offsets = self._logic_seed_offsets()
            route_pairs = []
            if _logic_is_multi_route_mode():
                route_pairs = [list(p) for p in logic_box_active_route_pairs()]
            elif _logic_is_single_multi_target_mode():
                src = str(self.logic_episode_source_port).upper()
                route_pairs = [[src, t] for t in self.logic_target_candidates]
            src = str(getattr(LogicBoxConfig, "SOURCE_PORT", "L1")).upper()
            if src not in ports:
                src = "L1"
            seed_points = []
            if len(route_pairs) > 0:
                for pair_src, _ in route_pairs:
                    if pair_src not in ports:
                        continue
                    src_side, src_xy = ports[pair_src]
                    seed_x = float(src_xy[0])
                    if src_side == "left":
                        seed_x = float(bx0) + max(
                            1e-4, float(getattr(LogicBoxConfig, "SOURCE_SEED_X_INSET", 1e-4))
                        )
                    for dy in seed_offsets:
                        seed_points.append([float(seed_x), float(src_xy[1] + dy)])
            else:
                src_side, src_xy = ports[src]
                seed_x = float(src_xy[0])
                if src_side == "left":
                    seed_x = float(bx0) + max(
                        1e-4, float(getattr(LogicBoxConfig, "SOURCE_SEED_X_INSET", 1e-4))
                    )
                seed_points = [[float(seed_x), float(src_xy[1] + dy)] for dy in seed_offsets]
            scene["logic_box"] = {
                "x_range": [float(bx0), float(bx1)],
                "y_range": [float(by0), float(by1)],
                "source_port": str(self.logic_episode_source_port).upper(),
                "target_port": str(self.logic_episode_target_port).upper(),
                "route_mode": _logic_route_mode(),
                "route_pairs": route_pairs,
                "show_box": bool(getattr(LogicBoxConfig, "SHOW_BOX", True)),
                "ports": {
                    k: {"side": v[0], "xy": [float(v[1][0]), float(v[1][1])]}
                    for k, v in ports.items()
                },
                "seed_points": seed_points,
                "show_fixed_centers": bool(getattr(self.env, "logic_fixed_layout", False)),
                "fixed_centers": [
                    [float(xx), float(yy)]
                    for xx, yy in zip(self.env.cylinders_x.tolist(), self.env.cylinders_y.tolist())
                ],
            }
        return scene
