import numpy as np


class LayoutModeConfig:
    # "free_layout": optimize x/y/r for each cylinder
    # "fixed_grid_3x3": fixed 3x3 centers, optimize only radii
    # "free_layout_inflow": free layout + uniform inflow
    # "free_layout_inflow_u_fixed": free layout + horizontal inflow fixed (u fixed, v=0)
    # "fixed_grid_3x3_inflow": 3x3 fixed layout + uniform inflow
    # "fixed_grid_3x3_inflow_u_fixed": 3x3 fixed layout + horizontal inflow fixed
    # "gate3_layout_inflow": 3 fixed gate cylinders + free design cylinders + inflow
    # "gate3_layout_inflow_u_fixed": gate3 layout + horizontal inflow fixed
    # "logic_box_layout_inflow": rectangular black-box routing with configurable ports
    # "logic_box_layout_inflow_u_fixed": logic-box routing + horizontal inflow fixed
    LAYOUT_MODE = "logic_box_layout_inflow_u_fixed"


class StokesCylinderConfig:
    FIXED_OMEGA = 1.256
    NUM_CYLINDERS = 9

    X_RANGE = (-0.15, 0.15)
    Y_RANGE = (-0.15, 0.15)

    MIN_R = 0.001
    MAX_R = 0.025
    EXIST_THRESHOLD = 0.003


class GlobalOmegaControlConfig:
    # Whether policy controls a single global omega shared by all cylinders.
    OPTIMIZE_OMEGA = True

    # "sign": only direction changes (+/-|FIXED_OMEGA|), same magnitude.
    # "continuous": omega is optimized in [OMEGA_MIN, OMEGA_MAX].
    MODE = "continuous"

    OMEGA_MIN = -1.256
    OMEGA_MAX = 1.256


class FixedGrid3x3Config:
    CENTER = (0.0, 0.0)
    PITCH = 0.06
    NUM_CYLINDERS = 9

    @staticmethod
    def grid_coords():
        cx, cy = FixedGrid3x3Config.CENTER
        p = FixedGrid3x3Config.PITCH
        xs = np.array([cx - p, cx, cx + p], dtype=np.float32)
        ys = np.array([cy + p, cy, cy - p], dtype=np.float32)
        gx, gy = [], []
        for y in ys:
            for x in xs:
                gx.append(float(x))
                gy.append(float(y))
        return np.array(gx, dtype=np.float32), np.array(gy, dtype=np.float32)


class TrajectorySettingConfig:
    SIMUL_DT = 0.005
    ACTION_DT = 0.1
    PATH_POINTS = 500


class InflowConfig:
    # Uniform inflow added to cylinder-induced velocity field.
    # Default is horizontal inflow from left to right.
    U_IN = 0.002
    V_IN = 0.0

    # Whether inflow components are optimized by policy in *_inflow modes.
    # In *_inflow_u_fixed modes, inflow is fixed (u fixed, v=0) regardless of OPTIMIZE_U/V.
    OPTIMIZE_U = True
    OPTIMIZE_V = True

    # Physical bounds for optimized inflow components.
    U_MIN = 0.0
    U_MAX = 0.008
    V_MIN = -0.008
    V_MAX = 0.008

    # Regularization for optimized inflow to prevent saturation at bounds.
    REG_WEIGHT = 35.0
    TARGET_U = 0.001
    TARGET_V = 0.0
    # Deadzone (physical units): penalty grows only when deviation exceeds this band.
    REG_DEADZONE = 0.002


class Gate3LevelConfig:
    # Total cylinder count in this mode uses StokesCylinderConfig.NUM_CYLINDERS.
    NUM_FIXED = 3

    # Three fixed cylinders, top to bottom, on the right side of the start region.
    FIXED_X = np.array([0.050, 0.050, 0.050], dtype=np.float32)
    FIXED_Y = np.array([0.090, 0.030, -0.030], dtype=np.float32)
    FIXED_R = np.array([0.018, 0.018, 0.018], dtype=np.float32)

    # Start region for lane-seed streamlines.
    START_X = -0.120
    START_Y = 0.010
    SEED_Y_OFFSETS = np.array([0.055, 0.010, -0.055], dtype=np.float32)

    # Evaluate whether streamlines pass the intended lane at this x position.
    CHECK_X = 0.090

    # Target passage y for the three lanes: above / gap / below.
    # These values are chosen to keep clearances from fixed cylinders.
    LANE_TARGET_Y = np.array([0.125, 0.060, -0.065], dtype=np.float32)

    # Streamline scoring mode:
    # "x_march" (recommended): march by x with dy/dx = vy/vx, much faster.
    # "time_trace": classic time stepping.
    TRACE_MODE = "x_march"

    # Fast x-march settings.
    X_STEP = 0.003
    MAX_X_STEPS = 140
    MIN_FORWARD_VX = 1e-4

    # Legacy time-trace settings (used only when TRACE_MODE="time_trace").
    TRACE_DT = 0.004
    TRACE_STEPS = 900

    # Penalty fallback when a lane never reaches CHECK_X.
    MISS_Y_PENALTY = 0.14

    # Reward weights for gate mode.
    W_LAYOUT = 8.0
    W_CLUSTER = 0.004
    W_LANE_ERR = 160.0
    W_LANE_MISS = 80.0
    W_LANE_COLLISION = 50.0
    W_ORDER = 40.0
    # Direction-consistency on selected lane path (for omega sign control).
    W_DIR_SIGN = 0.0
    W_DIR_REVERSE = 0.0
    # "single": evaluate only selected lane by PATH_TYPE (top/mid/bottom)
    # "all": evaluate all three lanes together
    LANE_REWARD_MODE = "single"

    @staticmethod
    def lane_seed_points() -> np.ndarray:
        ys = Gate3LevelConfig.START_Y + Gate3LevelConfig.SEED_Y_OFFSETS
        xs = np.full_like(ys, Gate3LevelConfig.START_X)
        return np.stack([xs, ys], axis=1).astype(np.float32)


class LogicBoxConfig:
    # Rectangular local black-box area (used when BOUNDS_MODE="local_box").
    BOX_X_RANGE = (-0.14, 0.18)
    BOX_Y_RANGE = (-0.14, 0.14)
    # Boundary mode for logic-box tasks:
    # - "local_box": use BOX_X/Y_RANGE
    # - "global_field": use full render field (RenderSettingConfig.X/Y_LIM)
    BOUNDS_MODE = "local_box"
    # Whether to draw the inner box frame in visualization.
    SHOW_BOX = True

    # Left-side 3 inlets (top -> bottom), right-side 3 outlets.
    LEFT_PORT_Y = np.array([0.065, 0.000, -0.065], dtype=np.float32)
    RIGHT_PORT_Y = np.array([0.065, 0.000, -0.065], dtype=np.float32)
    TOP_PORT_X = 0.000
    BOTTOM_PORT_X = 0.000
    # If True, port coordinates above are interpreted in BOX_X/Y_RANGE reference space
    # and automatically scaled to current logic bounds (local_box or global_field).
    PORT_AUTO_SCALE_TO_BOUNDS = True

    # Routing task: from SOURCE_PORT to TARGET_PORT.
    # SOURCE:  L0/L1/L2
    # TARGET:  R0/R1/R2/T0/B0
    SOURCE_PORT = "L1"
    TARGET_PORT = "R1"
    # Route objective mode:
    # - "single": optimize one route SOURCE_PORT -> TARGET_PORT
    # - "multi_map": optimize multiple fixed source->target routes together
    # - "multi_map_switch": episode-wise switch among multiple route maps.
    #   Each map is still evaluated as 3 inlets simultaneously routing to 3 outlets.
    # - "single_multi_target": fixed SOURCE_PORT, per-episode target sampled from TARGET_PORT_SET
    ROUTE_MODE = "multi_map_switch"
    TARGET_PORT_SET = ["T0", "R0", "R2"]
    # Target sampling mode for single_multi_target training:
    # - "random": uniform random from TARGET_PORT_SET
    # - "cycle": round-robin over TARGET_PORT_SET (recommended for balance)
    # - "weighted": sample by TARGET_SAMPLE_WEIGHTS
    TARGET_SAMPLE_MODE = "cycle"
    # Used when TARGET_SAMPLE_MODE="weighted". Keys are target port names.
    TARGET_SAMPLE_WEIGHTS = {"T0": 1.0, "R0": 1.0, "B0": 1.0}
    # single_multi_target best-checkpoint criterion:
    # - "min": maximize worst-target reward
    # - "mean": maximize average reward
    # - "success_then_mean": maximize number of successful targets first, then mean reward
    MULTI_TARGET_BEST_METRIC = "mean"
    # Success thresholds used when MULTI_TARGET_BEST_METRIC="success_then_mean".
    MULTI_TARGET_SUCCESS_MISS_MAX = 0.0
    MULTI_TARGET_SUCCESS_WRONG_MAX = 0.0
    MULTI_TARGET_SUCCESS_OUTLET_MAX = 0.60
    # Hard-min training (single_multi_target only):
    # at each eval checkpoint, find current worst target (lowest reward),
    # then force subsequent training episodes to that target until next eval.
    HARD_MIN_TRAIN_ENABLE = False
    # If True, also force eval_env to hard-min target between eval checkpoints.
    # Usually unnecessary; keep False to avoid side effects on generic eval logs.
    HARD_MIN_FORCE_EVAL_ENV = False
    # Used when ROUTE_MODE="multi_map".
    # Default mapping: L0->T0, L1->R1, L2->B0
    MULTI_ROUTE_PAIRS = [
        ("L0", "T0"),
        ("L1", "R1"),
        ("L2", "B0"),
    ]
    # Used when ROUTE_MODE="multi_map_switch".
    # Each element is one full route-map (usually 3 pairs for L0/L1/L2).
    # Policy should learn to output different (r, omega, inflow) for different map ids
    # while sharing the same x/y layout.
    MULTI_ROUTE_SETS = [
        [("L0", "T0"), ("L1", "R0"), ("L2", "B0")],
        [("L0", "R0"), ("L1", "R1"), ("L2", "B0")],
    ]
    # Sampling of route-map index in multi_map_switch:
    # - "random": uniform random over MULTI_ROUTE_SETS
    # - "cycle": round-robin over MULTI_ROUTE_SETS (recommended)
    # - "weighted": sample by MULTI_ROUTE_SET_WEIGHTS
    MULTI_ROUTE_SET_SAMPLE_MODE = "cycle"
    # Weights aligned with MULTI_ROUTE_SETS when sample mode is weighted.
    MULTI_ROUTE_SET_WEIGHTS = [1.0, 1.0]

    # Whether logic-box uses fixed cylinder centers (only radii are optimized).
    # This is useful for "fixed layout + variable radius/omega" tasks.
    FIXED_LAYOUT_ENABLE = True
    # Keep logic-box policy action dimension as x/y/r even when FIXED_LAYOUT_ENABLE=True.
    # This allows seamless stage switching (layout-search -> fixed-layout fine-tune)
    # without changing policy output dimension.
    KEEP_XY_ACTION_WHEN_FIXED = True
    # Default fixed centers (3x3 cylinders) inside local box.
    # You can replace these arrays with your own fixed arrangement.
    FIXED_LAYOUT_X = np.array(
        [-0.075, 0.020, 0.115, -0.075, 0.020, 0.115, -0.075, 0.020, 0.115],
        dtype=np.float32,
    )
    FIXED_LAYOUT_Y = np.array(
        [0.080, 0.080, 0.080, 0.000, 0.000, 0.000, -0.080, -0.080, -0.080],
        dtype=np.float32,
    )

    # Streamline tracing settings.
    # "x_march" is recommended for left-to-right routing tasks.
    TRACE_MODE = "x_march"
    X_STEP = 0.004
    MAX_X_STEPS = 180
    TRACE_DT = 0.004
    TRACE_STEPS = 300
    MIN_FORWARD_VX = 2e-5
    # In x_march mode, do not hard-fail immediately on low vx near inlet.
    # Use short time-trace fallback so streamlines can escape local recirculation.
    LOW_VX_USE_TIMETRACE_FALLBACK = True
    LOW_VX_PATIENCE = 100
    LOW_VX_FALLBACK_DT = 0.004
    # Seed/start x inset from left boundary to avoid hugging the wall.
    SOURCE_SEED_X_INSET = 0.006
    # Keep a single fixed seed x (no inward re-seeding retries).
    # Final seed x = x_left + SOURCE_SEED_X_INSET.
    SOURCE_RESEED_X_OFFSETS = np.array([0.0], dtype=np.float32)
    # Around-source seed offsets for robustness; keep short for speed.
    SOURCE_SEED_DY = np.array([-0.004, 0.0, 0.004], dtype=np.float32)
    # Split seed profiles:
    # - train: use fewer seeds for faster optimization
    # - eval: use denser seeds for robustness checking
    TRAIN_SOURCE_SEED_DY = np.array([0.0], dtype=np.float32)
    EVAL_SOURCE_SEED_DY = np.array([0.0], dtype=np.float32)

    # Outlet matching scale.
    OUTLET_SIGMA = 0.020
    # Soft clearance target around source seed region (for anti-block reward term).
    INLET_CLEARANCE = 0.012

    # Logic-box specific geometry controls (to avoid oversized/overlapped layouts).
    # This MAX_R value is defined for BOX_X/Y_RANGE scale. When
    # MAX_R_AUTO_SCALE_TO_BOUNDS=True, it scales with logic bounds size.
    MAX_R = 0.020
    MAX_R_AUTO_SCALE_TO_BOUNDS = False
    EXIST_THRESHOLD = 0.0045
    # Hard floor on active radius in logic-box mode (when FORBID_ELIMINATION=True).
    MIN_ACTIVE_R = 0.005
    # If True, logic-box radii are clamped to at least MIN_ACTIVE_R (no r=0 elimination).
    FORBID_ELIMINATION = False
    # Radius <= KILLED_EPS is counted as "eliminated" in diagnostics.
    KILLED_EPS = 1e-6
    # Stage-wise active-cylinder gate (optional):
    # - ACTIVE_CYL_LIMIT <= 0: disabled (all cylinders active)
    # - ACTIVE_CYL_LIMIT = K>0: only first K cylinders are active; others are forced to INACTIVE_LOCK_R
    # Useful for curriculum: early stages use small K, later stages increase K.
    ACTIVE_CYL_LIMIT = 0
    # Radius assigned to inactive cylinders when ACTIVE_CYL_LIMIT is enabled.
    # 0.0 means fully disabled (no flow influence).
    INACTIVE_LOCK_R = 0.0
    OVERLAP_MARGIN = 0.003
    # If True, enforce non-overlap as a hard geometric post-process in apply_layout().
    # This uses radius shrink (not center movement) to keep layout feasible.
    HARD_NON_OVERLAP_ENABLE = True
    # If True, when shrink-only is insufficient, perform minimal center-separation projection.
    HARD_NON_OVERLAP_MOVE_CENTERS = True
    # Iterations for hard non-overlap radius resolution.
    HARD_NON_OVERLAP_ITERS = 12
    # If False, overlap term is not included in reward layout penalty.
    # Useful when hard non-overlap is enabled and we want to avoid double-counting.
    USE_OVERLAP_PENALTY = False

    # Reward weights.
    # REWARD_MODE:
    # - "port_only": optimize outlet logic only (miss/wrong-side/outlet/collision/forward)
    # - "streamline_only": optimize route-shape fitting only (path_fit/path_cover)
    # - "hybrid": combine both groups (recommended default)
    REWARD_MODE = "port_only"

    W_LAYOUT = 36.0
    W_CLUSTER = 0.0
    W_MISS = 130.0
    W_COLLISION = 110.0
    W_WRONG_SIDE = 120.0
    W_OUTLET_POS = 60.0
    W_FORWARD = 18.0
    # Penalize layouts that block source-seed neighborhood near inlet.
    # Set to 0.0 to disable.
    W_INLET_BLOCK = 0.0
    # Extra penalty when all design radii are zero (avoid trivial empty-layout local optimum).
    W_EMPTY_LAYOUT = 30.0
    # Streamline-to-target-route fitting (one-shot, no particle rollout).
    W_PATH_FIT = 320.0
    W_PATH_COVER = 120.0


class RenderSettingConfig:
    X_LIM = (-0.20, 0.20)
    Y_LIM = (-0.20, 0.20)

    WIDTH = 640
    HEIGHT = 640
    DPI = 100
    GRID_RESOLUTION = 30


class TrainingSettingConfig:
    EPISODE_LENGTH = 100
    TOTAL_TIMESTEPS = 300_000_000
    BATCH_SIZE = 256
    PATH_TYPE = "logic_route"
    # Optional custom suffix appended to task result folder/tag.
    # Example: RUN_ALIAS = "omega_r_only_v1"
    RUN_ALIAS = "one2three_mean_Nonvoerlap"
    # Train-time diagnostics rollout for visualization only.
    RUN_DIAGNOSTIC_ROLLOUT = False
    # Visualization output switches:
    # - SAVE_EVAL_PREVIEW: save eval_* preview image every eval checkpoint.
    # - SAVE_BEST_PREVIEW: save best_* preview image only when best is updated.
    SAVE_EVAL_PREVIEW = False
    SAVE_BEST_PREVIEW = True
    # In inflow modes: freeze inflow at baseline for early training,
    # then unfreeze for fine adjustment.
    INFLOW_WARMUP_ENABLE = True
    INFLOW_WARMUP_RATIO = 0.20
    # Auto stage scheduler (single-run alternating free<->fixed layout).
    # This requires logic_box mode and is intended to avoid manual two-stage restarts.
    AUTO_STAGE_ENABLE = False
    # Switch stage every N eval checkpoints.
    AUTO_STAGE_EVALS_PER_PHASE = 3
    # Optional forced target used when snapshotting free-layout centers to freeze.
    # Empty string means use environment's sampled target.
    AUTO_STAGE_SNAPSHOT_TARGET = ""
    # One-stage shared-xy policy:
    # - xy is optimized by a target-agnostic actor branch
    # - r/omega/inflow is optimized by a target-aware actor branch
    # Requires logic_box free-layout action shape.
    SHARED_XY_ONE_STAGE_ENABLE = False


class RankineSettingConfig:
    # Backward-compatible fields for legacy scripts.
    R = float(StokesCylinderConfig.MAX_R)
    RANKINES_POS = np.array([[0.0, 0.0]], dtype=np.float32)


def get_logic_box_ranges():
    mode = str(getattr(LogicBoxConfig, "BOUNDS_MODE", "local_box")).strip().lower()
    if mode in {"global", "global_field", "full_field", "render_field", "world"}:
        return tuple(RenderSettingConfig.X_LIM), tuple(RenderSettingConfig.Y_LIM)
    return tuple(LogicBoxConfig.BOX_X_RANGE), tuple(LogicBoxConfig.BOX_Y_RANGE)


def _rescale_from_reference(values, ref_range, out_range):
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    r0, r1 = float(ref_range[0]), float(ref_range[1])
    o0, o1 = float(out_range[0]), float(out_range[1])
    r_mid = 0.5 * (r0 + r1)
    o_mid = 0.5 * (o0 + o1)
    r_half = max(0.5 * abs(r1 - r0), 1e-8)
    o_half = max(0.5 * abs(o1 - o0), 1e-8)
    rel = (vals - r_mid) / r_half
    out = o_mid + rel * o_half
    lo, hi = (o0, o1) if o0 <= o1 else (o1, o0)
    return np.clip(out, lo, hi).astype(np.float32)


def get_logic_port_coordinates():
    """
    Return logic-box port coordinates adapted to current bounds.
    """
    (x0, x1), (y0, y1) = get_logic_box_ranges()
    left_y = np.asarray(LogicBoxConfig.LEFT_PORT_Y, dtype=np.float32).reshape(-1)
    right_y = np.asarray(LogicBoxConfig.RIGHT_PORT_Y, dtype=np.float32).reshape(-1)
    top_x = np.array([float(LogicBoxConfig.TOP_PORT_X)], dtype=np.float32)
    bot_x = np.array([float(LogicBoxConfig.BOTTOM_PORT_X)], dtype=np.float32)

    if bool(getattr(LogicBoxConfig, "PORT_AUTO_SCALE_TO_BOUNDS", True)):
        left_y = _rescale_from_reference(left_y, LogicBoxConfig.BOX_Y_RANGE, (y0, y1))
        right_y = _rescale_from_reference(right_y, LogicBoxConfig.BOX_Y_RANGE, (y0, y1))
        top_x = _rescale_from_reference(top_x, LogicBoxConfig.BOX_X_RANGE, (x0, x1))
        bot_x = _rescale_from_reference(bot_x, LogicBoxConfig.BOX_X_RANGE, (x0, x1))

    left_y = np.clip(left_y, min(y0, y1), max(y0, y1)).astype(np.float32)
    right_y = np.clip(right_y, min(y0, y1), max(y0, y1)).astype(np.float32)
    top_x = float(np.clip(top_x[0], min(x0, x1), max(x0, x1)))
    bot_x = float(np.clip(bot_x[0], min(x0, x1), max(x0, x1)))
    return {
        "left_y": left_y,
        "right_y": right_y,
        "top_x": top_x,
        "bottom_x": bot_x,
    }


def get_logic_max_radius():
    """
    Effective logic-box MAX_R after optional bound-based scaling.
    """
    base = float(getattr(LogicBoxConfig, "MAX_R", StokesCylinderConfig.MAX_R))
    if bool(getattr(LogicBoxConfig, "MAX_R_AUTO_SCALE_TO_BOUNDS", True)):
        (x0, x1), (y0, y1) = get_logic_box_ranges()
        ref_x0, ref_x1 = float(LogicBoxConfig.BOX_X_RANGE[0]), float(LogicBoxConfig.BOX_X_RANGE[1])
        ref_y0, ref_y1 = float(LogicBoxConfig.BOX_Y_RANGE[0]), float(LogicBoxConfig.BOX_Y_RANGE[1])
        ref_span = min(abs(ref_x1 - ref_x0), abs(ref_y1 - ref_y0))
        cur_span = min(abs(float(x1) - float(x0)), abs(float(y1) - float(y0)))
        if ref_span > 1e-8:
            base = base * (cur_span / ref_span)
    return float(
        np.clip(
            base,
            float(StokesCylinderConfig.MIN_R),
            float(StokesCylinderConfig.MAX_R),
        )
    )


def get_logic_multi_route_pairs():
    raw = getattr(LogicBoxConfig, "MULTI_ROUTE_PAIRS", [])
    pairs = []
    if isinstance(raw, (list, tuple)):
        for it in raw:
            if not isinstance(it, (list, tuple)) or len(it) != 2:
                continue
            src = str(it[0]).strip().upper()
            tgt = str(it[1]).strip().upper()
            if len(src) == 0 or len(tgt) == 0:
                continue
            pairs.append((src, tgt))
    if len(pairs) == 0:
        pairs = [("L0", "T0"), ("L1", "R1"), ("L2", "B0")]
    return pairs


def get_logic_multi_route_sets():
    raw = getattr(LogicBoxConfig, "MULTI_ROUTE_SETS", [])
    route_sets = []
    if isinstance(raw, (list, tuple)):
        for block in raw:
            if not isinstance(block, (list, tuple)):
                continue
            pairs = []
            for it in block:
                if not isinstance(it, (list, tuple)) or len(it) != 2:
                    continue
                src = str(it[0]).strip().upper()
                tgt = str(it[1]).strip().upper()
                if len(src) == 0 or len(tgt) == 0:
                    continue
                pairs.append((src, tgt))
            if len(pairs) > 0:
                route_sets.append(pairs)
    if len(route_sets) == 0:
        route_sets = [get_logic_multi_route_pairs()]
    return route_sets
