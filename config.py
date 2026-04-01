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
    NUM_CYLINDERS = 10

    X_RANGE = (-0.15, 0.15)
    Y_RANGE = (-0.15, 0.15)

    MIN_R = 0.002
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
    BOX_X_RANGE = (-0.08, 0.08)
    BOX_Y_RANGE = (-0.10, 0.10)
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

    # Routing task: from SOURCE_PORT to TARGET_PORT.
    # SOURCE:  L0/L1/L2
    # TARGET:  R0/R1/R2/T0/B0
    SOURCE_PORT = "L1"
    TARGET_PORT = "R1"
    # Route objective mode:
    # - "single": optimize one route SOURCE_PORT -> TARGET_PORT
    # - "multi_map": optimize multiple fixed source->target routes together
    # - "single_multi_target": fixed SOURCE_PORT, per-episode target sampled from TARGET_PORT_SET
    ROUTE_MODE = "single_multi_target"
    TARGET_PORT_SET = ["R0", "R1", "R2"]
    # Used when ROUTE_MODE="multi_map".
    # Default mapping: L0->T0, L1->R1, L2->B0
    MULTI_ROUTE_PAIRS = [
        ("L0", "T0"),
        ("L1", "R1"),
        ("L2", "B0"),
    ]

    # Whether logic-box uses fixed cylinder centers (only radii are optimized).
    # This is useful for "fixed layout + variable radius/omega" tasks.
    FIXED_LAYOUT_ENABLE = False
    # Keep logic-box policy action dimension as x/y/r even when FIXED_LAYOUT_ENABLE=True.
    # This allows seamless stage switching (layout-search -> fixed-layout fine-tune)
    # without changing policy output dimension.
    KEEP_XY_ACTION_WHEN_FIXED = True
    # Default fixed centers (10 cylinders) inside local box.
    # You can replace these arrays with your own fixed arrangement.
    FIXED_LAYOUT_X = np.array(
        [-0.048, 0.000, 0.048, -0.048, 0.000, 0.048, -0.048, 0.000, 0.048, 0.000],
        dtype=np.float32,
    )
    FIXED_LAYOUT_Y = np.array(
        [0.064, 0.064, 0.064, 0.000, 0.000, 0.000, -0.064, -0.064, -0.064, 0.090],
        dtype=np.float32,
    )

    # Streamline tracing settings.
    # "x_march" is recommended for left-to-right routing tasks.
    TRACE_MODE = "x_march"
    X_STEP = 0.003
    MAX_X_STEPS = 240
    TRACE_DT = 0.004
    TRACE_STEPS = 300
    MIN_FORWARD_VX = 2e-5
    # Seed/start x inset from left boundary to avoid hugging the wall.
    SOURCE_SEED_X_INSET = 0.003
    # Around-source seed offsets for robustness; keep short for speed.
    SOURCE_SEED_DY = np.array([-0.004, 0.0, 0.004], dtype=np.float32)
    # Split seed profiles:
    # - train: use fewer seeds for faster optimization
    # - eval: use denser seeds for robustness checking
    TRAIN_SOURCE_SEED_DY = np.array([0.0], dtype=np.float32)
    EVAL_SOURCE_SEED_DY = np.array([0.0], dtype=np.float32)

    # Outlet matching scale.
    OUTLET_SIGMA = 0.020

    # Logic-box specific geometry controls (to avoid oversized/overlapped layouts).
    MAX_R = 0.014
    EXIST_THRESHOLD = 0.0045
    # Hard floor on active radius in logic-box mode (when FORBID_ELIMINATION=True).
    MIN_ACTIVE_R = 0.0035
    # If True, logic-box radii are clamped to at least MIN_ACTIVE_R (no r=0 elimination).
    FORBID_ELIMINATION = True
    # Radius <= KILLED_EPS is counted as "eliminated" in diagnostics.
    KILLED_EPS = 1e-6
    OVERLAP_MARGIN = 0.003

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
    RUN_ALIAS = "one2three_FORBID_ELIMINATION"
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


class RankineSettingConfig:
    # Backward-compatible fields for legacy scripts.
    R = float(StokesCylinderConfig.MAX_R)
    RANKINES_POS = np.array([[0.0, 0.0]], dtype=np.float32)


def get_logic_box_ranges():
    mode = str(getattr(LogicBoxConfig, "BOUNDS_MODE", "local_box")).strip().lower()
    if mode in {"global", "global_field", "full_field", "render_field", "world"}:
        return tuple(RenderSettingConfig.X_LIM), tuple(RenderSettingConfig.Y_LIM)
    return tuple(LogicBoxConfig.BOX_X_RANGE), tuple(LogicBoxConfig.BOX_Y_RANGE)


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
