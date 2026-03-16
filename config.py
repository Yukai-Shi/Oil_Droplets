import numpy as np


class LayoutModeConfig:
    # "free_layout": optimize x/y/r for each cylinder
    # "fixed_grid_3x3": fixed 3x3 centers, optimize only radii
    LAYOUT_MODE = "free_layout"


class StokesCylinderConfig:
    FIXED_OMEGA = 1.256
    NUM_CYLINDERS = 10

    X_RANGE = (-0.15, 0.15)
    Y_RANGE = (-0.15, 0.15)

    MIN_R = 0.002
    MAX_R = 0.025
    EXIST_THRESHOLD = 0.003


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
    PATH_TYPE = "soft_snake2"


class RankineSettingConfig:
    # Backward-compatible fields for legacy scripts.
    R = float(StokesCylinderConfig.MAX_R)
    RANKINES_POS = np.array([[0.0, 0.0]], dtype=np.float32)
