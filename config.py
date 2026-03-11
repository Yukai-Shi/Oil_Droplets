import numpy as np

class StokesCylinderConfig:
    # --- 物理常数 ---
    # 旋转速度固定为约 0.2 Hz (1.256 rad/s)
    FIXED_OMEGA = 1.256  
    
    # --- 组合优化搜索空间 ---
    NUM_CYLINDERS = 10      # 原 MAX_N，改为 NUM_CYLINDERS 与 FluidEnv 对齐
    
    # 放置圆柱的中心点允许范围 (单位: m)
    X_RANGE = (-0.15, 0.15)
    Y_RANGE = (-0.15, 0.15)
    
    # 半径变化范围 (a_i)
    MIN_R = 0.002           # 最小半径 2mm
    MAX_R = 0.025           # 最大半径 15mm
    
    # --- 任意数量逻辑阈值 ---
    # 如果 RL 输出的动作值指示半径低于此值，物理上将其视为 0 (不放置)
    EXIST_THRESHOLD = 0.003 

class TrajectorySettingConfig:
    # 物理仿真步长：数值稳定性关键，Stokes 流虽无惯性，但步长过大会导致欧拉积分发散
    SIMUL_DT = 0.005  
    # RL 决策步长：AI 每 0.1s 改变一次布局
    ACTION_DT = 0.1
    # 轨迹点总数
    PATH_POINTS = 500

class RenderSettingConfig:
    # 渲染边界应略大于搜索边界 (X_RANGE)，以便完整看到边缘的圆柱
    X_LIM = (-0.20, 0.20)
    Y_LIM = (-0.20, 0.20)
    
    # 图像参数
    WIDTH = 640
    HEIGHT = 640
    DPI = 100
    
    # 实时渲染流场时的网格分辨率 (30-50 为宜)
    GRID_RESOLUTION = 30

class TrainingSettingConfig:
    # 一局游戏的上限步数
    EPISODE_LENGTH = 100   
    # 训练总步数 (3亿步对于 PPO 来说非常充裕，可以配合 EvalCallback 早停)
    TOTAL_TIMESTEPS = 300_000_000
    # PPO 的 Buffer 大小控制
    BATCH_SIZE = 256