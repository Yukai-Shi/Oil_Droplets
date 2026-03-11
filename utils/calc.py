import numpy as np
from typing import Tuple, Sequence, Optional

# ---------------------------------------------------------
# 1. 网格速度计算 (用于 Renderer 绘图或流场匹配)
# ---------------------------------------------------------
def calculate_velocity_grid(
    X: np.ndarray, 
    Y: np.ndarray, 
    x_cyls: np.ndarray, 
    y_cyls: np.ndarray, 
    radii: np.ndarray, # <--- 修改：现在是数组
    omega: np.ndarray  # <--- 修改：现在是数组
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算整个网格上的速度场叠加。
    X, Y: 由 np.meshgrid 生成的网格坐标
    """
    # 转换输入确保为 numpy 数组
    x_cyls = np.asarray(x_cyls)
    y_cyls = np.asarray(y_cyls)
    radii = np.asarray(radii)
    omega = np.asarray(omega)

    # 计算强度因子 Gamma = 2 * pi * a^2 * omega
    gamma = 2 * np.pi * (radii**2) * omega

    # 利用广播机制计算位移 (H, W, N)
    dx = X[:, :, None] - x_cyls[None, None, :]
    dy = Y[:, :, None] - y_cyls[None, None, :]

    r2 = dx**2 + dy**2
    r = np.sqrt(r2)
    
    # 避免奇异点
    r2_safe = np.where(r2 == 0, np.inf, r2)

    # 逻辑修改：每个圆柱都有自己的 radii[i]
    # 我们用掩码处理圆柱内部（ core_mask ）和外部（ outer_mask ）
    # radii[None, None, :] 会自动对应到第 3 轴的 N 个圆柱
    core_mask = r <= radii[None, None, :]
    outer_mask = r > radii[None, None, :]

    u_x = np.zeros_like(X, dtype=float)
    u_y = np.zeros_like(Y, dtype=float)

    # 1. 内部场（刚性旋转部分）： v = -omega * dy, u = omega * dx
    # 使用 np.where 进行掩码计算并叠加
    u_x += np.sum(np.where(core_mask, -omega[None, None, :] * dy, 0), axis=2)
    u_y += np.sum(np.where(core_mask, omega[None, None, :] * dx, 0), axis=2)

    # 2. 外部场（Stokes Rotlet 部分）： v_theta = gamma / (2 * pi * r)
    u_x += np.sum(np.where(outer_mask, -gamma[None, None, :] * dy / (2 * np.pi * r2_safe), 0), axis=2)
    u_y += np.sum(np.where(outer_mask, gamma[None, None, :] * dx / (2 * np.pi * r2_safe), 0), axis=2)

    return u_x, u_y


# ---------------------------------------------------------
# 2. 单点速度计算 (用于 Particle 运动模拟)
# ---------------------------------------------------------
def calculate_point_velocity(
    x: float, 
    y: float, 
    x_cyls: np.ndarray, 
    y_cyls: np.ndarray, 
    radii: np.ndarray, 
    omega: np.ndarray
) -> Tuple[float, float]:
    
    x_cyls = np.asarray(x_cyls)
    y_cyls = np.asarray(y_cyls)
    omega = np.asarray(omega)
    radii = np.asarray(radii)

    # 物理强度
    gamma = 2 * np.pi * (radii**2) * omega
    dx = x - x_cyls
    dy = y - y_cyls
    r2 = dx**2 + dy**2
    
    # 此时只计算外部场（假设粒子不会进入圆柱内部，由 is_legal 保证）
    r2 = np.where(r2 < 1e-12, 1e-12, r2)

    vx = np.sum(-gamma * dy / (2 * np.pi * r2))
    vy = np.sum(gamma * dx / (2 * np.pi * r2))

    return float(vx), float(vy)

# ---------------------------------------------------------
# 3. 碰撞与边界检查
# ---------------------------------------------------------
def is_legal(
    pos: Sequence[float], 
    centers: np.ndarray, 
    radii: np.ndarray,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None
) -> bool:
    from config import RenderSettingConfig
    p = np.asarray(pos, dtype=np.float32).reshape(2)
    
    if xlim is None: xlim = RenderSettingConfig.X_LIM
    if ylim is None: ylim = RenderSettingConfig.Y_LIM

    # 边界
    if not (xlim[0] <= p[0] <= xlim[1] and ylim[0] <= p[1] <= ylim[1]):
        return False

    # 动态半径碰撞
    if centers.size == 0: return True
    
    dx = centers[:, 0] - p[0]
    dy = centers[:, 1] - p[1]
    dist_sq = dx**2 + dy**2
    
    # 核心：判断距离平方是否小于对应圆柱的半径平方
    return not np.any(dist_sq < (radii ** 2))