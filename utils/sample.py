import random
import numpy as np
from typing import Optional, Sequence, Tuple

from config import RankineSettingConfig, RenderSettingConfig

def _is_legal_point(   
    pos: Sequence[float], 
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    centers: Optional[Sequence[Sequence[float]]] = None,
    radius: Optional[float] = None,
    safety_margin: float = 0.005
) -> bool:
    p = np.asarray(pos, dtype=np.float32).reshape(2)
    if xlim is None:
        xlim = RenderSettingConfig.X_LIM
    if ylim is None:
        ylim = RenderSettingConfig.Y_LIM
    if centers is None:
        centers = RankineSettingConfig.RANKINES_POS
    if radius is None:
        radius = RankineSettingConfig.R

    if not (xlim[0] <= p[0] <= xlim[1] and ylim[0] <= p[1] <= ylim[1]):
        return False
    centers = np.asarray(centers, dtype=np.float32)
    if centers.size == 0:
        return True

    dx = centers[:, 0] - p[0]
    dy = centers[:, 1] - p[1]
    r2 = float(radius + safety_margin) ** 2
    return np.min(dx * dx + dy * dy) >= r2


def random_sample(
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    centers: Optional[Sequence[Sequence[float]]] = None,
    radius: Optional[float] = None,
    max_tries: int = 10000,
    rng: Optional[np.random.Generator] = None,
    safety_margin: float = 0.005
) -> np.ndarray:
    if xlim is None:
        xlim = RenderSettingConfig.X_LIM
    if ylim is None:
        ylim = RenderSettingConfig.Y_LIM
    if centers is None:
        centers = RankineSettingConfig.RANKINES_POS
    if radius is None:
        radius = RankineSettingConfig.R
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_tries):
        p = np.array([rng.uniform(*xlim), rng.uniform(*ylim)], dtype=np.float32)
        if _is_legal_point(p, xlim, ylim, centers, radius, safety_margin):
            return p

    raise RuntimeError("random_sample: exceed max_tries without finding a valid point")


def around_sample(
    base: Sequence[float],
    max_step: float = 0.01,
    min_step: float = 0.003,
    rng: Optional[np.random.Generator] = None,
    max_tries: int = 10000,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    base = np.asarray(base, dtype=np.float32).reshape(2)

    for _ in range(max_tries):
        r = rng.uniform(min_step, max_step)
        theta = rng.uniform(0.0, 2*np.pi)
        candidate = base + r * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        if _is_legal_point(candidate):
            return candidate.astype(np.float32)
    raise RuntimeError("around_sample: exceed max_tries without finding a valid point")