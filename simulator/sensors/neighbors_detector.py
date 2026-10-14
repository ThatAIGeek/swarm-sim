"""Сенсор neighbors_detector — виявлення сусідів дрона.

Для кожного живого дрона i на кожному такті:
1. відстані до всіх інших живих дронів;
2. лишаються ті, що ближче r_detect = 15 м; сортування за відстанню (стабільне), до k_max = 8 найближчих;
3. для кожного сусіда j — відносна позиція Δp = Rz(ψ_i)ᵀ (p_j − p_i) і відносна швидкість Δv = Rz(ψ_i)ᵀ (v_j − v_i)
   у body frame дрона i (x — уперед, y — ліворуч, z — угору);
4. незаповнені слоти — нулі з маскою 0.

Вихід: nb (..., N, k_max, 6) = [Δp (3), Δv (3)], mask (..., N, k_max). Мертвий дрон нічого не бачить (усі нулі) і
сам нікому не видимий.
"""
from __future__ import annotations

import numpy as np

from ..geometry import world_to_body

R_DETECT = 15.0      # м, радіус виявлення
K_MAX = 8            # скільки найближчих сусідів повертає сенсор


def select_neighbors(pos: np.ndarray, alive: np.ndarray, r_detect: float = R_DETECT, k_max: int = K_MAX):
    """Індекси найближчих сусідів.

    pos : (..., N, 3), alive : (..., N) bool. Повертає idx (..., N, K) — індекси, відсортовані за відстанню,
    і valid (..., N, K) bool (сусід існує, живий і ближче r_detect). K = min(k_max, N − 1).
    """
    N = pos.shape[-2]
    K = min(k_max, max(N - 1, 1))
    diff = pos[..., None, :, :] - pos[..., :, None, :]            # (..., i, j, 3) = p_j − p_i
    dist = np.linalg.norm(diff, axis=-1)
    ok = alive[..., None, :] & alive[..., :, None] & ~np.eye(N, dtype=bool)
    dist_m = np.where(ok, dist, np.inf)
    idx = np.argsort(dist_m, axis=-1, kind="stable")[..., :K]
    d_sel = np.take_along_axis(dist_m, idx, axis=-1)
    return idx, d_sel < r_detect


def _gather(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """a: (..., N, 3), idx: (..., N, K) -> (..., N, K, 3)."""
    flat = idx.reshape(*idx.shape[:-2], -1)
    out = np.take_along_axis(a, flat[..., None], axis=-2)
    return out.reshape(*idx.shape, a.shape[-1])


def detect_neighbors(pos, vel, yaw, alive, r_detect: float = R_DETECT, k_max: int = K_MAX):
    """nb (..., N, k_max, 6) і mask (..., N, k_max), float64 (див. опис модуля)."""
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    alive = np.asarray(alive, dtype=bool)
    lead, N = pos.shape[:-2], pos.shape[-2]
    nb = np.zeros(lead + (N, k_max, 6))
    mask = np.zeros(lead + (N, k_max))
    if N > 1:
        idx, valid = select_neighbors(pos, alive, r_detect, k_max)
        K = idx.shape[-1]
        y = yaw[..., :, None]
        v = valid[..., None]
        nb[..., :K, :3] = np.where(v, world_to_body(_gather(pos, idx) - pos[..., :, None, :], y), 0.0)
        nb[..., :K, 3:] = np.where(v, world_to_body(_gather(vel, idx) - vel[..., :, None, :], y), 0.0)
        mask[..., :K] = valid
    am = alive[..., None]
    return nb * am[..., None], mask * am
