"""Сенсор self_sensor — власний стан дрона і навігація до цілі.

self_state (..., N, 7):
    [0:3] v_body      — власна швидкість у body frame, м/с
    [3:6] goal_dir    — одиничний вектор на ціль у body frame (нулі, якщо дрон точно на цілі)
    [6]   d_goal_norm — min(відстань до цілі / 50 м, 1): «далеко» = 1, на цілі = 0
Мертвий дрон — нулі.
"""
from __future__ import annotations

import numpy as np

from ..geometry import world_to_body

GOAL_NORM_DIST = 50.0   # м, масштаб нормування відстані до цілі


def sense_self(pos, vel, yaw, alive, goal) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)[..., None, :] - pos
    gd = np.linalg.norm(g, axis=-1, keepdims=True)
    ss = np.concatenate([
        world_to_body(vel, yaw),
        world_to_body(np.where(gd > 1e-9, g / np.maximum(gd, 1e-9), 0.0), yaw),
        np.minimum(gd / GOAL_NORM_DIST, 1.0),
    ], axis=-1)
    return ss * np.asarray(alive, dtype=bool)[..., None]
