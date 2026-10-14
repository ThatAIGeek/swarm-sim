"""Геометрія симулятора: повороти між світовою системою і body frame дрона, обмеження швидкості.

Світова система: x, y — горизонталь, z — угору (м). ψ (yaw, курс) — кут повороту дрона навколо осі z,
куди «дивиться» ніс у горизонтальній площині (рад, (−π, π], 0 — уздовж осі x).
Body frame — система дрона, повернута на курс: x — уперед, y — ліворуч, z — угору.

    Rz(ψ) = [[cos ψ, −sin ψ, 0], [sin ψ, cos ψ, 0], [0, 0, 1]],   x_body = Rz(ψ)ᵀ x_world,   x_world = Rz(ψ) x_body
"""
from __future__ import annotations

import numpy as np

V_MAX = 8.0          # м/с — обмеження команди швидкості (актуатор), застосовується до будь-якої моделі


def world_to_body(x: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rz(ψ)ᵀ x. x: (..., 3), yaw: (...) — broadcast по провідних осях."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * x[..., 0] + s * x[..., 1], -s * x[..., 0] + c * x[..., 1], x[..., 2]], axis=-1)


def body_to_world(x: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rz(ψ) x."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * x[..., 0] - s * x[..., 1], s * x[..., 0] + c * x[..., 1], x[..., 2]], axis=-1)


def clip_norm(v: np.ndarray, vmax: float = V_MAX) -> np.ndarray:
    """Обрізати норму вектора (..., 3) до vmax, зберігши напрям."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, vmax / np.maximum(n, 1e-12))
