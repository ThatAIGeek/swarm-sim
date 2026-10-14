"""Спостереження — вхід моделі. Модель у метаданих ONNX вказує свій тип (lab4.observation), і симулятор подає
їй саме його. Батч B — усі живі дрони, якими керує модель, з усіх епізодів одночасно (B — кількість рядків).

A — «наївне», абсолютні світові координати, x (B, 63) = 6 + 9·6 + 3:
    [0:6]   власні p (3) і v (3) у світовій системі;
    [6:60]  p і v (по 6) для 9 інших дронів (N_max = 10) у порядку їхніх номерів, без сортування за відстанню;
            порожні місця (дронів менше 10) і мертві дрони — нулі;
    [60:63] координати цілі.
    Не інваріантне до зсуву, повороту і перестановки сусідів; фіксує N ≤ 10 (так задумано: завдання 2). Рій, де
    живий дрон із номером ≥ 10, спостереження A не вміщує — ValueError (а не мовчки обрізаний рій).
B — канонічне, плаский вектор x (B, 63) = 48 + 8 + 7: nb (8 сусідів × 6) підряд, mask (8), self_state (7).
C — множина сусідів, три входи: nb (B, 8, 6), mask (B, 8), self_state (B, 7) — ті самі дані, що й у B.

Колонки запису (record) для канонічного спостереження: NB_COLS, MASK_COLS, SELF_COLS.
"""
from __future__ import annotations

import numpy as np

from .neighbors_detector import K_MAX, detect_neighbors
from .self_sensor import sense_self

N_MAX_A = 10
A_DIM = 6 * N_MAX_A + 3          # 63
B_DIM = K_MAX * 6 + K_MAX + 7    # 63
OBS_KINDS = ("A", "B", "C")

NB_COLS = [f"nb{k}_{q}" for k in range(K_MAX) for q in ("dpx", "dpy", "dpz", "dvx", "dvy", "dvz")]
MASK_COLS = [f"nb{k}_mask" for k in range(K_MAX)]
SELF_COLS = ["self_vx", "self_vy", "self_vz", "self_gx", "self_gy", "self_gz", "self_dgoal"]


def canonical(pos, vel, yaw, alive, goal):
    """Сенсори neighbors_detector + self_sensor: nb (..., N, 8, 6), mask (..., N, 8), self_state (..., N, 7), float64."""
    nb, mask = detect_neighbors(pos, vel, yaw, alive)
    return nb, mask, sense_self(pos, vel, yaw, alive, goal)


def build_observation(pos, vel, yaw, alive, goal):
    """Канонічне спостереження у float32 (так, як його пише запис)."""
    nb, mask, ss = canonical(pos, vel, yaw, alive, goal)
    return nb.astype(np.float32), mask.astype(np.float32), ss.astype(np.float32)


def observation_A(pos, vel, alive, goal, n_max: int = N_MAX_A) -> np.ndarray:
    """A (..., N, 63), float64 (див. опис модуля)."""
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    alive = np.asarray(alive, dtype=bool)
    lead, N = pos.shape[:-2], pos.shape[-2]
    if N > n_max and alive[..., n_max:].any():               # доповнення пакета до спільного N — не живі, їх можна
        raise ValueError(f"спостереження A вміщує рій до {n_max} дронів (6 + 9·6 + 3 = 63 числа), а тут рій до "
                         f"{N} дронів: для N > {n_max} потрібна модель B або C (або сценарії з N ≤ {n_max})")
    pv = np.concatenate([pos, vel], axis=-1) * alive[..., None]              # (..., N, 6)
    out = np.zeros(lead + (N, 6 * n_max + 3))
    out[..., :6] = pv
    m = min(n_max - 1, N - 1)
    if m > 0:
        others = np.array([[j for j in range(N) if j != i][:m] for i in range(N)])   # (N, m), за номерами
        out[..., 6:6 + 6 * m] = pv[..., others, :].reshape(lead + (N, 6 * m))
    out[..., -3:] = np.asarray(goal, dtype=np.float64)[..., None, :]
    return out * alive[..., None]


def observation_B(nb, mask, ss) -> np.ndarray:
    """B (..., 63) з канонічного спостереження."""
    return np.concatenate([nb.reshape(nb.shape[:-2] + (-1,)), mask, ss], axis=-1)


def observation_inputs(kind: str, state, sel: np.ndarray, cache: dict) -> list[np.ndarray]:
    """Входи моделі типу kind для дронів маски sel (E, N), у порядку входів контракту (float64).

    state — SwarmState (pos, vel, yaw, alive, goal); cache — словник такту, щоб не рахувати сенсори двічі.
    """
    if kind == "A":
        if "A" not in cache:
            cache["A"] = observation_A(state.pos, state.vel, state.alive, state.goal)
        return [cache["A"][sel]]
    if "C" not in cache:
        cache["C"] = canonical(state.pos, state.vel, state.yaw, state.alive, state.goal)
    nb, mask, ss = cache["C"]
    if kind == "B":
        return [observation_B(nb[sel], mask[sel], ss[sel])]
    if kind == "C":
        return [nb[sel], mask[sel], ss[sel]]
    raise ValueError(f"невідомий тип спостереження {kind!r} (очікується A, B або C)")
