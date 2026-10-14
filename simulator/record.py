"""Запис (record) польоту: один рядок = один дрон, живий на початку такту, на одному такті (20 Гц).

Колонки:
    episode_id, step, t                — епізод, номер такту, час, с
    agent_id, n_agents                 — номер дрона в епізоді, розмір рою
    model                              — хто керував дроном: 0 — --model, 1 — --rest-model
    px py pz vx vy vz yaw              — стан дрона у світовій системі на початку такту (м, м/с, рад)
    status                             — стан після такту: 1 — летить, 2 — зіткнувся, 3 — вийшов із зони
    goal_x goal_y goal_z               — ціль епізоду
    nb{k}_dpx … nb{k}_dvz, nb{k}_mask  — сенсор neighbors_detector: сусід k = 0…7 у body frame
    self_vx … self_dgoal               — сенсор self_sensor (7)
    act_x act_y act_z                  — виконана дія в body frame, м/с
    act_wx act_wy act_wz               — та сама дія у світовій системі
    label_x … label_wz                 — дія вчителя (--teacher) у тих самих станах (body і world); лише якщо є вчитель
    cmd_x … cmd_wz                     — команда моделі до шуму дії (body і world); лише якщо --action-noise > 0:
                                         online MSE рахується з неї, а не з виконаної дії із шумом

Метадані файлу (ключ lab4.record, JSON): моделі (назва, sha256, тип спостереження), параметри симулятора,
тривалість, σ_y (для online MSE), шум дії і по кожному епізоду — сценарій, час першої події, мінімальну відстань
на кроці події і кінцеві статуси дронів. Тож запис самодостатній: з нього рахуються метрики і його показує пульт.

Формат — parquet (zstd): стовпчиковий, типізований, у кілька разів менший за CSV, читає лише потрібні колонки
і несе метадані в собі. Для перегляду очима: swarm-sim export rec.parquet --csv.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .sensors import MASK_COLS, NB_COLS, SELF_COLS

FORMAT = "lab4-swarm-record/1"
META_KEY = b"lab4.record"
ID_COLS = ["episode_id", "step", "t", "agent_id", "n_agents", "model"]
STATE_COLS = ["px", "py", "pz", "vx", "vy", "vz", "yaw"]
GOAL_COLS = ["goal_x", "goal_y", "goal_z"]
OBS_COLS = NB_COLS + MASK_COLS + SELF_COLS
ACT_COLS = ["act_x", "act_y", "act_z"]
ACT_WORLD_COLS = ["act_wx", "act_wy", "act_wz"]
LABEL_COLS = ["label_x", "label_y", "label_z"]
LABEL_WORLD_COLS = ["label_wx", "label_wy", "label_wz"]
CMD_COLS = ["cmd_x", "cmd_y", "cmd_z"]
CMD_WORLD_COLS = ["cmd_wx", "cmd_wy", "cmd_wz"]
INT_COLS = {"episode_id": np.int32, "step": np.int16, "agent_id": np.int8, "n_agents": np.int8, "model": np.int8,
            "status": np.int8}


def columns(teacher: bool, cmd: bool = False) -> list[str]:
    return (ID_COLS + STATE_COLS + ["status"] + GOAL_COLS + OBS_COLS + ACT_COLS + ACT_WORLD_COLS
            + (LABEL_COLS + LABEL_WORLD_COLS if teacher else []) + (CMD_COLS + CMD_WORLD_COLS if cmd else []))


@dataclass
class Record:
    """Запис польоту: таблиця рядків (може бути None, якщо не тримали в пам'яті) і метадані."""
    frame: pd.DataFrame | None
    meta: dict
    path: Path | None = None

    @property
    def episodes(self) -> list[dict]:
        return self.meta["episodes"]

    def load_frame(self, columns: list[str] | None = None) -> pd.DataFrame:
        if self.frame is not None and columns is None:
            return self.frame
        if self.frame is not None:
            return self.frame[columns]
        return pd.read_parquet(self.path, columns=columns)


def write_meta(meta: dict) -> dict:
    return {META_KEY: json.dumps(meta, ensure_ascii=False).encode()}


def read_meta(path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    if path.is_dir():
        raise IsADirectoryError(f"{path}: це тека, а не файл запису *.parquet")
    if not path.exists():
        raise FileNotFoundError(f"немає файлу запису {path}")
    try:
        kv = pq.read_metadata(path).metadata or {}
    except pa.ArrowException as e:                     # не parquet (JSON, обірваний запис без кінцевого блоку…)
        raise ValueError(f"{path.name}: це не файл parquet (запис симулятора — swarm-sim fly … -o rec.parquet)") from e
    if META_KEY not in kv:
        raise ValueError(f"{path}: це не запис симулятора (немає метаданих {META_KEY.decode()})")
    return json.loads(kv[META_KEY])


def load(path, columns: list[str] | None = None, frame: bool = True) -> Record:
    """Прочитати запис. columns — лише потрібні колонки; frame=False — тільки метадані."""
    path = Path(path)
    meta = read_meta(path)
    return Record(pd.read_parquet(path, columns=columns) if frame else None, meta, path)


def to_csv(path, out, episode: int | None = None, columns: list[str] | None = None) -> Path:
    """Фрагмент запису у CSV для перегляду очима (episode — лише один епізод; неіснуючий — ValueError)."""
    meta = read_meta(path)
    if episode is not None:
        ids = [e["episode_id"] for e in meta["episodes"]]
        if episode not in ids:
            rng = f"{min(ids)}…{max(ids)}" if ids else "немає жодного"
            raise ValueError(f"у записі {Path(path).name} немає епізоду {episode} (номери епізодів: {rng})")
    df = pd.read_parquet(path, columns=columns, filters=None if episode is None else [("episode_id", "==", episode)])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def to_arrays(df: pd.DataFrame, ep_meta: list[dict]) -> dict:
    """Рядки запису → масиви (T, E, N) для метрик і пульта (порядок епізодів — як у ep_meta)."""
    ids = [e["episode_id"] for e in ep_meta]
    pos_of = {eid: i for i, eid in enumerate(ids)}
    E = len(ids)
    N = max(e["n_agents"] for e in ep_meta)
    T = int(df.step.max()) + 1 if len(df) else 0
    e = df.episode_id.map(pos_of).to_numpy()
    k = df.step.to_numpy().astype(int)
    i = df.agent_id.to_numpy().astype(int)
    alive = np.zeros((T, E, N), dtype=bool)
    alive[k, e, i] = True
    out = dict(alive=alive, n_agents=np.array([x["n_agents"] for x in ep_meta]))
    for name, cols in (("pos", ["px", "py", "pz"]), ("vel", ["vx", "vy", "vz"]), ("act", ACT_WORLD_COLS)):
        a = np.zeros((T, E, N, 3))
        a[k, e, i] = df[cols].to_numpy(np.float64)
        out[name] = a
    yaw = np.zeros((T, E, N))
    yaw[k, e, i] = df.yaw.to_numpy(np.float64)
    out["yaw"] = yaw
    model = np.full((T, E, N), -1, dtype=np.int8)
    model[k, e, i] = df.model.to_numpy()
    out["model"] = model
    if "label_wx" in df:
        lab = np.full((T, E, N, 3), np.nan)
        lab[k, e, i] = df[LABEL_WORLD_COLS].to_numpy(np.float64)
        out["label"] = lab
    out["goal"] = np.array([x["scenario"]["goal"] for x in ep_meta], dtype=np.float64)
    return out
