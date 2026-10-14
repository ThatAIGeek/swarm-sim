"""Генерація сценаріїв (стартових умов епізоду). Стандартні значення — simulator/config.yaml (розділ scenario).

Старт: випадково в області cube (x, y, висота) з центром start.center, висота z0 … z0 + cube[2], мінімальна відстань
між дронами min_dist, курс випадковий, початкова швидкість 0–v0_max у випадковому напрямку. Ціль: за goal.distance
від центру старту по горизонталі, на висоті goal.height; напрямок — goal.azimuth_deg або випадковий; goal.point —
фіксована точка. Вітер: випадковий горизонтальний вектор |w| ∈ [0, wind_max], пориви з СКВ gust_std.
Зерно визначає сценарій повністю; послідовність випадкових чисел не залежить від налаштувань (фіксований напрямок
чи точка цілі замінює вже витягнуті значення), тож інші сценарії не зсуваються.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import Scenario
from .settings import SCENARIO as _SC

_ST, _GO, _WI = _SC["start"], _SC["goal"], _SC["wind"]
CUBE = tuple(float(x) for x in _ST["cube"])
Z0 = float(_ST["z0"])
MIN_START_DIST = float(_ST["min_dist"])
START_CENTER = np.array([float(x) for x in _ST["center"]])
V0_MAX = float(_ST["v0_max"])
GOAL_DIST = tuple(float(x) for x in _GO["distance"])
GOAL_Z = tuple(float(x) for x in _GO["height"])
GOAL_AZIMUTH = None if _GO.get("azimuth_deg") is None else float(_GO["azimuth_deg"])
GOAL_POINT = None if _GO.get("point") is None else np.array([float(x) for x in _GO["point"]])
WIND_MAX = float(_WI["max"])
GUST_RANGE = tuple(float(x) for x in _WI["gust_range"])
SCENARIOS_FORMAT = "lab4-swarm-scenarios/1"


def start_positions(rng: np.random.Generator, n: int, cube=CUBE, min_dist: float = MIN_START_DIST) -> np.ndarray:
    pts: list[np.ndarray] = []
    lo = np.array([START_CENTER[0] - cube[0] / 2, START_CENTER[1] - cube[1] / 2, Z0])
    hi = np.array([START_CENTER[0] + cube[0] / 2, START_CENTER[1] + cube[1] / 2, Z0 + cube[2]])
    for _ in range(100_000):
        p = rng.uniform(lo, hi)
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
            if len(pts) == n:
                return np.array(pts)
    raise RuntimeError(f"не вдалося розмістити {n} дронів")


def make_scenario(seed: int, n_agents: int, goal_dist=GOAL_DIST, wind_max: float = WIND_MAX,
                  gust_range=GUST_RANGE, cube=CUBE, v0_max: float = V0_MAX, meta: dict | None = None,
                  goal_z=GOAL_Z) -> Scenario:
    """Сценарій, повністю визначений зерном і параметрами."""
    if int(n_agents) < 1:
        raise ValueError(f"кількість дронів має бути ≥ 1, отримано {n_agents}")
    if int(seed) < 0:
        raise ValueError(f"зерно сценарію має бути ≥ 0, отримано {seed}")
    rng = np.random.default_rng(seed)
    pos = start_positions(rng, n_agents, cube)
    yaw = rng.uniform(-np.pi, np.pi, n_agents)
    a = rng.uniform(-np.pi, np.pi)
    r = rng.uniform(*goal_dist)
    gz = rng.uniform(*goal_z)                                  # витягується завжди: потік чисел той самий
    if GOAL_AZIMUTH is not None:
        a = np.deg2rad(GOAL_AZIMUTH)
    goal = np.array([START_CENTER[0] + r * np.cos(a), START_CENTER[1] + r * np.sin(a), gz])
    if GOAL_POINT is not None:
        goal = GOAL_POINT.copy()
    b = rng.uniform(-np.pi, np.pi)
    w = rng.uniform(0.0, wind_max)
    wind = np.array([w * np.cos(b), w * np.sin(b), 0.0])
    gust = rng.uniform(*gust_range)
    v0 = rng.normal(size=(n_agents, 3)) * np.array([1.0, 1.0, 0.3])
    v0 *= (rng.uniform(0, v0_max, n_agents) / np.maximum(np.linalg.norm(v0, axis=1), 1e-9))[:, None]
    return Scenario(n_agents, pos, yaw, goal, wind, gust, seed, dict(meta or {}), v0)


def make_scenarios(seed: int, n: int, n_range=(3, 8), n_fixed: int | None = None, **kw) -> list[Scenario]:
    """n сценаріїв: N рівномірно з n_range (або фіксоване n_fixed); дочірні зерна з головного."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        na = int(n_fixed) if n_fixed is not None else int(rng.integers(n_range[0], n_range[1] + 1))
        child = int(rng.integers(1, 2**31 - 1))
        out.append(make_scenario(child, na, meta={"index": i}, **kw))
    return out


# ---------------------------------------------------------------- параметри варіанта і файли сценаріїв

N_SHAPES = {               # розподіл кількості дронів N у межах діапазону варіанта
    "рівномірний": lambda ns: np.ones(len(ns)),
    "більше малих": lambda ns: 1.0 / ns,
    "більше великих": lambda ns: ns.astype(float),
}
GUST_MIN = GUST_RANGE[0]


def sample_n(rng: np.random.Generator, v: dict) -> int:
    ns = np.arange(v["n_lo"], v["n_hi"] + 1)
    w = N_SHAPES[v["n_shape"]](ns)
    return int(rng.choice(ns, p=w / w.sum()))


def scenario_kwargs(v: dict) -> dict:
    """Параметри make_scenario для варіанта v (словник variant.json)."""
    return dict(goal_dist=tuple(v.get("goal_dist", GOAL_DIST)), wind_max=v["wind_max"],
                gust_range=(v.get("gust_min", GUST_MIN), v["gust_max"]), cube=tuple(v.get("cube", CUBE)),
                v0_max=v.get("v0_max", V0_MAX), goal_z=tuple(v.get("goal_z", GOAL_Z)))


def variant_scenarios(v: dict, seed: int, n: int, n_fixed: int | None = None, tag: str = "", **over) -> list[Scenario]:
    """n сценаріїв із розподілу варіанта v: N за розподілом варіанта (або n_fixed), вітер і пориви варіанта."""
    if n_fixed is not None and int(n_fixed) < 1:                   # раніше 0 мовчки означав «за розподілом»
        raise ValueError(f"кількість дронів має бути ≥ 1, отримано {n_fixed}")
    rng = np.random.default_rng(seed)
    kw = scenario_kwargs(v) | over
    out = []
    for i in range(n):
        na = n_fixed if n_fixed is not None else sample_n(rng, v)
        child = int(rng.integers(1, 2**31 - 1))
        out.append(make_scenario(child, int(na), meta={"set": tag, "index": i}, **kw))
    return out


def find_variant_dir(variant: int, roots=None) -> Path:
    """Тека даних студента варіанта: variant_XX/student (архів) або data/variant_XX/student (репозиторій)."""
    here = Path(__file__).resolve().parent.parent
    for r in roots or (Path.cwd(), Path.cwd().parent, here):
        for p in (r / f"variant_{variant:02d}" / "student", r / "data" / f"variant_{variant:02d}" / "student"):
            if (p / "variant.json").exists() or (p / "eval_scenarios.json").exists():
                return p
    raise FileNotFoundError(f"не знайдено variant_{variant:02d}/student (дані варіанта з архіву)")


def variant_params(variant: int) -> dict:
    return json.loads((find_variant_dir(variant) / "variant.json").read_text(encoding="utf-8"))


def save_scenarios(scenarios: list[Scenario], path, **header) -> Path:
    """Файл сценаріїв: {"format", ...header (variant, sigma_y, …), "scenarios": [...]}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(format=SCENARIOS_FORMAT, **header, scenarios=[s.to_dict() for s in scenarios])
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def _doc(path) -> dict:
    path = Path(path)
    if path.is_dir():
        raise IsADirectoryError(f"{path}: це тека, а не файл сценаріїв *.json")
    if not path.exists():
        raise FileNotFoundError(f"немає файлу сценаріїв {path}")
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"{path.name}: це не JSON-файл сценаріїв ({e}); його дає swarm-sim scenarios") from e
    d = {"scenarios": d} if isinstance(d, list) else d
    if not isinstance(d, dict) or not isinstance(d.get("scenarios"), list):
        raise ValueError(f"{path.name}: немає списку \"scenarios\" — це не файл сценаріїв (swarm-sim scenarios)")
    return d


def load_scenarios(path) -> list[Scenario]:
    items = _doc(path)["scenarios"]
    try:
        return [Scenario.from_dict(d) for d in items]
    except (KeyError, TypeError, ValueError) as e:
        what = f"бракує поля {e.args[0]!r}" if isinstance(e, KeyError) else str(e)
        raise ValueError(f"{Path(path).name}: зіпсований сценарій ({what})") from e


def scenarios_sigma_y(path):
    """σ_y з файлу сценаріїв або з variant.json поруч (None, якщо немає)."""
    d = _doc(path)
    if d.get("sigma_y") is not None:
        return d["sigma_y"]
    vj = Path(path).parent / "variant.json"
    if vj.exists():
        return json.loads(vj.read_text(encoding="utf-8")).get("sigma_y")
    return None
