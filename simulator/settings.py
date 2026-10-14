"""Стандартні умови моделювання з simulator/config.yaml (або з файлу SWARM_SIM_CONFIG поверх нього).

Ключі файлу SWARM_SIM_CONFIG, яких немає в config.yaml (друкарська помилка: `tau` замість `tau_v`), не
ігноруються мовчки — попередження ConfigWarning. Неприпустимі значення (dt ≤ 0, тривалість ≤ 0) — ValueError.
"""
from __future__ import annotations

import copy
import math
import os
import warnings
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parent / "config.yaml"


class ConfigWarning(UserWarning):
    """Підозріле налаштування (невідомий ключ у SWARM_SIM_CONFIG)."""


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def unknown_keys(base: dict, over: dict, prefix: str = "") -> list[str]:
    """Шляхи ключів over, яких немає в base (рекурсивно): ["physics.tau", …]."""
    out = []
    for k, v in (over or {}).items():
        path = f"{prefix}{k}"
        if k not in base:
            out.append(path)
        elif isinstance(v, dict) and isinstance(base[k], dict):
            out += unknown_keys(base[k], v, path + ".")
    return out


def _positive(cfg: dict, section: str, key: str) -> float:
    try:
        x = float(cfg[section][key])
    except (TypeError, ValueError, KeyError) as e:
        got = (cfg.get(section) or {}).get(key)
        raise ValueError(f"налаштування {section}.{key}: потрібне число, отримано {got!r}") from e
    if not (math.isfinite(x) and x > 0):
        raise ValueError(f"налаштування {section}.{key} має бути > 0, отримано {x}")
    return x


def validate(cfg: dict) -> dict:
    """Перевірка значень, від яких залежить сама можливість моделювання (ділення на них, кількість тактів)."""
    for key in ("dt", "tau_v", "gust_tau", "zone_radius"):
        _positive(cfg, "physics", key)
    _positive(cfg, "episode", "duration")
    return cfg


@lru_cache(maxsize=None)
def _load(path: str | None) -> dict:
    cfg = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"SWARM_SIM_CONFIG: немає файлу {p}")
        over = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(over, dict):
            raise ValueError(f"SWARM_SIM_CONFIG {p}: очікується словник YAML (розділи scenario, physics, episode)")
        bad = unknown_keys(cfg, over)
        if bad:
            warnings.warn(f"SWARM_SIM_CONFIG {p.name}: невідомі ключі {', '.join(bad)} — їх проігноровано "
                          f"(див. simulator/config.yaml)", ConfigWarning, stacklevel=2)
        cfg = _merge(cfg, over)
    return validate(cfg)


def settings() -> dict:
    """Повний словник налаштувань (копія — змінювати безпечно)."""
    return copy.deepcopy(_load(os.environ.get("SWARM_SIM_CONFIG") or None))


SETTINGS = settings()
SCENARIO = SETTINGS["scenario"]
PHYSICS = SETTINGS["physics"]
EPISODE = SETTINGS["episode"]
DURATION = float(EPISODE["duration"])           # тривалість епізоду, с (стандартна для fly, пульта й генератора)


def n_steps(duration: float, dt: float) -> int:
    """Кількість тактів епізоду тривалістю duration при кроці dt (30 с / 0,05 с = 600)."""
    return int(round(float(duration) / float(dt)))
