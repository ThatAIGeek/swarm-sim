"""Симулятор рою квадрокоптерів — окрема програма swarm-sim (python -m simulator).

Модель у форматі ONNX (simulator/modelfile.py) «програється» на сценаріях, результат — запис (simulator/record.py).
Контролер Рейнольдса — одна з моделей, вбудована в пакет (simulator.REYNOLDS). Два режими: headless (fly, metrics) і візуальний
(view — 3D-пульт у браузері; рахує той самий рушій у Python).

Python API (ті самі функції, що й у CLI):
    from simulator import fly, metrics, scenario_set, new_scenarios, load_record, check, stamp, view

Імпорт лінивий: `import simulator` не тягне onnxruntime і не читає налаштувань, доки функція не потрібна
(CLI так показує помилки налаштувань SWARM_SIM_CONFIG повідомленням, а не трейсбеком).
"""
from __future__ import annotations

import importlib

_LAZY = {"REYNOLDS": ("engine", "REYNOLDS"), "fly": ("engine", "fly"), "Model": ("modelfile", "Model"),
         "check": ("modelfile", "check"), "stamp": ("modelfile", "stamp"), "load_record": ("record", "load"),
         "metrics": ("stats", "metrics"), "print_metrics": ("stats", "print_metrics")}


def __getattr__(name: str):
    if name in _LAZY:
        mod, attr = _LAZY[name]
        val = getattr(importlib.import_module(f".{mod}", __name__), attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def scenario_set(variant: int, name: str = "eval"):
    """Готовий набір сценаріїв варіанта з даних архіву: eval — 30 публічних (N = 5), stress — стрес-набір
    (N = 8–10, сильніший вітер)."""
    from .scenarios import find_variant_dir, load_scenarios

    return load_scenarios(find_variant_dir(variant) / f"{name}_scenarios.json")


def new_scenarios(variant: int, seed: int, n: int, n_agents: int | None = None):
    """n нових сценаріїв із розподілу варіанта (N, вітер, пориви) з власним зерном."""
    from .scenarios import variant_params, variant_scenarios

    return variant_scenarios(variant_params(variant), seed, n, n_fixed=n_agents, tag=f"seed_{seed}")


def view(*args, **kw):
    """Візуальний режим (3D-пульт у браузері), див. simulator/view."""
    from .view.server import view as _view

    return _view(*args, **kw)


__all__ = ["REYNOLDS", "fly", "metrics", "print_metrics", "scenario_set", "new_scenarios", "load_record", "check",
           "stamp", "Model", "view"]
