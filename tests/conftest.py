"""Спільні фікстури. Дані варіанта (variant_01/student: variant.json, eval_scenarios.json, stress_scenarios.json)
створюються в тимчасовій теці, яка на час тестів стає поточною: так працюють `--variant 1`, scenario_set(1) і живий
режим пульта (find_variant_dir шукає variant_XX/student від поточної теки)."""
import json
import os
import shutil
from pathlib import Path

import pytest

from simulator.scenarios import save_scenarios, variant_scenarios

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "variant_01" / "student" / "variant.json"


@pytest.fixture(scope="session", autouse=True)
def variant_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    sd = root / "variant_01" / "student"
    sd.mkdir(parents=True)
    shutil.copy(EXAMPLE, sd / "variant.json")
    v = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    save_scenarios(variant_scenarios(v, 1001, 30, n_fixed=5, tag="eval"), sd / "eval_scenarios.json",
                   variant=1, sigma_y=v["sigma_y"])
    stress = dict(v, n_lo=v["stress_n_range"][0], n_hi=v["stress_n_range"][1], wind_max=v["stress_wind_max"])
    save_scenarios(variant_scenarios(stress, 1002, v["n_stress"], tag="stress"), sd / "stress_scenarios.json",
                   variant=1, sigma_y=v["sigma_y"])
    old = os.getcwd()
    os.chdir(root)
    yield root
    os.chdir(old)
