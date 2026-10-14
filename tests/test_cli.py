"""swarm-sim як програма на поганих входах: одне повідомлення «ПОМИЛКА: …» у stderr, код 2, без трейсбека
(QA-5, QA-6, QA-7, QA-14, QA-16, ARCH-7, ARCH-8, ARCH-11); формат моделі — фіксований batch, транспонований вихід,
перевага виходу v_cmd (ARCH-3, QA-4, QA-16); спостереження A при N > 10 (ARCH-13). Даних варіантів не потребує."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simulator import REYNOLDS, fly, metrics
from simulator.modelfile import Model, ModelFormatError, check, stamp
from simulator.scenarios import make_scenarios, save_scenarios

ROOT = Path(__file__).resolve().parent.parent


def run(*args, env: dict | None = None, cwd=None) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("SWARM_SIM_DEBUG", "SWARM_SIM_CONFIG")}
    e["PYTHONPATH"] = str(ROOT) + os.pathsep + e.get("PYTHONPATH", "")
    e.update(env or {})
    return subprocess.run([sys.executable, "-m", "simulator", *map(str, args)], capture_output=True, text=True,
                          cwd=cwd or ROOT, env=e, timeout=300)


def assert_error(r: subprocess.CompletedProcess, *needles: str, code: int = 2):
    """Помилка для людини: код виходу, «ПОМИЛКА: …» у stderr, жодного трейсбека."""
    assert r.returncode == code, (r.returncode, r.stdout, r.stderr)
    assert "Traceback" not in r.stderr and "Traceback" not in r.stdout, r.stderr
    assert "ПОМИЛКА:" in r.stderr or "error:" in r.stderr, r.stderr
    for s in needles:
        assert s in r.stderr, (s, r.stderr)


# ---------------------------------------------------------------- моделі ONNX, зібрані вручну (onnx.helper)


def _onnx_model(path: Path, in_shape, out_shape, body: str = "matmul", outputs=("v_cmd",), obs: str = "B",
                frame: str = "body") -> Path:
    """Лінійна модель x (63) → 3 з довільними оголошеними формами входу/виходу.

    body: matmul — y = x·W (B, 3); transpose — y = (x·W)ᵀ (3, B). outputs — імена виходів; вихід не «v_cmd»
    (якщо їх кілька) — нулі тієї ж форми (допоміжний вихід, що стоїть першим)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    W = np.random.default_rng(0).normal(0, 0.05, (63, 3)).astype(np.float32)
    inits = [numpy_helper.from_array(W, "W"), numpy_helper.from_array(np.zeros(1, np.float32), "zero")]
    nodes = [helper.make_node("MatMul", ["x", "W"], ["y0"]),
             helper.make_node("Transpose", ["y0"], ["y"], perm=[1, 0]) if body == "transpose" else
             helper.make_node("Identity", ["y0"], ["y"])]
    outs = []
    for name in outputs:
        aux = len(outputs) > 1 and name != "v_cmd"
        nodes.append(helper.make_node("Mul", ["y", "zero"], [name]) if aux else
                     helper.make_node("Identity", ["y"], [name]))
        outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, out_shape))
    g = helper.make_graph(nodes, "m", [helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)], outs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))
    return stamp(path, observation=obs, frame=frame)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    return dict(
        fixed=_onnx_model(d / "fixed.onnx", [1, 63], [1, 3]),               # експорт без dynamic_axes
        transposed=_onnx_model(d / "transp.onnx", ["B", 63], [3, "B"], body="transpose"),
        transposed_undeclared=_onnx_model(d / "transp2.onnx", ["B", 63], None, body="transpose"),
        two_out=_onnx_model(d / "two.onnx", ["B", 63], ["B", 3], outputs=("aux", "v_cmd")),
        ok=_onnx_model(d / "ok.onnx", ["B", 63], ["B", 3]),
        ok_A=_onnx_model(d / "okA.onnx", ["B", 63], ["B", 3], obs="A", frame="world"),
    )


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    sc = save_scenarios(make_scenarios(3, 2, n_fixed=3), d / "s.json")
    n1 = save_scenarios(make_scenarios(4, 2, n_fixed=1), d / "n1.json")
    big = save_scenarios(make_scenarios(5, 1, n_fixed=12), d / "n12.json")
    (d / "empty.json").write_text("{}")
    (d / "nolist.json").write_text("[]")
    (d / "bad.json").write_text("не json")
    (d / "adir").mkdir()
    rec = d / "rec.parquet"
    fly(REYNOLDS, sc, out=rec, teacher=REYNOLDS, duration=2.0, hz=False)
    rec1 = d / "n1.parquet"
    fly(REYNOLDS, n1, out=rec1, teacher=REYNOLDS, duration=2.0, hz=False)
    return dict(d=d, sc=sc, n1=n1, big=big, rec=rec, rec1=rec1)


# ---------------------------------------------------------------- формат моделі


def test_fixed_batch_rejected_with_hint(models):
    with pytest.raises(ModelFormatError, match="dynamic_axes"):
        Model(models["fixed"])
    r = run("check", models["fixed"])
    assert_error(r, "фіксовану вісь батча", "dynamic_axes", code=1)


@pytest.mark.parametrize("key", ["transposed", "transposed_undeclared"])
def test_transposed_output_rejected(models, key):
    """Вихід (3, B): раніше y.size == 3·B проходив, і reshape мовчки перемішував дії дронів."""
    with pytest.raises(ModelFormatError, match=r"очікується \(10, 3\)|очікується"):
        check(models[key])
    r = run("check", models[key])
    assert_error(r, "форму", code=1)


def test_output_named_v_cmd_preferred(models):
    m = Model(models["two_out"])
    assert m.output == "v_cmd"
    y = m.act([np.ones((4, 63))])
    assert np.abs(y).sum() > 0                                       # не нульовий «aux»
    assert check(models["ok"])["ok"]


def test_onnxruntime_errors_become_format_errors(models):
    """Помилка виконання onnxruntime (фіксований batch, обійдений перевіркою) — ModelFormatError, а не сирий збій."""
    m = Model(models["ok"])
    with pytest.raises(ModelFormatError, match="onnxruntime"):
        m.act([np.ones((4, 62))])


def test_probe_inputs_A_plausible():
    """Пробні спостереження A — як справжні: слоти p, v підряд, порожні — нулі, рій компактний (ARCH-10)."""
    from simulator.modelfile import probe_inputs
    x = probe_inputs("A", 500, np.random.default_rng(0))[0]
    assert x.shape == (500, 63)
    slots = x[:, 6:60].reshape(500, 9, 6)
    filled = np.abs(slots).sum(-1) > 0
    assert (~filled).any() and filled.any()
    own = x[:, :3]
    d = np.linalg.norm(slots[..., :3] - own[:, None], axis=-1)[filled]
    assert np.median(d) < 30                                           # інші дрони — поруч, а не за сотні метрів
    assert np.abs(slots[..., 3:][filled]).max() < 30                   # швидкості — у м/с рою
    # B і C не змінились: відбитки поведінки B/C-моделей (лідерборд) лишаються дійсними
    a = probe_inputs("C", 8, np.random.default_rng(1))
    b = probe_inputs("C", 8, np.random.default_rng(1))
    np.testing.assert_array_equal(a[0], b[0])


# ---------------------------------------------------------------- спостереження A при N > 10


def test_observation_A_rejects_more_than_10(models, files):
    from simulator.sensors import observation_A
    pos, vel = np.zeros((12, 3)), np.zeros((12, 3))
    with pytest.raises(ValueError, match="до 10 дронів"):
        observation_A(pos, vel, np.ones(12, bool), np.zeros(3))
    x = observation_A(pos, vel, np.r_[np.ones(10, bool), np.zeros(2, bool)], np.zeros(3))  # доповнення — можна
    assert x.shape == (12, 63)
    with pytest.raises(ValueError, match="спостереження A"):
        fly(models["ok_A"], files["big"], duration=1.0, hz=False)
    assert_error(run("fly", "--model", models["ok_A"], "--scenarios", files["big"], "-o", files["d"] / "x.parquet"),
                 "спостереження A")


# ---------------------------------------------------------------- CLI: файли й формати


def test_cli_missing_and_wrong_files(files):
    d = files["d"]
    assert_error(run("fly", "--model", d / "no.onnx", "--scenarios", files["sc"], "-o", d / "o.parquet"),
                 "немає файлу моделі")
    assert_error(run("fly", "--model", REYNOLDS, "--scenarios", d / "no.json", "-o", d / "o.parquet"), "немає файлу")
    assert_error(run("fly", "--model", REYNOLDS, "--scenarios", d / "empty.json", "-o", d / "o.parquet"), "scenarios")
    assert_error(run("fly", "--model", REYNOLDS, "--scenarios", d / "nolist.json", "-o", d / "o.parquet"),
                 "порожній список")
    assert_error(run("fly", "--model", REYNOLDS, "--scenarios", d / "bad.json", "-o", d / "o.parquet"), "не JSON")
    assert_error(run("fly", "--model", REYNOLDS, "--scenarios", files["sc"], "-o", d / "adir"), "тека")
    assert_error(run("metrics", files["sc"]), "не файл parquet")
    assert_error(run("check", d / "adir"), "тека")
    assert_error(run("view", d / "no.parquet", "--export", d / "x.html"), "немає файлу")
    assert_error(run("scenarios", "--variant", 99, "--set", "eval", "-o", d / "x.json"), "variant_99")
    r = run("view", "--port", 70000)
    assert r.returncode == 2 and "65535" in r.stderr and "Traceback" not in r.stderr


def test_cli_debug_shows_traceback(files):
    r = run("metrics", files["sc"], env={"SWARM_SIM_DEBUG": "1"})
    assert r.returncode != 0 and "Traceback" in r.stderr


# ---------------------------------------------------------------- CLI: параметри польоту


@pytest.mark.parametrize("args, needle", [
    (["--duration", "0"], "> 0"),
    (["--duration", "-1"], "> 0"),
    (["--duration", "1700"], "32767"),                     # step — int16: раніше тихе переповнення
    (["--action-noise", "-1"], "≥ 0"),
    (["--noise-seed", "-1"], "≥ 0"),
    (["--episode-offset", "-3"], "≥ 0"),
    (["--nn-agents", "0"], "≥ 1"),
    (["--rest-model", str(REYNOLDS)], "--nn-agents"),
])
def test_cli_fly_bad_params(files, args, needle):
    out = files["d"] / "bad.parquet"
    r = run("fly", "--model", REYNOLDS, "--scenarios", files["sc"], "-o", out, "--quiet", *args)
    assert_error(r, needle)
    assert not out.exists()


def test_fly_api_validates():
    sc = make_scenarios(1, 1, n_fixed=2)
    for kw in (dict(duration=0.0), dict(duration=float("nan")), dict(action_noise=-0.5), dict(noise_seed=-1),
               dict(episode_offset=-1), dict(nn_agents=0)):
        with pytest.raises(ValueError):
            fly(REYNOLDS, sc, hz=False, **kw)
    with pytest.raises(ValueError):
        make_scenarios(1, 1, n_fixed=0)


def test_cli_scenarios_n_agents(tmp_path, variant_root):
    """--n-agents 0 / −2 — помилка одразу (раніше 0 мовчки ігнорувався, −2 — 14 с роботи і RuntimeError)."""
    from simulator.scenarios import variant_scenarios
    v = dict(n_lo=3, n_hi=8, n_shape="рівномірний", wind_max=4.0, gust_max=1.0)
    with pytest.raises(ValueError):
        variant_scenarios(v, 1, 2, n_fixed=0)
    assert {s.n_agents for s in variant_scenarios(v, 1, 3, n_fixed=4)} == {4}
    from simulator.scenarios import find_variant_dir
    for n in ("0", "-2"):
        r = run("scenarios", "--variant", 1, "--seed", 5, "--n", 2, "--n-agents", n, "-o", tmp_path / "x.json",
                cwd=variant_root)
        assert_error(r, "--n-agents")
    r = run("scenarios", "--variant", 1, "--set", "eval", "--n-agents", 3, "-o", tmp_path / "e.json",
            cwd=variant_root)
    assert r.returncode == 0 and "УВАГА" in r.stderr and "30 сценаріїв" in r.stdout
    if (find_variant_dir(1, roots=[variant_root]) / "stress_scenarios.json").exists():
        r = run("scenarios", "--variant", 1, "--set", "stress", "-o", tmp_path / "s.json", cwd=variant_root)
        assert r.returncode == 0 and "100 сценаріїв" in r.stdout, r.stderr
        ns = {x["n_agents"] for x in json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))["scenarios"]}
        assert ns <= {8, 9, 10}
    r = run("scenarios", "--variant", 1, "--seed", 5, "--n", 1, "-o", tmp_path / "one.json", cwd=variant_root)
    assert r.returncode == 0 and "1 сценарій →" in r.stdout
    r = run("scenarios", "--variant", 1, "--seed", 5, "--n", 2, "-o", tmp_path / "two.json", cwd=variant_root)
    assert "2 сценарії →" in r.stdout


def test_plural():
    from simulator.__main__ import plural
    f = lambda n: plural(n, "сценарій", "сценарії", "сценаріїв")      # noqa: E731
    assert [f(n) for n in (1, 2, 4, 5, 11, 12, 14, 21, 22, 25, 101, 111)] == [
        "сценарій", "сценарії", "сценарії", "сценаріїв", "сценаріїв", "сценаріїв", "сценаріїв", "сценарій",
        "сценарії", "сценаріїв", "сценарій", "сценаріїв"]


# ---------------------------------------------------------------- metrics / export


def test_metrics_by_episode_single_drone(files, tmp_path):
    """N = 1: min_separation = None → «—», а не TypeError (ARCH-8, QA-5); --json у неіснуючу теку."""
    out = tmp_path / "нова" / "тека" / "m.json"
    r = run("metrics", files["rec1"], "--by-episode", "--json", out)
    assert r.returncode == 0, r.stderr
    assert "—" in r.stdout and out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["per_episode"]["min_separation"] == [None, None]


def test_export_episode(files, tmp_path):
    r = run("export", files["rec"], "--csv", tmp_path / "e.csv", "--episode", 999)
    assert_error(r, "немає епізоду 999")
    assert not (tmp_path / "e.csv").exists()
    r = run("export", files["rec"], "--csv", tmp_path / "sub" / "e1.csv", "--episode", 1)
    assert r.returncode == 0, r.stderr
    import pandas as pd
    assert set(pd.read_csv(tmp_path / "sub" / "e1.csv").episode_id) == {1}


def test_help_texts():
    r = run("fly", "--help")
    assert r.returncode == 0
    assert "пролетіти сценарії моделлю (headless) і зберегти запис" in run("--help").stdout
    assert "фрагмент запису у CSV" in run("--help").stdout
    for opt in ("--duration", "--noise-seed", "--quiet"):
        line = r.stdout.split(opt, 1)[1].split("\n  -", 1)[0]
        assert len(line.strip()) > 10, opt                                  # у кожного параметра є довідка
    assert ", с" in r.stdout and "м/с" in r.stdout


# ---------------------------------------------------------------- атомарний запис


def test_fly_write_is_atomic(tmp_path, files, models):
    """Перерваний політ не лишає часткового файлу і не затирає попередній запис (ARCH-11, QA-9)."""
    out = tmp_path / "rec.parquet"
    fly(REYNOLDS, files["sc"], out=out, duration=1.0, hz=False)
    before = out.read_bytes()
    # перший пакет (N = 3: вихід (3, 3) збігається з (B, 3)) уже записано, другий (N = 4) падає на формі виходу
    sc = make_scenarios(3, 1, n_fixed=3) + make_scenarios(6, 1, n_fixed=4)
    with pytest.raises(ModelFormatError):
        fly(models["transposed_undeclared"], sc, out=out, duration=1.0, hz=False, batch=1)
    assert out.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
    assert metrics(out)["n_episodes"] == 2


# ---------------------------------------------------------------- налаштування


def test_config_unknown_keys_warn_and_bad_dt(tmp_path, files):
    cfg = tmp_path / "c.yaml"
    fly_args = ("fly", "--model", REYNOLDS, "--scenarios", files["sc"], "-o", tmp_path / "r.parquet", "--quiet",
                "--duration", 0.5)
    cfg.write_text("physics:\n  tau: 5.0\nscenari: {}\n", encoding="utf-8")
    r = run(*fly_args, env={"SWARM_SIM_CONFIG": str(cfg)})
    assert r.returncode == 0 and "УВАГА" in r.stderr and "physics.tau" in r.stderr and "scenari" in r.stderr
    assert "Traceback" not in r.stderr
    ours = [ln for ln in r.stderr.splitlines() if "onnxruntime" not in ln]   # службові рядки ORT — не наші
    assert len(ours) == 1 and ours[0].startswith("УВАГА")                    # одним рядком
    cfg.write_text("physics:\n  dt: 0\n", encoding="utf-8")
    assert_error(run(*fly_args, env={"SWARM_SIM_CONFIG": str(cfg)}), "dt")
    cfg.write_text("episode:\n  duration: -5\n", encoding="utf-8")
    assert_error(run(*fly_args, env={"SWARM_SIM_CONFIG": str(cfg)}), "duration")


def test_settings_helpers():
    from simulator.core import SimParams
    from simulator.settings import DURATION, n_steps, unknown_keys
    assert DURATION == 30.0 and n_steps(DURATION, SimParams().dt) == 600
    assert unknown_keys({"a": {"b": 1}, "c": None}, {"a": {"b": 2, "x": 1}, "c": [1], "d": 0}) == ["a.x", "d"]
    with pytest.raises(ValueError):
        SimParams(dt=0.0)
    from simulator.engine import DURATION as D2
    from simulator.stats import GOAL_RADIUS
    assert D2 == DURATION and GOAL_RADIUS == 5.0


def test_reynolds_is_package_asset(tmp_path):
    """Рейнольдс — вбудована модель пакета (simulator/assets/reynolds.onnx); у CLI — «reynolds» замість шляху."""
    from simulator import engine
    assert engine.REYNOLDS == ROOT / "simulator" / "assets" / "reynolds.onnx" and engine.REYNOLDS.exists()
    assert run("check", "reynolds").returncode == 0
    sp = save_scenarios(make_scenarios(3, 1, n_fixed=3), tmp_path / "s.json")
    r = run("fly", "--model", "reynolds", "--teacher", "reynolds", "--scenarios", sp, "-o", tmp_path / "r.parquet",
            "--duration", "1", "--quiet")
    assert r.returncode == 0, r.stderr
