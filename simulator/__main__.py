"""swarm-sim — симулятор рою як окрема програма.

    swarm-sim scenarios --variant 2 --set eval -o eval.json             # 30 публічних сценаріїв варіанта
    swarm-sim scenarios --variant 2 --set stress -o stress.json         # стрес-набір (N = 8–10, сильніший вітер)
    swarm-sim scenarios --variant 2 --seed 7001 --n 100 -o s1.json      # нові стартові умови з параметрами варіанта
    swarm-sim fly --model M.onnx --scenarios eval.json -o rec.parquet [--teacher reynolds.onnx]
                  [--nn-agents 2 --rest-model other.onnx] [--action-noise 2.0] [--episode-offset 100000]
    swarm-sim metrics rec.parquet [--by-episode] [--json out.json]
    swarm-sim check M.onnx
    swarm-sim export rec.parquet --csv rec.csv [--episode 0]
    swarm-sim view --model M.onnx --variant 2          # візуальний режим (3D-пульт у браузері)
    swarm-sim view rec.parquet                          # переглянути готовий запис

(без встановлення: python -m simulator …; «reynolds» замість шляху — вбудований контролер Рейнольдса)

Помилки (немає файлу, не той формат, неприпустимий параметр) друкуються одним рядком «ПОМИЛКА: …» у stderr,
код виходу 2; повний трейсбек — з SWARM_SIM_DEBUG=1. `check` для моделі, що не відповідає формату, — код 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

EXIT_ERROR = 2


class UsageError(ValueError):
    """Неприпустиме поєднання параметрів командного рядка."""



def model_path(s: str) -> str:
    """Аргумент-модель: шлях до *.onnx або «reynolds» — вбудований контролер Рейнольдса (simulator.REYNOLDS)."""
    if s == "reynolds":
        from .engine import REYNOLDS
        return str(REYNOLDS)
    return s

def plural(n: int, one: str, few: str, many: str) -> str:
    """Узгодження з числом: 1 сценарій, 2 сценарії, 5 сценаріїв (11–14 — «many»)."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def _n_scenarios(n: int) -> str:
    return f"{n} {plural(n, 'сценарій', 'сценарії', 'сценаріїв')}"


def _warn(msg: str):
    print(f"УВАГА: {msg}", file=sys.stderr)


def _fmt(x, spec: str, width: int) -> str:
    """Число за форматом або «—» для None / NaN (min_separation при N = 1, online MSE без вчителя)."""
    if x is None or (isinstance(x, float) and x != x):
        return "—".rjust(width)
    return format(x, spec).rjust(width)


def _scenarios(a):
    from . import new_scenarios, scenario_set
    from .scenarios import find_variant_dir, save_scenarios

    if (a.set is None) == (a.seed is None):
        raise UsageError("потрібно або --set eval | stress, або --seed S --n N")
    vdir = find_variant_dir(a.variant)
    vj = vdir / "variant.json"
    header = dict(variant=a.variant)
    if vj.exists():
        header["sigma_y"] = json.loads(vj.read_text(encoding="utf-8")).get("sigma_y")
    if a.set is not None:
        if a.n_agents is not None:
            _warn(f"--n-agents {a.n_agents} ігнорується з --set {a.set}: готовий набір має свої N")
        sc = scenario_set(a.variant, a.set)
        header["set"] = a.set
    else:
        if a.n < 1:
            raise UsageError("--n має бути ≥ 1")
        if a.n_agents is not None and not 1 <= a.n_agents <= 127:
            raise UsageError("--n-agents має бути від 1 до 127")
        if a.seed < 0:
            raise UsageError("--seed має бути ≥ 0")
        sc = new_scenarios(a.variant, a.seed, a.n, a.n_agents)
        header.update(seed=a.seed)
    out = save_scenarios(sc, a.out, **header)
    print(f"{_n_scenarios(len(sc))} → {out}")


def _fly(a):
    from .engine import fly
    from .stats import metrics, print_metrics

    if a.nn_agents is not None and a.nn_agents < 1:
        raise UsageError("--nn-agents має бути ≥ 1")
    if a.rest_model and a.nn_agents is None:
        raise UsageError("--rest-model має сенс лише з --nn-agents k")
    rec = fly(a.model, a.scenarios, out=a.out, teacher=a.teacher, nn_agents=a.nn_agents, rest_model=a.rest_model,
              action_noise=a.action_noise, noise_seed=a.noise_seed, duration=a.duration,
              episode_offset=a.episode_offset, keep=True, progress=not a.quiet)
    print(f"запис: {a.out} ({len(rec.frame)} рядків, {_n_scenarios(len(rec.episodes))})")
    if not a.quiet:
        print_metrics(metrics(rec))


def _metrics(a):
    from .stats import metrics, print_metrics

    res = metrics(a.record, by_episode=a.by_episode)
    print_metrics(res)
    if a.by_episode:
        pe = res["per_episode"]
        print("\n  епізод  survival  min_sep  online_mse  success")
        for j, eid in enumerate(res["episode_id"]):
            print(f"  {eid:6d}  {_fmt(pe['survival_time'][j], '.2f', 8)}  {_fmt(pe['min_separation'][j], '.3f', 7)}  "
                  f"{_fmt(pe['online_mse'][j], '.4f', 10)}  {'так' if pe['success'][j] else 'ні':>7s}")
    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"збережено: {out}")


def _check(a):
    from .modelfile import ModelFormatError, check

    try:
        r = check(a.model)
    except ModelFormatError as e:                 # вердикт «не відповідає формату», а не збій програми
        print(f"ПОМИЛКА: {e}", file=sys.stderr)
        raise SystemExit(1)
    for k, v in r.items():
        print(f"  {k:14s} {v}")
    print("OK: модель відповідає формату" if r["ok"] else "НЕ OK: див. finite / inference_hz (≥ 20 Гц)")
    raise SystemExit(0 if r["ok"] else 1)


def _export(a):
    from .record import to_csv

    print("збережено:", to_csv(a.record, a.csv, a.episode))


def _view(a):
    from .view.server import view

    view(record=a.record, model=a.model, variant=a.variant, teacher=a.teacher, rest_model=a.rest_model,
         models=a.extra_model, port=a.port, open_browser=not a.no_browser, export=a.export,
         teacher_mode=a.teacher_mode)


def _port(s: str) -> int:
    try:
        p = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"порт — ціле число, отримано {s!r}") from None
    if not 1 <= p <= 65535:
        raise argparse.ArgumentTypeError(f"порт має бути від 1 до 65535, отримано {p}")
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="swarm-sim", description="Симулятор рою: модель ONNX → політ → запис",
                                 epilog="Повний трейсбек помилок: SWARM_SIM_DEBUG=1 swarm-sim …")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scenarios", help="набір сценаріїв (стартових умов) варіанта у файл JSON")
    p.add_argument("--variant", type=int, required=True, help="номер варіанта (дані — student/variant_XX/data)")
    p.add_argument("--set", choices=["eval", "stress"], default=None,
                   help="готовий набір з архіву (eval — 30 публічних, N = 5; stress — стрес-набір, N = 8–10)")
    p.add_argument("--seed", type=int, default=None, help="зерно нових сценаріїв (≥ 0)")
    p.add_argument("--n", type=int, default=100, help="скільки нових сценаріїв (типово 100)")
    p.add_argument("--n-agents", type=int, default=None,
                   help="фіксоване N дронів 1…127 (інакше — за розподілом варіанта); лише з --seed")
    p.add_argument("-o", "--out", required=True, help="файл сценаріїв *.json")
    p.set_defaults(fn=_scenarios)

    p = sub.add_parser("fly", help="пролетіти сценарії моделлю (headless) і зберегти запис")
    p.add_argument("--model", required=True, type=model_path, help="модель *.onnx, яку випробовують")
    p.add_argument("--scenarios", required=True, help="файл сценаріїв *.json (swarm-sim scenarios)")
    p.add_argument("-o", "--out", required=True, help="запис *.parquet (пишеться атомарно)")
    p.add_argument("--teacher", default=None, type=model_path, help="модель-вчитель *.onnx: пише мітки label_*, не керує")
    p.add_argument("--nn-agents", type=int, default=None, help="моделлю летять перші k ≥ 1 дронів рою")
    p.add_argument("--rest-model", default=None, type=model_path, help="модель решти дронів (типово вбудований Рейнольдс)")
    p.add_argument("--action-noise", type=float, default=0.0, help="σ шуму виконуваної дії, м/с (≥ 0; типово 0)")
    p.add_argument("--noise-seed", type=int, default=0, help="зерно шуму дії (≥ 0; типово 0)")
    p.add_argument("--episode-offset", type=int, default=0,
                   help="зсув номерів епізодів ≥ 0 (щоб не збігались між записами)")
    p.add_argument("--duration", type=float, default=None,
                   help="тривалість епізоду, с (> 0; типово — episode.duration із config.yaml, 30 с)")
    p.add_argument("--quiet", action="store_true", help="без прогресу й таблиці метрик")
    p.set_defaults(fn=_fly)

    p = sub.add_parser("metrics", help="метрики запису")
    p.add_argument("record", help="запис *.parquet (swarm-sim fly)")
    p.add_argument("--by-episode", action="store_true", help="ще й таблиця по епізодах")
    p.add_argument("--json", default=None, help="зберегти метрики у файл *.json (теку буде створено)")
    p.set_defaults(fn=_metrics)

    p = sub.add_parser("check", help="перевірити файл моделі (формат, скінченність виходу, ≥ 20 Гц)")
    p.add_argument("model", type=model_path, help="модель *.onnx")
    p.set_defaults(fn=_check)

    p = sub.add_parser("export", help="фрагмент запису у CSV")
    p.add_argument("record", help="запис *.parquet")
    p.add_argument("--csv", required=True, help="вихідний файл *.csv")
    p.add_argument("--episode", type=int, default=None, help="лише епізод із цим episode_id (інакше — увесь запис)")
    p.set_defaults(fn=_export)

    p = sub.add_parser("view", help="візуальний режим: 3D-пульт у браузері")
    p.add_argument("record", nargs="?", default=None, help="готовий запис *.parquet (перегляд)")
    p.add_argument("--model", default=None, type=model_path, help="модель *.onnx для живого режиму")
    p.add_argument("--extra-model", action="append", default=[], help="ще модель у списку пульта (можна кілька)")
    p.add_argument("--variant", type=int, default=None, help="номер варіанта (живий режим: сценарії, σ_y)")
    p.add_argument("--teacher", default=None, type=model_path, help="модель-вчитель для online MSE (типово вбудований Рейнольдс)")
    p.add_argument("--rest-model", default=None, type=model_path, help="модель решти рою в драбині (типово вбудований Рейнольдс)")
    p.add_argument("--port", type=_port, default=8765, help="порт локального сервера 1…65535 (типово 8765)")
    p.add_argument("--no-browser", action="store_true", help="не відкривати браузер автоматично")
    p.add_argument("--export", default=None, help="зберегти самодостатню сторінку з записом *.html і вийти")
    p.add_argument("--teacher-mode", action="store_true",
                   help="викладацька версія (еталонні моделі, приховані сценарії)")
    p.set_defaults(fn=_view)
    return ap


def _describe(e: BaseException) -> str:
    """Повідомлення помилки для людини (без трейсбека)."""
    if isinstance(e, IsADirectoryError):
        return f"{e.filename}: це тека, а не файл" if e.filename else str(e)
    if isinstance(e, FileNotFoundError):
        return f"немає файлу {e.filename}" if e.filename else str(e)
    if isinstance(e, PermissionError):
        return f"немає доступу до {e.filename}" if e.filename else str(e)
    if isinstance(e, KeyError):
        return f"бракує ключа {e.args[0]!r}" if e.args else "бракує ключа"
    if isinstance(e, (ValueError, OSError, RuntimeError)):
        return str(e) or type(e).__name__
    return f"{type(e).__name__}: {e}"


def main(argv=None) -> int:
    warnings.formatwarning = lambda msg, *a, **k: f"УВАГА: {msg}\n"     # ConfigWarning та ін. — одним рядком
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "fly" and a.duration is None:              # типова тривалість — з config.yaml (episode.duration)
            from .settings import DURATION
            a.duration = DURATION
        a.fn(a)
    except KeyboardInterrupt:
        print("перервано", file=sys.stderr)
        return 130
    except Exception as e:                                             # noqa: BLE001 — одна точка для всіх команд
        if os.environ.get("SWARM_SIM_DEBUG") == "1":
            raise
        print(f"ПОМИЛКА: {_describe(e)}", file=sys.stderr)
        if not isinstance(e, (OSError, ValueError, KeyError, RuntimeError)):
            print("(повний трейсбек: SWARM_SIM_DEBUG=1)", file=sys.stderr)
        return EXIT_ERROR
    return 0


if __name__ == "__main__":
    sys.exit(main())
