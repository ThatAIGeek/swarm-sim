# swarm-sim — симулятор рою квадрокоптерів

Окрема програма й бібліотека (`python -m simulator` / `swarm-sim`): модель ONNX + сценарії → запис (parquet), метрики,
3D-пульт. Публічний репозиторій: тут лише симулятор — навчальні матеріали, дані варіантів і оцінювання лабораторної
роботи живуть в окремому репозиторії курсу, який підключає цей як git-підмодуль.

## Структура

- `simulator/` — `core.py` (фізика 20 Гц), `sensors/` (`neighbors_detector`, `self_sensor`, спостереження A/B/C),
  `modelfile.py` (формат моделі lab4-swarm-model/1, `check`, `export_torch`, `export_sklearn`, відбиток поведінки),
  `engine.py` (`fly`: ролі model / rest_model / teacher, шум дії; `REYNOLDS` — вбудована модель
  `assets/reynolds.onnx`), `record.py`, `stats.py` (метрики із запису), `scenarios.py` (`find_variant_dir` —
  `variant_XX/student` від поточної теки), `settings.py` + `config.yaml`, `view/` (3D-пульт: `server.py`, `page.html`)
- `tools/` — `map_tiles.py`, `map_vector.py`: ресурси сцени «Кремль» (`simulator/view/assets/maps/`)
- `examples/` — приклад `variant.json`; `tests/` — pytest (`conftest.py` створює дані варіанта 1 у тимчасовій теці)

## Конвенції

- Python ≥ 3.11, type hints; залежності бібліотеки — діапазони (`pyproject.toml`), для розробки — `uv.lock`
- Детермінізм через `numpy.random.default_rng`; кожен епізод — власний потік зерен (результат не залежить від пакета)
- Документація, коментарі, коміти — українською
- Модель — файл ONNX з метаданими `lab4.observation` (A|B|C) і `lab4.frame` (world|body); вихід — v_cmd у м/с.
  Симулятор не виконує Python-код моделі
- `assets/reynolds.onnx` — вбудована модель Рейнольдса; збирається в репозиторії курсу й копіюється сюди,
  формулу тут не відтворювати
- Візуальний режим рахує той самий рушій, що й headless: окремої JS-фізики немає

## Команди

```bash
uv sync && uv run pytest tests/ -q
uv run swarm-sim fly --model reynolds --scenarios eval.json -o rec.parquet
uv run swarm-sim view --model best.onnx --variant 1          # дані: variant_01/student/ у поточній теці
uv build --wheel                                             # dist/swarm_sim-*.whl
```

## Параметри

Рейнольдс (вбудована модель): w_sep=2.0, w_ali=1.0, w_coh=0.8, w_mig=7.0, r_arr=10.0, r_sep=4.0, r_neighbor=15.0,
v_max=8.0, k_max=8. Симулятор: τ_v=0.35 с, κ_w=0.25 1/с, пориви OU τ_g=1 с, k_ψ=3 1/с, ω_max=1.5 рад/с,
зіткнення < 0.9 м, зона 250 м.
