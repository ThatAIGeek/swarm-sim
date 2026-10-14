# swarm-sim — симулятор рою квадрокоптерів

Окрема програма (`swarm-sim`, або `python -m simulator`): на вхід — **модель керування у форматі ONNX** і
**сценарії** (стартові умови), на виході — **запис польоту** (parquet) і метрики. Два режими: **headless**
(`fly`, `metrics`) і **візуальний** (`view` — 3D-пульт у браузері; рахує той самий Python-рушій, окремої
JS-фізики немає). Розроблено для лабораторної роботи з імітаційного навчання (клонування контролера Рейнольдса,
covariate shift, DAgger), але від навчального коду не залежить.

- **Модель — файл ONNX** із метаданими `lab4.observation` (A | B | C — яке спостереження подає симулятор) і
  `lab4.frame` (world | body); вихід — `v_cmd` у м/с (нормалізація вшита в граф). Будь-яка: поліноміальна регресія
  чи ліс (scikit-learn → skl2onnx), нейромережа (PyTorch → torch.onnx). Симулятор не виконує Python-код моделі.
- **Контролер Рейнольдса — вбудована модель** (`simulator/assets/reynolds.onnx`, `simulator.REYNOLDS`; у CLI —
  слово `reynolds` замість шляху). Він же **вчитель**: `--teacher` у тих самих станах пише свою дію (`label_*`),
  але не керує.
- **Сенсори**: `neighbors_detector` (до 8 найближчих сусідів у 15 м, Δp і Δv у body frame, маска) і `self_sensor`
  (власна швидкість, напрям на ціль, нормована відстань). Спостереження: A — 63 числа у світовій системі (слоти
  інших дронів за номером), B — ті самі сенсори одним вектором, C — `neighbors (8, 6)`, `mask (8,)`,
  `self_state (7,)`.
- **Фізика** 20 Гц: контур швидкості (τ_v = 0,35 с), вітер і пориви (Орнштейн — Уленбек), курс, зіткнення < 0,9 м,
  зона 250 м. Детермінізм: кожен епізод — власний потік зерен, результат не залежить від складу пакета.

## Встановлення

Python ≥ 3.11. Як програма й бібліотека:

```bash
pip install "swarm-sim[export] @ git+https://github.com/ThatAIGeek/swarm-sim.git"   # [export] — torch, scikit-learn, skl2onnx
swarm-sim --help
```

Для розробки — [uv](https://docs.astral.sh/uv/):

```bash
uv sync                        # .venv із закріпленими версіями (uv.lock), пакет — editable
uv run swarm-sim --help
uv run pytest                  # тести
uv build --wheel               # dist/swarm_sim-*.whl (модель Рейнольдса всередині)
```

## Команди

```bash
swarm-sim scenarios --variant 1 --set eval -o eval.json               # готовий набір варіанта (variant_01/student/)
swarm-sim scenarios --variant 1 --seed 7001 --n 100 -o s1.json        # нові стартові умови з параметрами варіанта
swarm-sim fly --model reynolds --scenarios eval.json -o expert.parquet           # Рейнольдс — верхня межа
swarm-sim fly --model best.onnx --teacher reynolds --scenarios eval.json -o best.parquet
swarm-sim fly --model reynolds --teacher best.onnx --scenarios eval.json -o shadow.parquet
                                           # «тінь»: летить Рейнольдс, модель лише пише мітки (online MSE на його станах)
swarm-sim fly --model best.onnx --teacher reynolds --nn-agents 2 --scenarios eval.json -o k2.parquet
                                           # змішаний рій: 2 дрони за моделлю, решта — за --rest-model (типово Рейнольдс)
swarm-sim fly --model reynolds --teacher reynolds --action-noise 2 --scenarios s1.json --episode-offset 200000 \
              -o noise.parquet                                                   # noise injection
swarm-sim metrics best.parquet [--by-episode] [--json out.json]
swarm-sim check best.onnx                                             # файл відповідає формату
swarm-sim export best.parquet --csv best.csv --episode 0              # фрагмент запису у CSV
swarm-sim view --model best.onnx --variant 1                          # 3D-пульт (живий режим)
swarm-sim view best.parquet [--export best.html]                      # перегляд запису / самодостатня сторінка
```

Помилки (немає файлу, не той формат, неприпустимий параметр) — один рядок «ПОМИЛКА: …» у stderr, код виходу 2;
трейсбек — із `SWARM_SIM_DEBUG=1`.

**Дані варіанта.** `--variant N` шукає `variant_NN/student/` (або `data/variant_NN/student/`) у поточній теці
чи батьківській: `variant.json` — розподіл стартових умов (кількість дронів, вітер, пориви, ціль, σ_y для online
MSE), `eval_scenarios.json` / `stress_scenarios.json` — готові набори. Приклад і опис полів — `examples/`.

**Стандартні умови моделювання** — `simulator/config.yaml`: стартова область, ціль (відстань і висота від центру
старту, фіксований азимут або точка `[x, y, z]`), вітер і пориви, фізика (τ_v, κ_w, курс, зіткнення, зона), тривалість.
Свій файл поверх стандартного: `SWARM_SIM_CONFIG=мій.yaml swarm-sim …` (лише ключі, які змінюються).

## Python API

Ті самі функції, що й у CLI:

```python
from simulator import REYNOLDS, fly, metrics, scenario_set, new_scenarios, view, check
from simulator.modelfile import export_torch, export_sklearn, stamp     # модель → файл у форматі симулятора

rec = fly("best.onnx", scenario_set(1, "eval"), teacher=REYNOLDS, out="best.parquet")
print(metrics(rec)["summary"])
view(model="best.onnx", variant=1)        # у Colab вбудовує пульт у комірку
```

Список сценаріїв (на відміну від файлу) не несе σ_y варіанта: для online MSE у масштабі CLI передайте
`fly(..., sigma_y=…)` з `variant.json`. `episode_offset=` зсуває номери епізодів у записі.

**Запис** (`simulator/record.py`): рядок = дрон, живий на початку такту; спостереження (`nb*`, `self_*`),
`act_*` — виконана дія, `label_*` — дія вчителя, `model` (0 — модель, 1 — решта рою), `status`.

**Метрики** (`swarm-sim metrics`, `simulator/stats.py`): `success_rate` — місію виконано (жодної події і кожен дрон
хоч раз ближче 5 м до цілі) з 95 % ДІ Вілсона `success_ci95`; `no_event_rate` — лише без подій; `survival_time`,
`goal_reach_rate`, `min_separation`, `formation_std` (радіус інерції рою), `order_parameter`, `online_mse` і
`online_mse_pre` (до першої події) — у нормалізованих діях, `inference_hz`.

## Пульт рою (візуальний режим)

`swarm-sim view` відкриває 3D-пульт у браузері (three.js, без мережі): вибір сценарію (набори варіанта або вільний
режим), модель (Рейнольдс, своя — перетягнути `*.onnx`), повзунок змішаного рою «k з N», зміна цілі в польоті,
шари (сліди, зв'язки, небезпека), камери, живі метрики й графіки, пакетний прогін набору, ефекти (феєрверк,
вибух; лише відображення — фізика й метрики не змінюються). Моделі квадрокоптерів нахиляються в бік руху (лише
вигляд). Кожен епізод рахує Python-рушій, тож числа збігаються зі `swarm-sim metrics`. Тека `variant_NN/teacher/`
поруч зі `student/` (`--teacher-mode`) додає еталонні моделі й приховані сценарії, якщо вони там є.

**Сцена: «3D-простір» або «Кремль».** «Кремль» — карта центру Москви з даних OpenStreetMap (Overture Maps):
підкладка 1600 × 1600 м і ≈ 4,3 тис. будівель у 3D (`simulator/view/assets/maps/`); оновити —
`python -m tools.map_vector kremlin` (extra `maps`). Фізика й метрики від сцени не залежать.

## Структура

```
simulator/        core.py (фізика 20 Гц), sensors/ (neighbors_detector, self_sensor, спостереження A/B/C),
                  modelfile.py (формат ONNX, check, export_torch/export_sklearn, відбиток поведінки),
                  engine.py (fly: ролі model / rest_model / teacher, шум дії), record.py (схема запису),
                  stats.py (метрики), scenarios.py, settings.py + config.yaml, geometry.py,
                  assets/reynolds.onnx (вбудований Рейнольдс), view/ (3D-пульт: server.py, page.html, vendor/, assets/),
                  __main__.py (CLI)
tools/            map_tiles.py, map_vector.py — ресурси сцени «Кремль» для пульта
examples/         приклад variant.json і опис теки даних варіанта
tests/            test_core (фізика), test_cli (програма на поганих входах), test_program (формат, політ, запис, пульт)
```

## Ліцензія

Код — [MIT](LICENSE). Дані карти сцени «Кремль» (`simulator/view/assets/maps/`) — © учасники OpenStreetMap,
ODbL, через Overture Maps Foundation (див. `simulator/view/assets/maps/NOTICE`). three.js — MIT
(`simulator/view/vendor/three-0.147.0/LICENSE`).
