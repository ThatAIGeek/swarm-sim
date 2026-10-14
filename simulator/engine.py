"""Політ: модель програється в симуляторі, результат — запис (record).

Ролі моделей:
    model       — модель, яку випробовують (усі дрони, або перші nn_agents дронів кожного рою);
    rest_model  — модель для решти дронів змішаного рою (діагностична драбина), типово вбудований Рейнольдс (REYNOLDS);
    teacher     — модель-вчитель: не керує жодним дроном, лише в тих самих станах пише свою дію (label_*).

Такт (dt = 0,05 с), для всіх епізодів пакета одночасно:
    1. сенсори neighbors_detector і self_sensor → спостереження потрібного типу (A / B / C);
    2. model і rest_model видають v_cmd для своїх дронів; frame=body → поворот Rz(ψ); обрізання |v| ≤ 8 м/с;
    3. вчитель (якщо є) — мітка для всіх живих дронів, не виконується;
    4. шум дії (action_noise > 0): v_cmd += N(0, σ²), знову обрізання (noise injection);
    5. фізика (core.Simulator.step): пориви, динаміка, курс, події;
    6. рядки запису для дронів, живих на початку такту.
Епізод — DURATION = 30 с (simulator/config.yaml, episode.duration; 600 тактів по dt) або доки є живі дрони.
Кроки 1–3 — step_actions(): та сама функція рахує дії і в fly, і в пульті (view/episode.py).
Запис пишеться атомарно: у <out>.tmp, а на місце out — лише повністю записаний файл (os.replace).
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import record as R
from .core import Scenario, SimParams, Simulator
from .geometry import body_to_world, clip_norm, world_to_body
from .modelfile import Model, load_model
from .sensors import N_MAX_A, build_observation, observation_inputs
from .settings import DURATION, n_steps

# контролер Рейнольдса — вбудована модель пакета (simulator/assets/reynolds.onnx)
REYNOLDS = Path(__file__).resolve().parent / "assets" / "reynolds.onnx"
BATCH = 100
MAX_STEPS = 2 ** 15 - 1          # step у записі — int16: до 32767 тактів (≈ 1638 с при 20 Гц)
MAX_AGENTS = 127                 # agent_id і n_agents у записі — int8


def model_actions(m: Model, st, sel: np.ndarray, cache: dict) -> np.ndarray:
    """Світові команди моделі m для дронів маски sel (E, N): (M, 3), обрізані |v| ≤ 8 м/с."""
    y = m.act(observation_inputs(m.observation, st, sel, cache))
    if m.frame == "body":
        y = body_to_world(y, st.yaw[sel])
    return clip_norm(y)


_model_actions = model_actions                      # стара назва (сумісність)


def step_actions(st, main: Model, rest: Model | None, nn_agents: int | None, teacher: Model | None,
                 cache: dict) -> dict:
    """Дії одного такту для пакета (E, N): u — світові команди (до шуму дії), main_sel / rest_sel — чиї дрони,
    label — мітки вчителя для всіх живих (або None), t_main — час інференсу моделі main, с."""
    alive = st.alive
    E, N = alive.shape
    u = np.zeros((E, N, 3))
    main_sel = alive & ((np.arange(N)[None, :] < nn_agents) if nn_agents is not None else True)
    rest_sel = alive & ~main_sel
    t0 = time.perf_counter()
    if main_sel.any():
        u[main_sel] = model_actions(main, st, main_sel, cache)
    t_main = time.perf_counter() - t0
    if rest_sel.any():
        u[rest_sel] = model_actions(rest, st, rest_sel, cache)
    label = model_actions(teacher, st, alive, cache) if teacher is not None and alive.any() else None
    return dict(u=u, main_sel=main_sel, rest_sel=rest_sel, label=label, t_main=t_main)


def validate_flight(scenarios: list[Scenario], models, duration: float, dt: float, action_noise: float = 0.0,
                    noise_seed: int = 0, episode_offset: int = 0, nn_agents: int | None = None):
    """Перевірка параметрів польоту до його початку (ValueError з поясненням, а не тихе переповнення)."""
    if not scenarios:
        raise ValueError("порожній список сценаріїв")
    if not (math.isfinite(duration) and duration > 0):
        raise ValueError(f"тривалість польоту має бути > 0 с, отримано {duration}")
    T = n_steps(duration, dt)
    if T < 1:
        raise ValueError(f"тривалість {duration} с коротша за один такт ({dt} с)")
    if T > MAX_STEPS:
        raise ValueError(f"тривалість {duration} с = {T} тактів, а запис вміщує до {MAX_STEPS} "
                         f"(≈ {MAX_STEPS * dt:.0f} с)")
    if not (math.isfinite(action_noise) and action_noise >= 0):
        raise ValueError(f"σ шуму дії має бути ≥ 0 м/с, отримано {action_noise}")
    if int(noise_seed) < 0:
        raise ValueError(f"зерно шуму має бути ≥ 0, отримано {noise_seed}")
    if int(episode_offset) < 0 or int(episode_offset) + len(scenarios) > 2 ** 31 - 1:
        raise ValueError(f"зсув номерів епізодів має бути ≥ 0 (і з кількістю сценаріїв < 2³¹), "
                         f"отримано {episode_offset}")
    if nn_agents is not None and int(nn_agents) < 1:
        raise ValueError(f"nn_agents має бути ≥ 1, отримано {nn_agents}")
    n = [int(s.n_agents) for s in scenarios]
    if min(n) < 1 or max(n) > MAX_AGENTS:
        raise ValueError(f"кількість дронів у сценарії має бути 1…{MAX_AGENTS}, отримано {min(n)}…{max(n)}")
    for m in models:
        if m is not None and m.observation == "A" and max(n) > N_MAX_A:
            raise ValueError(f"модель {m.name} має спостереження A (рій до {N_MAX_A} дронів), а в сценаріях N до "
                             f"{max(n)}: візьміть сценарії з N ≤ {N_MAX_A} або модель B / C")


def _fly_batch(scen: list[Scenario], ids: np.ndarray, main: Model, rest: Model | None, nn_agents: int | None,
               teacher: Model | None, action_noise: float, noise_seed: int, duration: float,
               params: SimParams | None):
    sim = Simulator(scen, params)
    T = n_steps(duration, sim.p.dt)
    n_ag = np.array([s.n_agents for s in scen])
    # потік шуму — від зерна сценарію, а не від його позиції у файлі: перестановка сценаріїв не змінює польоту
    rngs = [np.random.default_rng([noise_seed, int(s.seed), 0x5EED]) for s in scen]
    rows = []
    for k in range(T):
        st = sim.state
        alive = st.alive.copy()
        if not alive.any():
            break
        cache: dict = {}
        acts = step_actions(st, main, rest, nn_agents, teacher, cache)
        u, lab = acts["u"], acts["label"]
        who = acts["rest_sel"].astype(np.int8)                 # 0 — --model, 1 — --rest-model
        if action_noise > 0:
            u_cmd = u.copy()                                   # команда до шуму — для online MSE
            for e, r in enumerate(rngs):                       # окремий потік на епізод: не залежить від пакета
                u[e, :n_ag[e]] += r.normal(0.0, action_noise, (n_ag[e], 3))
            u = clip_norm(np.where(alive[..., None], u, 0.0))
        if "C" in cache:
            nb, mask, ss = (a.astype(np.float32) for a in cache["C"])
        else:
            nb, mask, ss = build_observation(st.pos, st.vel, st.yaw, alive, st.goal)
        ee, ii = np.nonzero(alive)
        pos, vel, yaw = st.pos[ee, ii], st.vel[ee, ii], st.yaw[ee, ii]
        goal = st.goal[ee]
        uw = u[ee, ii]
        sim.step(u)
        n = len(ee)
        block = [pos, vel, yaw[:, None], goal, nb[ee, ii].reshape(n, -1), mask[ee, ii], ss[ee, ii],
                 world_to_body(uw, yaw), uw]
        if teacher is not None:
            block += [world_to_body(lab, yaw), lab]
        if action_noise > 0:
            uc = u_cmd[ee, ii]
            block += [world_to_body(uc, yaw), uc]
        rows.append(dict(e=ee, i=ii, k=np.full(n, k), who=who[ee, ii], status=sim.state.status[ee, ii].copy(),
                         f=np.concatenate(block, axis=1).astype(np.float32)))
    cols = R.columns(teacher is not None, cmd=action_noise > 0)
    if rows:
        e = np.concatenate([r["e"] for r in rows]); i = np.concatenate([r["i"] for r in rows])
        k = np.concatenate([r["k"] for r in rows]); f = np.concatenate([r["f"] for r in rows])
        who = np.concatenate([r["who"] for r in rows]); status = np.concatenate([r["status"] for r in rows])
        order = np.lexsort((i, k, e))                          # епізод → такт → дрон
        e, i, k, f, who, status = e[order], i[order], k[order], f[order], who[order], status[order]
    else:
        e = i = k = who = status = np.zeros(0, dtype=int)
        f = np.zeros((0, len(cols) - len(R.ID_COLS) - 1), dtype=np.float32)
    data = {"episode_id": ids[e], "step": k, "t": (k * sim.p.dt).astype(np.float32), "agent_id": i,
            "n_agents": n_ag[e], "model": who}
    fcols = [c for c in cols if c not in R.ID_COLS and c != "status"]
    for j, c in enumerate(fcols):
        data[c] = f[:, j]
    data["status"] = status
    df = pd.DataFrame(data)[cols]
    for c, t in R.INT_COLS.items():
        df[c] = df[c].astype(t)
    eps = []
    for b, s in enumerate(scen):
        fe = float(sim.first_event[b])
        eps.append(dict(episode_id=int(ids[b]), n_agents=int(s.n_agents),
                        first_event=None if not np.isfinite(fe) else round(fe, 6),
                        min_sep_event=None if not np.isfinite(sim.min_sep[b]) else float(sim.min_sep[b]),
                        final_status=[int(x) for x in sim.state.status[b, :s.n_agents]],
                        scenario=s.to_dict()))
    return df, eps


def _sigma(s):
    """σ_y: {"body": [3], "world": [3]} (СКВ дій датасету варіанта; variant.json)."""
    if s is None:
        return None
    return {k: [float(x) for x in v] for k, v in dict(s).items() if v is not None}


def measure_hz(m: Model, scenario: Scenario, steps: int = 100) -> float:
    """Частота інференсу моделі для одного рою (усі дрони — модель), тактів за секунду."""
    sim = Simulator([scenario])
    st, sel = sim.state, sim.state.alive
    for _ in range(3):
        model_actions(m, st, sel, {})
    t0 = time.perf_counter()
    for _ in range(steps):
        model_actions(m, st, sel, {})
    return steps / (time.perf_counter() - t0)


def fly(model, scenarios: list[Scenario] | str | Path, out=None, teacher=None, nn_agents: int | None = None,
        rest_model=None, action_noise: float = 0.0, noise_seed: int = 0, duration: float = DURATION,
        params: SimParams | None = None, episode_offset: int = 0, sigma_y=None, batch: int = BATCH,
        keep: bool = True, hz: bool = True, progress: bool = False) -> R.Record:
    """Пролетіти сценарії моделлю і повернути запис (і записати його в out, якщо задано).

    model, teacher, rest_model — шлях до *.onnx або Model; scenarios — список Scenario або шлях до scenarios.json.
    nn_agents = k — моделлю летять перші k дронів кожного рою, решта — rest_model (типово Рейнольдс).
    sigma_y — СКВ дій датасету варіанта {"body": [3], "world": [3]} для online MSE; keep=False — не тримати таблицю в пам'яті.
    """
    from .scenarios import load_scenarios, scenarios_sigma_y

    if not isinstance(scenarios, list):
        if sigma_y is None:
            sigma_y = scenarios_sigma_y(scenarios)
        scenarios = load_scenarios(scenarios)
    params = params or SimParams()
    validate_flight(scenarios, [], duration, params.dt, action_noise, noise_seed, episode_offset, nn_agents)
    main = load_model(model)
    rest = None
    if nn_agents is not None:
        rest = load_model(rest_model or REYNOLDS)
    tch = load_model(teacher) if teacher is not None else None
    validate_flight(scenarios, [main, rest, tch], duration, params.dt, action_noise, noise_seed, episode_offset,
                    nn_agents)
    if out is not None:
        out = Path(out)
        if out.is_dir():
            raise IsADirectoryError(f"{out}: це тека — вкажіть файл запису, наприклад {out / 'rec.parquet'}")
    ids_all = episode_offset + np.arange(len(scenarios))
    writer = None
    tmp = None
    frames, episodes = [], []
    meta = dict(format=R.FORMAT, model=main.info(), rest_model=rest.info() if rest else None,
                teacher=tch.info() if tch else None, nn_agents=nn_agents, action_noise=float(action_noise),
                noise_seed=int(noise_seed), duration=float(duration), dt=params.dt, params=params.__dict__,
                sigma_y=_sigma(sigma_y), episode_offset=int(episode_offset))
    try:
        for b0 in range(0, len(scenarios), batch):
            sl = slice(b0, b0 + batch)
            df, eps = _fly_batch(scenarios[sl], ids_all[sl], main, rest, nn_agents, tch, action_noise, noise_seed,
                                 duration, params)
            episodes += eps
            if out is not None:
                import pyarrow as pa
                import pyarrow.parquet as pq

                tbl = pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(None)
                if writer is None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_name(out.name + ".tmp")
                    writer = pq.ParquetWriter(str(tmp), tbl.schema, compression="zstd")
                writer.write_table(tbl)
            if keep:
                frames.append(df)
            if progress:
                print(f"  епізоди {b0 + len(eps)}/{len(scenarios)}", flush=True)
        meta["episodes"] = episodes
        meta["inference_hz"] = round(measure_hz(main, scenarios[0]), 1) if hz else None
        if writer is not None:
            writer.add_key_value_metadata(R.write_meta(meta))
            writer.close()
            writer = None
            os.replace(tmp, out)                               # на місці out — лише повний запис
    finally:
        if writer is not None:                                 # перервано (виняток, Ctrl+C): попередній out цілий
            writer.close()
        if tmp is not None and tmp.exists():
            tmp.unlink()
    frame = pd.concat(frames, ignore_index=True) if keep else None
    return R.Record(frame, meta, Path(out) if out is not None else None)
