"""Метрики польоту — рахуються із запису (record), по епізодах.

survival_time    — час до першої події в рої (зіткнення < 0,9 м, удар об землю, аварія, вихід за зону), с;
                   обрізаний тривалістю польоту (30 с), тож середнє — «обрізаний» середній час
success          — місію виконано: епізод без жодної події І кожен дрон рою хоч раз був ближче 5 м до цілі
                   (частка таких — success_rate; рій, що нікуди не летить, не зіштовхується, але й не успішний)
no_event         — епізод без жодної події, незалежно від цілі (частка — no_event_rate; діагностика)
min_separation   — мінімальна відстань між живими дронами за епізод (з кроком події), м
order_parameter  — φ = ‖Σ v̂ᵢ‖ / n: 1 — усі летять в одному напрямку; середнє за час до першої події
goal_reach_rate  — частка дронів, що хоч раз були ближче 5 м до цілі
formation_std    — розкид рою: СКВ положень дронів відносно центроїда (радіус інерції √(mean‖pᵢ − c‖²)),
                   середнє за час до першої події, м; зростає, коли рій розлітається
online_mse       — команда моделі (--model) проти мітки вчителя в станах, які відвідала сама модель, у нормованих
                   одиницях σ_y (як val MSE); є лише, якщо політ мав вчителя. З --action-noise береться команда
                   моделі до шуму (колонки cmd_*), тож шум не додається до похибки
online_mse_pre   — те саме, але лише до першої події: без хаосу після зіткнення (порівнянна між моделями)
inference_hz     — частота інференсу моделі для одного рою, тактів за секунду (з метаданих запису)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import record as R

GOAL_RADIUS = 5.0
METRICS = ["survival_time", "min_separation", "order_parameter", "goal_reach_rate", "formation_std", "online_mse",
           "online_mse_pre"]
FLAGS = ("success", "no_event")          # булеві по епізодах → частки success_rate, no_event_rate


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % довірчий інтервал Вілсона для частки k / n (для малих вибірок точніший за ± 2σ)."""
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(float(max(0.0, c - h)), 4), round(float(min(1.0, c + h)), 4)


def episode_metrics(rec: R.Record) -> dict:
    """dict назва → масив (E,) у порядку rec.episodes."""
    meta, eps = rec.meta, rec.episodes
    df = rec.load_frame()
    a = R.to_arrays(df, eps)
    out = flight_metrics(a["pos"], a["vel"], a["alive"], a["goal"], a["n_agents"], first_events(eps),
                         meta["duration"], meta["dt"], min_sep_event=min_sep_events(eps))
    out.update(online_mse=online_mse(rec, df), online_mse_pre=online_mse(rec, df, pre_event=True))
    return out


def first_events(eps: list[dict]) -> np.ndarray:
    """Час першої події кожного епізоду з метаданих запису (inf — події не було)."""
    return np.array([np.inf if e["first_event"] is None else e["first_event"] for e in eps], dtype=np.float64)


def min_sep_events(eps: list[dict]) -> np.ndarray:
    """Мінімальна відстань на кроках симуляції (зокрема на кроці зіткнення) з метаданих запису (inf — немає)."""
    return np.array([np.inf if e.get("min_sep_event") is None else e["min_sep_event"] for e in eps], dtype=np.float64)


def formation_spread(pos: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Радіус інерції рою √(mean‖pᵢ − c‖²) по живих дронах; pos (..., N, 3), alive (..., N) → (...)."""
    na = alive.sum(-1)
    cen = (pos * alive[..., None]).sum(-2) / np.maximum(na, 1)[..., None]
    d2 = ((pos - cen[..., None, :]) ** 2).sum(-1)
    return np.sqrt((d2 * alive).sum(-1) / np.maximum(na, 1))


def frame_telemetry(pos: np.ndarray, vel: np.ndarray, alive: np.ndarray) -> dict:
    """Телеметрія кожного кадру (стан перед тактом). pos, vel (T, E, N, 3), alive (T, E, N) → масиви (T, E):
    min_sep — мін. відстань між живими дронами (inf, якщо живих < 2), phi — параметр порядку φ, n_moving — скільки
    дронів рухаються (|v| > 0,1 м/с; φ має сенс за n_moving ≥ 2), spread — радіус інерції, n_alive.
    Ті самі функції рахують метрики запису (swarm-sim metrics) і пульта (view), тож числа збігаються."""
    T, E, N = alive.shape
    ms = np.full((T, E), np.inf)
    if N > 1:
        eye = np.eye(N, dtype=bool)
        for k in range(T):                                     # по кадру: (E, N, N), а не (T, E, N, N) у пам'яті
            d = np.linalg.norm(pos[k, :, :, None] - pos[k, :, None, :], axis=-1)
            ok = alive[k, :, :, None] & alive[k, :, None, :] & ~eye
            ms[k] = np.where(ok, d, np.inf).min(axis=(1, 2))
    sp = np.linalg.norm(vel, axis=-1)
    moving = alive & (sp > 0.1)
    vhat = np.where(moving[..., None], vel / np.maximum(sp, 1e-9)[..., None], 0.0)
    cnt = moving.sum(-1)
    phi = np.linalg.norm(vhat.sum(2), axis=-1) / np.maximum(cnt, 1)
    return dict(min_sep=ms, phi=phi, n_moving=cnt, spread=formation_spread(pos, alive), n_alive=alive.sum(-1))


def flight_metrics(pos: np.ndarray, vel: np.ndarray, alive: np.ndarray, goal: np.ndarray, n_agents,
                   first_event, duration: float, dt: float, min_sep_event=None, reach_start=None,
                   telemetry: dict | None = None) -> dict:
    """Метрики польоту по епізодах (див. опис модуля) з кадрів pos, vel (T, E, N, 3), alive (T, E, N).

    goal — (E, 3) або ціль кожного кадру (T, E, 3) (пульт: зміни цілі в польоті); reach_start (E,) — кадр, з якого
    рахується досягнення цілі (остання зміна цілі); first_event (E,) — inf без події; duration — тривалість польоту
    (обрізання survival_time); min_sep_event (E,) — мін. відстань на кроках симуляції (крок зіткнення)."""
    T, E, N = alive.shape
    tel = telemetry or frame_telemetry(pos, vel, alive)
    fe = np.asarray(first_event, dtype=np.float64)
    before = np.arange(T)[:, None] < np.rint(fe / dt)[None, :]                       # (T, E): кадри до події
    ms = np.full(E, np.inf) if min_sep_event is None else np.asarray(min_sep_event, dtype=np.float64)
    if T:
        ms = np.minimum(ms, tel["min_sep"].min(0))
    w = before & (tel["n_moving"] >= 2)
    order = np.where(w.sum(0) > 0, (tel["phi"] * w).sum(0) / np.maximum(w.sum(0), 1), np.nan)
    # формація: радіус інерції (СКВ положень відносно центроїда), а не СКВ відстаней — та дорівнює 0 і для двох
    # дронів за 100 м один від одного, і для рою, що розлетівся кільцем
    w2 = before & (tel["n_alive"] >= 2)
    form = np.where(w2.sum(0) > 0, (tel["spread"] * w2).sum(0) / np.maximum(w2.sum(0), 1), np.nan)
    # ціль: кожен дрон хоч раз ближче GOAL_RADIUS (живим)
    g = np.asarray(goal, dtype=np.float64)
    g = g[None] if g.ndim == 2 else g                                                # (1 | T, E, 3)
    near = (np.linalg.norm(pos - g[:, :, None, :], axis=-1) < GOAL_RADIUS) & alive
    if reach_start is not None:
        near &= np.arange(T)[:, None, None] >= np.asarray(reach_start)[None, :, None]
    n_ag = np.asarray(n_agents)
    exists = np.arange(N)[None, :] < n_ag[:, None]
    reach = (near.any(0) & exists).sum(1) / np.maximum(n_ag, 1)
    no_event = np.isinf(fe)
    return dict(survival_time=np.minimum(fe, duration), success=no_event & (reach >= 1.0 - 1e-9),
                no_event=no_event, min_separation=ms, order_parameter=order, goal_reach_rate=reach,
                formation_std=form)


def sigma_for(sigma, frame: str) -> np.ndarray | None:
    """σ_y системи дії моделі (frame = body | world) зі словника {"body": [3], "world": [3]} (variant.json)."""
    if isinstance(sigma, dict) and sigma.get(frame) is not None:
        return np.asarray(sigma[frame], dtype=np.float64)
    return None


def action_error(cmd: np.ndarray, label: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Нормована похибка дії mean(((cmd − label) / σ_y)²) по осях; (..., 3) → (...). Нескінченна (модель видала
    NaN / inf) → inf: такий рядок не відкидається мовчки."""
    err = (((np.asarray(cmd, np.float64) - np.asarray(label, np.float64)) / s) ** 2).mean(-1)
    return np.where(np.isfinite(err), err, np.inf)


def sigma_y(meta: dict, frame: str, df: pd.DataFrame | None = None):
    """σ_y для нормування online MSE: з метаданих запису (variant.json), інакше — СКВ міток у самому записі."""
    s = sigma_for(meta.get("sigma_y"), frame)
    if s is not None:
        return s, "variant"
    cols = R.LABEL_COLS if frame == "body" else R.LABEL_WORLD_COLS
    if df is not None and cols[0] in df:
        return df[cols].to_numpy(np.float64).std(0) + 1e-9, "record"
    return None, None


def _err_cols(df: pd.DataFrame, frame: str):
    """Колонки (дія моделі, мітка вчителя) у системі моделі; команда до шуму (cmd_*), якщо вона є в записі."""
    if frame == "body":
        return (R.CMD_COLS if R.CMD_COLS[0] in df else R.ACT_COLS), R.LABEL_COLS
    return (R.CMD_WORLD_COLS if R.CMD_WORLD_COLS[0] in df else R.ACT_WORLD_COLS), R.LABEL_WORLD_COLS


def row_errors(rec: R.Record, df: pd.DataFrame | None = None, pre_event: bool = False):
    """Нормована похибка кожного рядка запису (action_error) і маска рядків, де вона рахується (дрони --model;
    з pre_event — лише такти до першої події). None, якщо в записі немає вчителя."""
    df = rec.load_frame() if df is None else df
    if "label_x" not in df:
        return None
    frame = rec.meta["model"]["frame"]
    a_cols, l_cols = _err_cols(df, frame)
    s, _ = sigma_y(rec.meta, frame, df)
    sel = df.model.to_numpy() == 0
    if pre_event:
        fe = {e["episode_id"]: (np.inf if e["first_event"] is None else e["first_event"]) for e in rec.episodes}
        lim = df.episode_id.map(fe).to_numpy(np.float64)
        sel &= df.step.to_numpy() * rec.meta["dt"] < lim - 1e-9
    return action_error(df[a_cols].to_numpy(np.float64), df[l_cols].to_numpy(np.float64), s), sel


def online_mse(rec: R.Record, df: pd.DataFrame | None = None, pre_event: bool = False) -> np.ndarray:
    """По епізодах: mean(((дія − мітка) / σ_y)²) по рядках дронів, якими керувала --model; NaN без вчителя.

    pre_event — лише такти до першої події епізоду. Нескінченна похибка (модель видала NaN / inf) не
    відкидається мовчки: такий епізод має online MSE = inf."""
    eps = rec.episodes
    df = rec.load_frame() if df is None else df
    r = row_errors(rec, df, pre_event)
    if r is None:
        return np.full(len(eps), np.nan)
    err, sel = r
    per = pd.Series(err[sel]).groupby(df.episode_id.to_numpy()[sel]).mean()
    return np.array([per.get(e["episode_id"], np.nan) for e in eps])


def summarize(per_ep: dict) -> dict:
    """Середнє, СКВ, мін, макс по епізодах для кожної метрики (NaN пропускаються; inf — ні: епізод, де модель
    видала NaN, робить середнє online MSE нескінченним, а не зникає)."""
    out = {}
    for k, v in per_ep.items():
        if k in FLAGS:
            continue
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        out[k] = None if len(v) == 0 else dict(mean=_r4(v.mean()), std=_r4(v.std()) if np.isfinite(v).all() else None,
                                               min=_r4(v.min()), max=_r4(v.max()))
    return out


def _r4(x) -> float:
    x = float(x)
    return round(x, 4) if np.isfinite(x) else x


def metrics(rec, by_episode: bool = False) -> dict:
    """Повний набір метрик запису (Record або шлях до *.parquet)."""
    if not isinstance(rec, R.Record):
        rec = R.load(rec)
    per = episode_metrics(rec)
    m = rec.meta
    res = dict(model=m["model"]["name"], teacher=(m.get("teacher") or {}).get("name"),
               rest_model=(m.get("rest_model") or {}).get("name"), nn_agents=m.get("nn_agents"),
               action_noise=m.get("action_noise"), n_episodes=len(rec.episodes),
               success_rate=round(float(np.mean(per["success"])), 4),
               success_ci95=wilson(int(np.sum(per["success"])), len(rec.episodes)),
               no_event_rate=round(float(np.mean(per["no_event"])), 4), summary=summarize(per),
               inference_hz=m.get("inference_hz"))
    if "label_x" in rec.load_frame().columns:
        res["sigma_y_source"] = sigma_y(m, m["model"]["frame"], rec.load_frame())[1]
    if by_episode:
        res["per_episode"] = {k: ([bool(x) for x in v] if k in FLAGS else
                                  [None if not np.isfinite(float(x)) else round(float(x), 4) for x in v])
                              for k, v in per.items()}
        res["episode_id"] = [e["episode_id"] for e in rec.episodes]
    return res


def print_metrics(res: dict, title: str | None = None):
    s = res["summary"]
    print(f"\n--- {title or res['model']}: {res['n_episodes']} епізодів"
          + (f", модель керує {res['nn_agents']} дронами, решта — {res['rest_model']}" if res.get("nn_agents") else "")
          + " ---")
    for k in METRICS:
        if s.get(k):
            sd = "" if s[k]["std"] is None else f" ± {s[k]['std']:.4f}"
            print(f"  {k:16s} {s[k]['mean']:9.4f}{sd}")
    ci = res.get("success_ci95")
    ci = f", 95 % ДІ [{ci[0]:.2f}; {ci[1]:.2f}]" if ci else ""
    print(f"  {'success_rate':16s} {res['success_rate']:9.4f}   (без подій і весь рій долетів до цілі{ci})")
    print(f"  {'no_event_rate':16s} {res['no_event_rate']:9.4f}   (лише без зіткнень, землі й виходу із зони)")
    if res.get("inference_hz"):
        print(f"  {'inference_hz':16s} {res['inference_hz']:9.1f}")
