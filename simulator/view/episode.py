"""Один епізод для візуального режиму: той самий рушій, що й fly(), плюс те, що потрібно пульту.

- зміни цілі під час польоту (план [(такт, ціль)]); зона польоту розширюється до найдальшої цілі;
- «до прильоту»: після DURATION (30 с) симуляція триває, доки весь живий рій не побуває біля поточної цілі
  (+ 4 с), не довше 20 хв;
- телеметрія по тактах: мін. відстань, φ, online MSE (модель проти вчителя) — для графіків;
- метрики епізоду за всю симуляцію.
Дії такту — engine.step_actions, телеметрія й метрики — функції simulator/stats.py (frame_telemetry,
flight_metrics, action_error): ті самі, що в fly + swarm-sim metrics, тож числа пульта й headless збігаються
(σ_y — за системою дії моделі, frame).
Кадри: стан до кожного такту і кінцевий стан (T + 1 кадр).
"""
from __future__ import annotations

import base64

import numpy as np

from ..core import CRASHED, Scenario, SimParams, Simulator
from ..engine import step_actions
from ..geometry import world_to_body
from ..modelfile import Model
from ..settings import DURATION, n_steps
from ..stats import GOAL_RADIUS, action_error, flight_metrics, frame_telemetry, sigma_for

MAX_DURATION = 1200.0
TAIL_S = 4.0
ZONE_PAD = 40.0
GOAL_R_ZONES = 4.0               # ціль — не далі 4 радіусів зони від її центру (1000 м при зоні 250 м)
GOAL_Z = (3.0, 70.0)             # висота цілі, м (нижче стелі зони z_max)

__all__ = ["run_episode", "clamp_goal", "goal_r_max", "b64", "GOAL_RADIUS"]


def b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def goal_r_max(params: SimParams | None = None) -> float:
    """Найбільша горизонтальна відстань цілі від центру зони, м (межа зміни цілі в пульті)."""
    return GOAL_R_ZONES * (params or SimParams()).zone_radius


def clamp_goal(g, params: SimParams | None = None) -> list[float]:
    """Ціль у розумних межах: скінченні числа, ≤ goal_r_max від центру зони, висота 3–70 м."""
    g = [float(x) for x in g]
    if len(g) != 3 or not all(np.isfinite(g)):
        raise ValueError(f"ціль має бути трьома скінченними числами [x, y, z], отримано {g}")
    p = params or SimParams()
    zc = p.zone_center
    dx, dy = g[0] - zc[0], g[1] - zc[1]
    r, rmax = float(np.hypot(dx, dy)), goal_r_max(p)
    k = rmax / r if r > rmax else 1.0
    return [zc[0] + dx * k, zc[1] + dy * k, max(GOAL_Z[0], min(GOAL_Z[1], g[2]))]


def run_episode(sc: Scenario, model: Model, k: int | None, rest: Model | None, teacher: Model | None,
                sigma: dict | None, goal_plan=(), until_arrive: bool = False, goal0=None) -> dict:
    """sc.goal — початкова ціль (можливо, зсунута); goal0 — стандартна ціль сценарію (до неї прив'язана карта)."""
    plan = sorted(((int(p["f"]), clamp_goal(p["g"])) for p in goal_plan), key=lambda x: x[0])
    goals = [list(map(float, sc.goal))]
    edited = goal0 is not None or bool(plan)
    base = SimParams()
    zone_r = base.zone_radius
    if edited:
        zc = base.zone_center
        zone_r = max([zone_r] + [float(np.hypot(g[0] - zc[0], g[1] - zc[1])) + ZONE_PAD
                                 for g in goals + [g for _, g in plan]])
    params = SimParams(**{**base.__dict__, "zone_radius": zone_r})
    sim = Simulator([sc], params)
    dt, n = params.dt, sc.n_agents
    T_std = n_steps(DURATION, dt)
    T_max = n_steps(MAX_DURATION, dt) if until_arrive else T_std
    tail = n_steps(TAIL_S, dt)
    sig = sigma_for(sigma, model.frame)                    # σ_y за системою дії моделі — як у swarm-sim metrics
    sig = np.ones(3) if sig is None else sig
    pos_f, vel_f, yaw_f, alive_f, nn_f, gi_f = [], [], [], [], [], []
    err_sum, err_cnt = [], []                              # online MSE по тактах: сума і кількість рядків моделі
    reached_all = np.zeros(n, bool)
    arrive_each: list[int | None] = [None] * n
    arrive = None
    gi, pi, reach_start = 0, 0, 0
    t_inf, n_inf = 0.0, 0
    events, prev_status = [], sim.state.status[0].copy()

    def snap(k_):
        nonlocal arrive
        st = sim.state
        p, al = st.pos[0, :n].copy(), st.alive[0, :n].copy()
        near = np.linalg.norm(p - np.asarray(goals[gi]), axis=1) < GOAL_RADIUS
        reached_all[near] = True
        for i in np.nonzero(near & al)[0]:
            if arrive_each[i] is None:
                arrive_each[i] = k_
        if al.any() and reached_all[al].all() and arrive is None:
            arrive = k_
        pos_f.append(p); vel_f.append(st.vel[0, :n].copy()); yaw_f.append(st.yaw[0, :n].copy())
        alive_f.append(al); gi_f.append(gi)

    k_ = 0
    while True:
        while pi < len(plan) and plan[pi][0] <= k_:
            sim.state.goal[0] = plan[pi][1]
            goals.append(plan[pi][1]); gi = len(goals) - 1
            reached_all[:] = False; arrive = None; reach_start = k_       # метрика цілі — від останньої зміни
            pi += 1
        snap(k_)
        st = sim.state
        if k_ >= T_max or not st.alive.any():
            nn_f.append(np.zeros(n, bool))
            break
        acts = step_actions(st, model, rest, k, teacher, {})
        u, main_sel = acts["u"], acts["main_sel"]
        t_inf += acts["t_main"]; n_inf += 1
        if acts["label"] is not None and main_sel.any():
            alive = st.alive
            lab = np.zeros_like(u)
            lab[alive] = acts["label"]
            diff_u, diff_l = u[main_sel], lab[main_sel]
            if model.frame == "body":
                yaw = st.yaw[main_sel]
                diff_u, diff_l = world_to_body(diff_u, yaw), world_to_body(diff_l, yaw)
            e = action_error(diff_u, diff_l, sig)
            err_sum.append(float(e.sum())); err_cnt.append(len(e))
        else:
            err_sum.append(0.0); err_cnt.append(0)
        nn_f.append(main_sel[0, :n].copy())
        sim.step(u)
        for i in np.nonzero((prev_status != sim.state.status[0]) & (prev_status == 1))[0]:
            if i < n:
                events.append(dict(t=round(sim.state.t, 6), i=int(i),
                                   kind="crash" if sim.state.status[0, i] == CRASHED else "lost"))
        prev_status = sim.state.status[0].copy()
        k_ += 1
        if k_ >= T_std and arrive is not None and k_ >= arrive + tail and (pi >= len(plan) or plan[pi][0] > k_):
            snap(k_); nn_f.append(nn_f[-1].copy())
            break
    # метрики — ті самі функції, що й у swarm-sim metrics, на кадрах до кожного такту (як рядки запису)
    T = len(err_sum)
    pos = np.stack(pos_f[:T])[:, None] if T else np.zeros((0, 1, n, 3))
    vel = np.stack(vel_f[:T])[:, None] if T else np.zeros((0, 1, n, 3))
    alive = np.stack(alive_f[:T])[:, None] if T else np.zeros((0, 1, n), bool)
    goal = np.asarray([goals[g] for g in gi_f[:T]], dtype=np.float64)[:, None] if T else np.zeros((0, 1, 3))
    fe = float(sim.first_event[0])
    tel = frame_telemetry(pos, vel, alive)
    per = flight_metrics(pos, vel, alive, goal, [n], [fe], T * dt, dt, min_sep_event=[float(sim.min_sep[0])],
                         reach_start=[reach_start], telemetry=tel)
    es, ec = np.asarray(err_sum), np.asarray(err_cnt)
    pre = np.arange(T) < (round(fe / dt) if np.isfinite(fe) else T + 1)
    m = {key: float(v[0]) for key, v in per.items() if key not in ("success", "no_event")}
    m.update(success=bool(per["success"][0]), no_event=bool(per["no_event"][0]),
             online_mse=float(es.sum() / ec.sum()) if ec.sum() else None,
             online_mse_pre=float(es[pre].sum() / ec[pre].sum()) if ec[pre].sum() else None,
             inference_hz=n_inf / t_inf if t_inf > 0 else None)
    omse = np.where(ec > 0, es / np.maximum(ec, 1), np.nan)
    crash_t = {str(e["i"]): e["t"] for e in events if e["kind"] == "crash"}
    dur = (len(pos_f) - 1) * dt
    return dict(
        n=n, T=len(pos_f), dt=dt, dur=dur, zoneR=zone_r, edited=edited, goals=goals, arrive=arrive,
        arriveEach=arrive_each, events=events, crashT=crash_t, firstEvent=fe if np.isfinite(fe) else None,
        sc=dict(n_agents=n, wind=[float(x) for x in sc.wind], gust_std=float(sc.gust_std), seed=int(sc.seed),
                goal=goals[0], goal0=None if goal0 is None else [float(x) for x in goal0]),
        m={k2: (None if (v2 is None or (isinstance(v2, float) and not np.isfinite(v2))) else v2) for k2, v2 in m.items()},
        pos=b64(np.stack(pos_f), "<f4"), yaw=b64(np.stack(yaw_f), "<f4"), alive=b64(np.stack(alive_f), "u1"),
        nn=b64(np.stack(nn_f[:len(pos_f)]), "u1"), gi=b64(np.array(gi_f), "u1"),
        minsep=b64(tel["min_sep"][:, 0], "<f4"), phi=b64(tel["phi"][:, 0], "<f4"), omse=b64(omse, "<f4"))
