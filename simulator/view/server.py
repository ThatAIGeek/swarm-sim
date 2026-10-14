"""Візуальний режим симулятора: 3D-пульт у браузері (сторінка simulator/view/page.html).

    swarm-sim view --model best.onnx --variant 2              # живий режим: епізоди рахує цей процес
    swarm-sim view --variant 2 --teacher-mode                 # викладач: еталонні моделі й приховані сценарії
    swarm-sim view rec.parquet                                 # перегляд готового запису
    swarm-sim view rec.parquet --export rec.html               # самодостатня сторінка з записом (без Python)

Рушій один — той самий, що в headless (simulator/engine.py): сторінка надсилає запит (сценарій, модель, k, зміни
цілі), Python рахує епізод і повертає кадри, сторінка їх відтворює. Сервер — стандартна бібліотека (http.server),
слухає лише 127.0.0.1. У Colab пульт відкривається в комірці через проксі порту.

Захист локального сервера (від інших вкладок браузера):
- POST /api/* — лише із заголовком X-Lab4-Token: <D.token> (випадковий рядок сесії, вбудований у сторінку), інакше 403;
- Host — лише 127.0.0.1:<порт> / localhost:<порт> (захист від DNS rebinding), інакше 403; у Colab перевірку Host
  вимкнено (запити йдуть через проксі Colab з іншим Host);
- тіло запиту — до MAX_BODY (64 МБ), більше — 413; тимчасова тека завантажених моделей видаляється при виході;
- у повідомленнях про помилки — без абсолютних шляхів; 5xx пишуться в stderr з трейсбеком.
"""
from __future__ import annotations

import atexit
import base64
import json
import math
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .. import record as R
from ..core import Scenario, SimParams
from ..engine import REYNOLDS, fly
from ..modelfile import Model, ModelFormatError, sha256_file
from ..scenarios import find_variant_dir, load_scenarios, make_scenario, scenarios_sigma_y
from ..sensors import K_MAX, N_MAX_A, R_DETECT
from ..settings import DURATION
from ..stats import GOAL_RADIUS, frame_telemetry, metrics, row_errors
from .episode import b64, clamp_goal, goal_r_max, run_episode
from .maps import player_maps

MAX_BODY = 64 * 2 ** 20          # байтів у тілі POST (модель ONNX — до кількох МБ)
TOKEN_HEADER = "X-Lab4-Token"
FREE_N = (2, 16)                 # N у вільному режимі пульта (для моделі A — до N_MAX_A)

HERE = Path(__file__).resolve().parent
PAGE = HERE / "page.html"
THREE = HERE / "vendor" / "three-0.147.0"
ASSETS = HERE / "assets"
ROOT = HERE.parent.parent          # корінь репозиторію (або site-packages) — лише щоб прибрати шляхи з повідомлень

# еталонні моделі викладацького режиму: (id, файл у variant_XX/teacher/models поруч із student/, назва, група)
REFERENCE = [
    ("poly_B", "poly_B", "Поліном 2-го ст. · спостереження B", "Behavior Cloning"),
    ("bc_A", "bc_A", "BC-A · MLP, світова СК", "Behavior Cloning"),
    ("bc_B", "bc_B", "BC-B · MLP, k найближчих", "Behavior Cloning"),
    ("bc_C", "bc_C", "BC-C · Deep Sets", "Behavior Cloning"),
    *[(f"dagger_{i}", f"dagger_{i}", f"DAgger · ітерація {i}", "DAgger (D ∪)") for i in range(1, 5)],
    ("noise_C", "noise_C", "C + noise injection", "Інші"),
    ("capacity_C", "capacity_C", "BC-C · ширша мережа, довше навчання", "Інші"),
]


def logos() -> list:
    """Логотипи (assets/logo_<N> <назва>.svg|png|jpg) як data URI; порядок — за N."""
    mime = {".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    out = []
    for f in sorted(ASSETS.glob("logo_*")):
        if f.suffix.lower() in mime:
            out.append({"src": f"data:{mime[f.suffix.lower()]};base64," + base64.b64encode(f.read_bytes()).decode(),
                        "alt": f.stem[5:].lstrip("0123456789 ")})
    return out


def _script(js: str) -> str:
    return re.sub(r"</(script)", r"<\\/\1", js, flags=re.I)


def render(payload: dict) -> str:
    """Сторінка пульта з вбудованими three.js і даними."""
    p = dict(payload, logos=logos(), maps=player_maps())
    html = PAGE.read_text(encoding="utf-8")
    data = json.dumps(p, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    for key, val in (("/*__THREE__*/", _script((THREE / "three.min.js").read_text(encoding="utf-8"))),
                     ("/*__ORBIT__*/", _script((THREE / "OrbitControls.js").read_text(encoding="utf-8"))),
                     ("/*__DATA__*/null", data)):
        assert html.count(key) == 1, key
        html = html.replace(key, val)
    v = p.get("variant") or {}
    title = f"Пульт рою · варіант {v['variant_id']:02d}" if v.get("variant_id") else "Пульт рою · запис"
    return html.replace("<title>Пульт рою ЛР4</title>", f"<title>{title}</title>", 1)


def _params(sp: SimParams | None = None, duration: float = DURATION) -> dict:
    """Параметри для сторінки (D.P): поточні налаштування або (перегляд запису) — з метаданих запису."""
    sp = sp or SimParams()
    return dict(dt=sp.dt, duration=float(duration), goal_radius=GOAL_RADIUS, crash_dist=sp.crash_dist,
                zone_radius=sp.zone_radius, zone_center=list(sp.zone_center), r_neighbor=R_DETECT, k_max=K_MAX,
                goal_r_max=goal_r_max(sp), n_max_a=N_MAX_A)


class RequestError(ValueError):
    """Некоректний запит сторінки (400) — з повідомленням для людини."""


def _int(req: dict, key: str, lo: int, hi: int, default=None) -> int:
    v = req.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or int(v) != v:
        raise RequestError(f"поле {key}: потрібне ціле число, отримано {v!r}")
    if not lo <= int(v) <= hi:
        raise RequestError(f"поле {key}: має бути від {lo} до {hi}, отримано {int(v)}")
    return int(v)


def _num(req: dict, key: str, lo: float, hi: float, default=None) -> float:
    v = req.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise RequestError(f"поле {key}: потрібне скінченне число, отримано {v!r}")
    if not lo <= float(v) <= hi:
        raise RequestError(f"поле {key}: має бути від {lo:g} до {hi:g}, отримано {v}")
    return float(v)


def _vec3(v, what: str) -> list[float]:
    if not isinstance(v, (list, tuple)) or len(v) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                              or not math.isfinite(x) for x in v):
        raise RequestError(f"{what}: потрібні три скінченні числа [x, y, z], отримано {v!r}")
    return [float(x) for x in v]


def free_seed(seed: int) -> int:
    """Зерно сценарію вільного режиму з номера на сторінці (1, 2, …): не підряд, щоб сусідні номери не мали
    суміжних зерен (і потоків чисел) — сценарії незалежні."""
    return int(np.random.default_rng([int(seed), 0xF4EE]).integers(1, 2 ** 31 - 1))


def _summary(s: Scenario) -> dict:
    return dict(n_agents=int(s.n_agents), wind=[float(x) for x in s.wind], gust_std=float(s.gust_std),
                seed=int(s.seed), goal=[float(x) for x in s.goal])


# ---------------------------------------------------------------- еталонні числа (reference_metrics.json)

def _ref(x: dict | None) -> dict | None:
    if not x:
        return None
    # ключів, яких немає в старому reference_metrics.json (no_event_rate, success_ci95), — None: сторінка ставить «—»
    return {"survival": x.get("survival_per_episode"), "success_rate": x.get("success_rate"),
            "success_ci95": x.get("success_ci95"), "no_event_rate": x.get("no_event_rate"),
            "survival_time": x.get("survival_time"), "online_mse": x.get("online_mse")}


def reference_map(rm: dict) -> dict:
    """reference_metrics.json → {модель: {набір: {"all" | "1" | "2": {survival[], success_rate, …}}}}."""
    cl, lad, hid = rm.get("closed_loop", {}), rm.get("ladder", {}), rm.get("hidden", {})
    out: dict = {"reynolds": {"eval": {"all": _ref(cl.get("expert"))}, "hidden": {"all": _ref(hid.get("expert"))}}}
    for key, name in (("poly_B", "poly_B"), ("bc_A", "BC_A"), ("bc_B", "BC_B"), ("bc_C", "BC_C")):
        e = {"all": _ref(cl.get(name))}
        for k, x in lad.get(name[-1] if name.startswith("BC") else name, {}).items():
            if int(k) < 5:
                e[k] = _ref(x)
        out[key] = {"eval": e}
    out["bc_C"]["hidden"] = {"all": _ref(hid.get("BC_C"))}
    for row in rm.get("dagger", [])[1:]:
        e = {"all": _ref(row.get("closed_loop"))}
        e.update({k: _ref(x) for k, x in row.get("ladder", {}).items()})
        out[f"dagger_{row['iteration']}"] = {"eval": e}
    if rm.get("dagger"):
        out.setdefault(f"dagger_{rm['dagger'][-1]['iteration']}", {})["hidden"] = {"all": _ref(hid.get("DAgger"))}
    out["noise_C"] = {"eval": {"all": _ref(rm.get("noise_injection", {}).get("closed_loop"))},
                      "hidden": {"all": _ref(hid.get("noise_C"))}}
    out["capacity_C"] = {"eval": {"all": _ref(rm.get("capacity_control", {}).get("closed_loop"))}}
    st = rm.get("stress") or {}                       # стрес-набір (є в еталоні з цієї версії)
    for key, name in (("reynolds", "expert"), ("poly_B", "poly_B"), ("bc_A", "BC_A"), ("bc_B", "BC_B"),
                      ("bc_C", "BC_C"), ("noise_C", "noise_C"), ("capacity_C", "capacity_C")):
        out.setdefault(key, {})["stress"] = {"all": _ref(st.get(name))}
    for row in rm.get("dagger", [])[1:]:
        out.setdefault(f"dagger_{row['iteration']}", {})["stress"] = {"all": _ref(row.get("stress"))}
    out = {c: {s: {k: x for k, x in d.items() if x} for s, d in m.items()} for c, m in out.items()}
    return {c: {s: d for s, d in m.items() if d} for c, m in out.items()}


# ---------------------------------------------------------------- живий режим

class LiveApp:
    """Стан живого режиму: набори сценаріїв варіанта, моделі, вчитель (для online MSE)."""

    def __init__(self, variant: int, models=(), teacher=None, rest_model=None, teacher_mode: bool = False):
        self.variant = variant
        self.teacher_mode = teacher_mode
        vdir = find_variant_dir(variant)
        vj = vdir / "variant.json"
        self.vinfo = json.loads(vj.read_text(encoding="utf-8")) if vj.exists() else {"variant_id": variant}
        self.sigma = self.vinfo.get("sigma_y") or scenarios_sigma_y(vdir / "eval_scenarios.json")
        self.sets = {"eval": ("Публічні", load_scenarios(vdir / "eval_scenarios.json"))}
        if (vdir / "stress_scenarios.json").exists():
            self.sets["stress"] = ("Стрес", load_scenarios(vdir / "stress_scenarios.json"))
        tdir = vdir.parent / "teacher"                      # викладацька тека варіанта — поруч зі student/
        if teacher_mode and (tdir / "hidden_scenarios.json").exists():
            self.sets["hidden"] = ("Приховані", load_scenarios(tdir / "hidden_scenarios.json"))
        self.teacher = Model(teacher or REYNOLDS)
        self.rest = Model(rest_model or REYNOLDS)
        self.rey_sha = Model(REYNOLDS).sha256
        self.models: dict[str, dict] = {}
        self._add("reynolds", Model(REYNOLDS), "Рейнольдс (reynolds.onnx)", "Вчитель", kind="reynolds")
        if teacher_mode:
            for mid, f, label, group in REFERENCE:
                path = tdir / "models" / f"{f}.onnx"
                if path.exists():
                    self._add(mid, Model(path), label, group, kind="reference")
            rm = tdir / "reference_metrics.json"
            self.reference = reference_map(json.loads(rm.read_text(encoding="utf-8"))) if rm.exists() else {}
        else:
            self.reference = {}
        self.default = None
        for m in models:
            mid = self.add_file(Path(m), Path(m).name)["id"]
            self.default = self.default or mid
        if self.default is None:
            self.default = "bc_C" if "bc_C" in self.models else "reynolds"
        self.token = secrets.token_urlsafe(24)               # D.token: POST без нього — 403
        self.upload_dir = Path(tempfile.mkdtemp(prefix="swarm-view-"))
        atexit.register(self.close)
        self.lock = threading.Lock()

    def close(self):
        """Видалити тимчасову теку завантажених моделей."""
        shutil.rmtree(self.upload_dir, ignore_errors=True)

    def scrub(self, msg: str) -> str:
        """Повідомлення для сторінки — без абсолютних шляхів (тимчасова тека, репозиторій)."""
        for d in (self.upload_dir, ROOT, Path.cwd()):
            msg = msg.replace(str(d) + "/", "").replace(str(d), "")
        return msg

    def _add(self, mid: str, m: Model, label: str, group: str, kind: str, note: str = "") -> dict:
        if kind != "reynolds" and m.sha256 == self.rey_sha:
            kind = "reynolds"
        info = dict(id=mid, label=label, group=group, kind=kind, note=note, observation=m.observation, frame=m.frame)
        self.models[mid] = dict(info, model=m)
        return info

    def add_file(self, path: Path, name: str) -> dict:
        m = Model(path)
        return self._add("my:" + m.sha256[:12], m, f"{name} ({m.name})", "Мої моделі", kind="student")

    def payload(self) -> dict:
        v = {k: self.vinfo.get(k) for k in ("variant_id", "n_lo", "n_hi", "wind_max", "gust_max")}
        return dict(mode="teacher" if self.teacher_mode else "student", variant=v, P=_params(), token=self.token,
                    sets={k: {"label": f"{lab} ({len(sc)})", "scenarios": [_summary(s) for s in sc]}
                          for k, (lab, sc) in self.sets.items()},
                    controllers=[{k: v2 for k, v2 in m.items() if k != "model"} for m in self.models.values()],
                    default_ctrl=self.default, reference=self.reference)

    def _ctrl(self, req: dict) -> dict:
        c = req.get("ctrl")
        if c not in self.models:
            raise RequestError(f"невідома модель {c!r} (оновіть сторінку пульта)")
        return self.models[c]

    def _set(self, req: dict) -> list[Scenario]:
        name = req.get("set")
        if name not in self.sets:
            raise RequestError(f"невідомий набір сценаріїв {name!r}")
        return self.sets[name][1]

    def _k(self, req: dict, m: dict) -> int | None:
        if m["kind"] == "reynolds" or req.get("k") is None:
            return None
        return _int(req, "k", 1, FREE_N[1])

    def _scenario(self, req: dict, m: dict) -> tuple[Scenario, list | None]:
        if req.get("set") == "free":
            f = req.get("free") or {}
            if not isinstance(f, dict):
                raise RequestError("поле free: потрібен об'єкт {n, seed, wind, gust}")
            n_hi = N_MAX_A if m["observation"] == "A" else FREE_N[1]   # A вміщує рій до 10 дронів
            n = max(FREE_N[0], min(n_hi, _int(f, "n", 1, 1000, 5)))
            gust = _num(f, "gust", 0.0, 10.0, 2.0)
            seed = free_seed(_int(f, "seed", 0, 2 ** 31 - 2, 1))
            sc = make_scenario(seed, n, wind_max=_num(f, "wind", 0.0, 20.0, 5.0), gust_range=(min(0.5, gust), gust))
            sc.meta = {"free": True}
        else:
            scs = self._set(req)
            sc = scs[_int(req, "idx", 0, len(scs) - 1)]
            if m["observation"] == "A" and sc.n_agents > N_MAX_A:
                raise RequestError(f"модель зі спостереженням A вміщує рій до {N_MAX_A} дронів, а в цьому сценарії "
                                   f"N = {sc.n_agents}: оберіть модель B / C")
        d = _vec3(req.get("goalD") or [0, 0, 0], "зсув цілі goalD")
        if not any(d):
            return sc, None
        g0 = [float(x) for x in sc.goal]
        sc2 = Scenario(sc.n_agents, sc.start_pos, sc.start_yaw, np.array(clamp_goal([a + b for a, b in zip(g0, d)])),
                       sc.wind, sc.gust_std, sc.seed, dict(sc.meta), sc.start_vel)
        return sc2, g0

    def episode(self, req: dict) -> dict:
        m = self._ctrl(req)
        sc, goal0 = self._scenario(req, m)
        rey = m["kind"] == "reynolds"
        k = self._k(req, m)
        plan = req.get("plan") or []
        if not isinstance(plan, list):
            raise RequestError("поле plan: потрібен список змін цілі [{f, g}]")
        steps = []
        for p in plan:
            if not isinstance(p, dict):
                raise RequestError("поле plan: кожна зміна цілі — об'єкт {f: такт, g: [x, y, z]}")
            steps.append(dict(f=_int(p, "f", 0, 10 ** 6), g=_vec3(p.get("g"), "ціль у plan")))
        plan = steps
        out = run_episode(sc, m["model"], k, self.rest, None if rey else self.teacher, self.sigma,
                          plan, bool(req.get("until")), goal0)
        out["sc"]["free"] = bool(sc.meta.get("free"))
        if rey:                                              # Рейнольдс — «синій», як решта рою
            out["nn"] = b64(np.zeros(out["T"] * out["n"]), "u1")
        return out

    def batch(self, req: dict) -> dict:
        t0 = time.perf_counter()
        m = self._ctrl(req)
        sc = self._set(req)
        rey = m["kind"] == "reynolds"
        k = self._k(req, m)
        rec = fly(m["model"], sc, teacher=None if rey else self.teacher, nn_agents=k, rest_model=self.rest,
                  sigma_y=self.sigma, hz=False)
        return dict(res=batch_rows(metrics(rec, by_episode=True)["per_episode"], len(sc)),
                    ms=1000 * (time.perf_counter() - t0))

    def upload(self, data: bytes, name: str) -> dict:
        name = re.sub(r"[^\w.\- ()]+", "_", name)[:80] or "model.onnx"
        path = self.upload_dir / f"{len(list(self.upload_dir.iterdir())):03d}.onnx"
        path.write_bytes(data)
        try:
            return self.add_file(path, name)
        except (ModelFormatError, OSError) as e:
            path.unlink(missing_ok=True)
            # у повідомленні — ім'я файла користувача, а не тимчасове 000.onnx і не шлях до тимчасової теки
            raise RequestError(self.scrub(str(e)).replace(path.name, name)) from e


def batch_rows(per: dict, n: int) -> list[dict]:
    """Рядки пакетного прогону для сторінки (контракт):
    survival_time, success, no_event, goal_reach_rate, online_mse."""
    return [dict(survival_time=per["survival_time"][i], success=bool(per["success"][i]),
                 no_event=bool(per["no_event"][i]), goal_reach_rate=per["goal_reach_rate"][i],
                 online_mse=per["online_mse"][i]) for i in range(n)]


# ---------------------------------------------------------------- перегляд запису

def record_payload(path) -> dict:
    """Запис → дані пульта: епізоди з кадрами й телеметрією, таблиця метрик (усе вже пораховано).

    Метрики й телеметрія — ті самі функції simulator/stats.py, що й у swarm-sim metrics; параметри (dt, тривалість,
    зона) — з метаданих запису, а не з поточного config.yaml."""
    rec = R.load(path)
    meta, eps = rec.meta, rec.episodes
    df = rec.frame
    sp = SimParams.from_dict(meta.get("params"))
    dt = float(meta.get("dt") or sp.dt)
    duration = float(meta.get("duration") or DURATION)
    a = R.to_arrays(df, eps)
    tel = frame_telemetry(a["pos"], a["vel"], a["alive"])
    per = metrics(rec, by_episode=True)["per_episode"]
    main_is_rey = REYNOLDS.exists() and meta["model"]["sha256"] == sha256_file(REYNOLDS)
    # online MSE по тактах — з тих самих похибок рядків, що й online_mse запису (команда до шуму, σ_y за frame)
    omse_tab = {}
    r = None if main_is_rey else row_errors(rec, df)
    if r is not None:
        err, sel = r
        g = df.loc[sel, ["episode_id", "step"]].assign(err=err[sel]).groupby(["episode_id", "step"])["err"].mean()
        for (eid, k), v in g.items():
            omse_tab.setdefault(int(eid), {})[int(k)] = float(v)
    out_eps, sets = [], []
    for e, ep in enumerate(eps):
        n = ep["n_agents"]
        sc = Scenario.from_dict(ep["scenario"])
        al = a["alive"][:, e, :n]
        if al.any():
            last = int(np.nonzero(al.any(1))[0].max()) + 1
            pos, yaw = a["pos"][:last, e, :n].copy(), a["yaw"][:last, e, :n].copy()
            al = al[:last]
            minsep, phi = tel["min_sep"][:last, e], tel["phi"][:last, e]
        else:                                             # у записі немає жодного рядка епізоду — показати старт
            last = 1
            pos, yaw = sc.start_pos[None].astype(float), sc.start_yaw[None].astype(float)
            al = np.ones((1, n), bool)
            minsep, phi = np.full(1, np.inf), np.zeros(1)
        for k in range(1, last):                          # вибулі дрони лишаються на місці події
            dead = ~al[k]
            pos[k, dead], yaw[k, dead] = pos[k - 1, dead], yaw[k - 1, dead]
        nn = (a["model"][:last, e, :n] == 0) & al & (not main_is_rey) if len(a["model"]) else np.zeros((last, n), bool)
        omse = np.full(last, np.nan)
        for k, v in omse_tab.get(int(ep["episode_id"]), {}).items():
            if k < last:
                omse[k] = v
        sub = df[(df.episode_id == ep["episode_id"]) & (df.status != 1)]
        events = [dict(t=round((int(r.step) + 1) * dt, 6), i=int(r.agent_id), kind="crash" if r.status == 2 else "lost")
                  for r in sub.itertuples()]
        goal = [float(x) for x in ep["scenario"]["goal"]]
        dg = np.linalg.norm(pos - np.asarray(goal), axis=-1) < GOAL_RADIUS
        arrive_each = [int(np.nonzero(dg[:, i] & al[:, i])[0][0]) if (dg[:, i] & al[:, i]).any() else None
                       for i in range(n)]
        reached = np.maximum.accumulate(dg, axis=0)
        all_ok = [k for k in range(last) if al[k].any() and reached[k][al[k]].all()]
        fe = ep["first_event"]
        m = {k2: per[k2][e] for k2 in ("survival_time", "success", "no_event", "min_separation", "order_parameter",
                                       "goal_reach_rate", "formation_std", "online_mse", "online_mse_pre")}
        m["inference_hz"] = meta.get("inference_hz")
        out_eps.append(dict(
            # dur = last·dt: такт k триває [k·dt, (k+1)·dt), тож 600 записаних тактів — 30,0 с (як у живому режимі)
            n=n, T=last, dt=dt, dur=round(last * dt, 9), zoneR=sp.zone_radius, edited=False, goals=[goal],
            arrive=all_ok[0] if all_ok else None, arriveEach=arrive_each, events=events,
            crashT={str(x["i"]): x["t"] for x in events if x["kind"] == "crash"}, firstEvent=fe,
            sc=dict(_summary(sc), goal0=None, free=False), m=m,
            pos=b64(pos, "<f4"), yaw=b64(yaw, "<f4"), alive=b64(al, "u1"), nn=b64(nn, "u1"),
            gi=b64(np.zeros(last), "u1"), minsep=b64(minsep, "<f4"), phi=b64(phi, "<f4"), omse=b64(omse, "<f4")))
        sets.append(_summary(sc))
    batch = dict(res=batch_rows(per, len(eps)), ms=0)
    mm = meta["model"]
    kind = "reynolds" if main_is_rey else "student"
    label = mm["name"] + (f" · {meta['nn_agents']} дронів моделі" if meta.get("nn_agents") else "")
    return dict(mode="record", title=f"Запис {Path(path).name}: {mm['name']}", variant=None, P=_params(sp, duration),
                sets={"record": {"label": f"Запис ({len(eps)})", "scenarios": sets}},
                controllers=[dict(id="rec", label=label, group="Запис", kind=kind, note="",
                                  observation=mm["observation"], frame=mm["frame"])],
                default_ctrl="rec", default_k=meta.get("nn_agents"), reference={},
                static=dict(episodes=out_eps, batch=batch))


# ---------------------------------------------------------------- HTTP

def make_handler(html: str, app: LiveApp | None, check_host: bool = True):
    """Обробник HTTP. check_host — пускати лише Host 127.0.0.1:<порт> / localhost:<порт> (у Colab — False)."""
    class H(BaseHTTPRequestHandler):
        def log_request(self, code="-", size="-"):          # успішні запити — тихо; помилки — у stderr
            if isinstance(code, int) and code >= 400 or str(code)[:1] in ("4", "5"):
                super().log_request(code, size)

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            body = json.dumps(obj, ensure_ascii=False, allow_nan=False, default=_json_default).encode()
            self._send(code, body, "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            if not check_host:
                return True
            port = self.server.server_address[1]
            return (self.headers.get("Host") or "").lower() in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "заборонено: пульт відкривається лише як http://127.0.0.1:<порт>/"})
            if self.path.split("?")[0] in ("/", "/index.html"):
                self._send(200, html.encode(), "text/html; charset=utf-8")
            else:
                self._json(404, {"error": "немає такої сторінки"})

        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {"error": "заборонено: невідомий Host"})
            if app is None:
                return self._json(400, {"error": "це перегляд запису: симулятор не рахує нових епізодів"})
            got = (self.headers.get(TOKEN_HEADER) or "").encode("utf-8", "surrogateescape")
            if not secrets.compare_digest(got, app.token.encode()):           # bytes: не-ASCII не валить порівняння
                return self._json(403, {"error": "заборонено: немає токена сесії пульта (оновіть сторінку)"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._json(400, {"error": "некоректний Content-Length"})
            if n < 0:
                return self._json(400, {"error": "некоректний Content-Length"})
            if n > MAX_BODY:
                return self._json(413, {"error": f"запит завеликий: {n / 2 ** 20:.0f} МБ "
                                                 f"(межа — {MAX_BODY // 2 ** 20} МБ)"})
            body = self.rfile.read(n)
            route, _, qs = self.path.partition("?")
            try:
                if route.endswith("/api/episode"):
                    req = _json_body(body)
                    with app.lock:
                        return self._json(200, app.episode(req))
                if route.endswith("/api/batch"):
                    req = _json_body(body)
                    with app.lock:
                        return self._json(200, app.batch(req))
                if route.endswith("/api/model"):
                    from urllib.parse import parse_qs, unquote
                    name = unquote(parse_qs(qs).get("name", ["model.onnx"])[0])
                    with app.lock:
                        return self._json(200, app.upload(body, name))
                return self._json(404, {"error": "невідомий запит"})
            except KeyError as e:
                return self._json(400, {"error": f"у запиті бракує поля {e.args[0]!r}"})
            except (ValueError, IndexError) as e:            # RequestError, ModelFormatError, A при N > 10, …
                return self._json(400, {"error": app.scrub(str(e))})
            except Exception as e:                                   # noqa: BLE001 — показати в пульті
                sys.stderr.write(f"swarm-sim view: помилка обробки {route}\n")
                traceback.print_exc(file=sys.stderr)
                return self._json(500, {"error": app.scrub(f"{type(e).__name__}: {e}")})
    return H


def _json_body(body: bytes) -> dict:
    try:
        req = json.loads(body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RequestError("тіло запиту — не JSON") from e
    if not isinstance(req, dict):
        raise RequestError("тіло запиту має бути JSON-об'єктом")
    return req


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x).__name__)


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def view(record=None, model=None, variant=None, teacher=None, rest_model=None, models=(), port: int = 8765,
         open_browser: bool = True, export=None, teacher_mode: bool = False, block: bool | None = None):
    """Відкрити пульт. record — запис *.parquet (перегляд); інакше живий режим для variant (і моделей model, models).
    export — зберегти самодостатню сторінку (лише для запису) і вийти. У Colab сервер працює у фоні, пульт
    вбудовується в комірку. Повертає сервер (або шлях до сторінки для export)."""
    if export and record is None:
        raise ValueError("--export — для запису: swarm-sim view rec.parquet --export page.html")
    if not 0 <= int(port) <= 65535:
        raise ValueError(f"порт має бути 1…65535 (0 — будь-який вільний), отримано {port}")
    if record is not None:
        payload, app = record_payload(record), None
    else:
        if variant is None:
            raise ValueError("для живого режиму потрібен --variant (або передайте запис: swarm-sim view rec.parquet)")
        app = LiveApp(variant, [m for m in [model, *models] if m], teacher, rest_model, teacher_mode)
        payload = app.payload()
    html = render(payload)
    if export:
        export = Path(export)
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text(html, encoding="utf-8")
        print(f"збережено: {export} ({len(html.encode()) / 1e6:.1f} МБ)")
        return export
    colab = _in_colab()
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(html, app, check_host=not colab))
    srv.app = app                                          # type: ignore[attr-defined] — для тестів і Colab
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    if block is None:
        block = not colab
    if colab:
        from google.colab import output  # type: ignore
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        output.serve_kernel_port_as_iframe(srv.server_address[1], height=820)
        return srv
    print(f"Пульт рою: {url}  (Ctrl+C — зупинити)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    if not block:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nзупинено")
    finally:
        srv.server_close()
        if app is not None:
            app.close()
    return srv
