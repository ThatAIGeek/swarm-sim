"""Легкий симулятор рою квадрокоптерів (без Gazebo), 20 Гц, пакетний по епізодах.

Модель кожного дрона (крок dt = 0.05 с, напівнеявний Ейлер):

    v̇ = (v_cmd − v) / τ_v  +  κ_w · (w + g_i)        # контур швидкості PX4 + вітер
    ṗ = v
    ψ̇ = clip(k_ψ · wrap(ψ_cmd − ψ), ±ω_max)            # курс повертається за горизонтальною командою

v_cmd — команда контролера у світовій системі (бажана швидкість, |v_cmd| ≤ 8 м/с);
w — постійний вітер епізоду, g_i — пориви (процес Орнштейна–Уленбека, свій для кожного дрона).
Контур швидкості компенсує вітер не повністю: стале відхилення ≈ κ_w·τ_v·w.

Події:
- зіткнення — два активні дрони ближче за crash_dist, або висота нижче ground_z (обидва / один → CRASHED);
- вихід за зону — горизонтальна відстань від центру зони > zone_radius або висота > z_max → LOST.
Дрон після події вибуває (не рухається і не є сусідом). survival_time — час до першої події в рої.

Масиви мають форму (E, N, ...): E епізодів, доповнених до спільного N (exists = False для доповнення).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import clip_norm
from .settings import DURATION, n_steps
from .settings import PHYSICS as _PH

DT = float(_PH["dt"])                 # крок, с (settings перевіряє dt > 0)
HZ = int(round(1 / DT))               # тактів за секунду (20)

PADDING, ACTIVE, CRASHED, LOST = 0, 1, 2, 3
GUST_STREAM = 0x6057              # потік поривів епізоду — default_rng([seed, GUST_STREAM]): не збігається з потоком
                                  # генерації сценарію (default_rng(seed)) ні цього, ні сусіднього зерна (seed + 1)


@dataclass(frozen=True)
class SimParams:
    """Фізичні параметри симулятора — за замовчуванням із simulator/config.yaml (розділ physics)."""
    dt: float = DT
    tau_v: float = float(_PH["tau_v"])            # стала часу контуру швидкості, с
    kappa_w: float = float(_PH["kappa_w"])        # вплив вітру на прискорення, 1/с
    gust_tau: float = float(_PH["gust_tau"])      # кореляційний час поривів, с
    k_yaw: float = float(_PH["k_yaw"])            # підсилення регулятора курсу, 1/с
    yaw_rate_max: float = float(_PH["yaw_rate_max"])   # макс. кутова швидкість курсу, рад/с
    crash_dist: float = float(_PH["crash_dist"])  # відстань зіткнення, м (Рейнольдс не підходить ближче ≈ 1,1 м)
    ground_z: float = float(_PH["ground_z"])      # висота «землі», м
    zone_radius: float = float(_PH["zone_radius"])  # радіус робочої зони (горизонтально від zone_center), м
    z_max: float = float(_PH["z_max"])            # стеля зони, м
    zone_center: tuple = tuple(float(x) for x in _PH["zone_center"])   # центр зони (x, y), м

    def __post_init__(self):
        if not (np.isfinite(self.dt) and self.dt > 0):
            raise ValueError(f"SimParams.dt має бути > 0, отримано {self.dt}")
        if not (self.tau_v > 0 and self.gust_tau > 0):
            raise ValueError("SimParams: tau_v і gust_tau мають бути > 0")

    @staticmethod
    def from_dict(d: dict | None) -> "SimParams":
        """Параметри з метаданих запису (meta["params"]); ключі, яких там немає (старі записи) — стандартні."""
        names = SimParams.__dataclass_fields__
        kw = {k: (tuple(v) if k == "zone_center" else v) for k, v in (d or {}).items() if k in names}
        return SimParams(**kw)


@dataclass
class Scenario:
    """Один сценарій: старт, ціль, вітер. Повністю визначається своїм зерном."""
    n_agents: int
    start_pos: np.ndarray        # (N, 3)
    start_yaw: np.ndarray        # (N,)
    goal: np.ndarray             # (3,)
    wind: np.ndarray             # (3,) постійний вітер, м/с
    gust_std: float              # СКВ поривів, м/с
    seed: int
    meta: dict = field(default_factory=dict)
    start_vel: np.ndarray | None = None   # (N, 3), за замовчуванням нулі

    def to_dict(self) -> dict:
        # без округлення: JSON зберігає float64 точно, тож сценарій із файлу дає ті самі траєкторії
        # (динаміка рою хаотична — зсув старту на 1e-6 м змінює долю частини епізодів)
        d = dict(n_agents=int(self.n_agents), start_pos=np.asarray(self.start_pos, float).tolist(),
                 start_yaw=np.asarray(self.start_yaw, float).tolist(), goal=np.asarray(self.goal, float).tolist(),
                 wind=np.asarray(self.wind, float).tolist(), gust_std=float(self.gust_std), seed=int(self.seed),
                 meta=self.meta)
        if self.start_vel is not None:
            d["start_vel"] = np.asarray(self.start_vel, float).tolist()
        return d

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        return Scenario(int(d["n_agents"]), np.array(d["start_pos"], float), np.array(d["start_yaw"], float),
                        np.array(d["goal"], float), np.array(d["wind"], float), float(d["gust_std"]),
                        int(d["seed"]), dict(d.get("meta", {})),
                        np.array(d["start_vel"], float) if d.get("start_vel") is not None else None)


@dataclass
class SwarmState:
    """Стан пакета епізодів."""
    pos: np.ndarray       # (E, N, 3)
    vel: np.ndarray       # (E, N, 3)
    yaw: np.ndarray       # (E, N)
    status: np.ndarray    # (E, N) int: PADDING / ACTIVE / CRASHED / LOST
    goal: np.ndarray      # (E, 3)
    t: float = 0.0

    @property
    def alive(self) -> np.ndarray:
        return self.status == ACTIVE

    @property
    def exists(self) -> np.ndarray:
        return self.status != PADDING


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


class Simulator:
    """Пакет із E незалежних епізодів одного кроку часу."""

    def __init__(self, scenarios: list[Scenario], params: SimParams | None = None):
        self.p = params or SimParams()
        self.scenarios = scenarios
        E = len(scenarios)
        N = max(s.n_agents for s in scenarios)
        self.E, self.N = E, N
        pos = np.zeros((E, N, 3))
        vel = np.zeros((E, N, 3))
        yaw = np.zeros((E, N))
        status = np.zeros((E, N), dtype=np.int8)
        for e, s in enumerate(scenarios):
            pos[e, :s.n_agents] = s.start_pos
            yaw[e, :s.n_agents] = s.start_yaw
            if s.start_vel is not None:
                vel[e, :s.n_agents] = s.start_vel
            status[e, :s.n_agents] = ACTIVE
            pos[e, s.n_agents:] = 1e6 + 100.0 * np.arange(N - s.n_agents)[:, None]   # далеко й неактивні
        self.state = SwarmState(pos, vel, yaw, status, np.stack([s.goal for s in scenarios]))
        self.wind = np.stack([s.wind for s in scenarios])                       # (E, 3)
        self.gust_std = np.array([s.gust_std for s in scenarios])               # (E,)
        self.rngs = [np.random.default_rng([s.seed, GUST_STREAM]) for s in scenarios]   # окремий потік на епізод
        self.n_agents = np.array([s.n_agents for s in scenarios])
        self.gust = self._noise() * self.gust_std[:, None, None]
        self.first_event = np.full(E, np.inf)                                   # час першої події в рої
        self.min_sep = np.full(E, np.inf)                       # мін. відстань між активними після кожного кроку

    def _noise(self) -> np.ndarray:
        """Стандартний шум (E, N, 3); кожен епізод тягне рівно n_agents×3 чисел зі свого потоку,
        тож результат епізоду не залежить від складу пакета."""
        xi = np.zeros((self.E, self.N, 3))
        for e, (r, n) in enumerate(zip(self.rngs, self.n_agents)):
            xi[e, :n] = r.normal(size=(n, 3))
        xi[..., 2] *= 0.3
        return xi

    # ------------------------------------------------------------------ крок
    def step(self, v_cmd: np.ndarray) -> SwarmState:
        p, st = self.p, self.state
        alive = st.alive
        v_cmd = np.where(alive[..., None], np.asarray(v_cmd, dtype=np.float64), 0.0)

        # пориви: OU, детерміновано за зерном кожного епізоду
        a = np.exp(-p.dt / p.gust_tau)
        xi = self._noise()
        self.gust = a * self.gust + np.sqrt(1 - a * a) * self.gust_std[:, None, None] * xi

        acc = (v_cmd - st.vel) / p.tau_v + p.kappa_w * (self.wind[:, None, :] + self.gust)
        vel = st.vel + p.dt * acc
        pos = st.pos + p.dt * vel

        # курс: за горизонтальною командою (якщо вона не надто мала)
        vh = np.linalg.norm(v_cmd[..., :2], axis=-1)
        psi_cmd = np.arctan2(v_cmd[..., 1], v_cmd[..., 0])
        dpsi = np.where(vh > 0.3, wrap_angle(psi_cmd - st.yaw), 0.0)
        yaw = wrap_angle(st.yaw + p.dt * np.clip(p.k_yaw * dpsi, -p.yaw_rate_max, p.yaw_rate_max))

        # нескінченна команда (NaN/inf з політики) — аварія: інакше всі порівняння з NaN хибні й дрон «безсмертний»
        bad = alive & ~(np.isfinite(pos).all(-1) & np.isfinite(vel).all(-1) & np.isfinite(yaw))
        st.pos = np.where((alive & ~bad)[..., None], pos, st.pos)
        st.vel = np.where((alive & ~bad)[..., None], vel, 0.0)
        st.yaw = np.where(alive & ~bad, yaw, st.yaw)
        st.t += p.dt
        self._events(bad)
        return st

    def _events(self, bad: np.ndarray | None = None):
        p, st = self.p, self.state
        alive = st.alive
        if self.N > 1:
            d = np.linalg.norm(st.pos[:, :, None, :] - st.pos[:, None, :, :], axis=-1)
            live = alive[:, :, None] & alive[:, None, :] & ~np.eye(self.N, dtype=bool)
            # мінімальна відстань і на кроці зіткнення (записані кадри — до кроку, там пара вже неактивна)
            self.min_sep = np.minimum(self.min_sep, np.where(live, d, np.inf).min(axis=(1, 2)))
            crash = ((d < p.crash_dist) & live).any(axis=2)
        else:
            crash = np.zeros_like(alive)
        if bad is not None:
            crash |= bad
        crash |= alive & (st.pos[..., 2] < p.ground_z)
        lost = alive & ~crash & ((np.linalg.norm(st.pos[..., :2] - np.asarray(p.zone_center), axis=-1) > p.zone_radius)
                                 | (st.pos[..., 2] > p.z_max))
        st.status = np.where(crash, CRASHED, np.where(lost, LOST, st.status)).astype(np.int8)
        ev = (crash | lost).any(axis=1)
        self.first_event = np.where(ev & np.isinf(self.first_event), st.t, self.first_event)


@dataclass
class Trajectory:
    """Записаний пакет епізодів: T кроків перед кожною дією."""
    pos: np.ndarray        # (T, E, N, 3)
    vel: np.ndarray        # (T, E, N, 3)
    yaw: np.ndarray        # (T, E, N)
    alive: np.ndarray      # (T, E, N) bool
    act: np.ndarray        # (T, E, N, 3) виконана дія (світова система)
    goal: np.ndarray       # (E, 3)
    first_event: np.ndarray  # (E,) час першої події (inf — не було)
    final_status: np.ndarray  # (E, N)
    n_agents: np.ndarray   # (E,)
    min_sep: np.ndarray | None = None  # (E,) мін. відстань між активними після кожного кроку (з кроком зіткнення)


def rollout(scenarios: list[Scenario], controller, duration: float = DURATION, params: SimParams | None = None,
            record: bool = True, callback=None) -> Trajectory:
    """Прогнати пакет епізодів аналітичним контролером (Python-функцією) — хелпер для тестів та ілюстрацій
    (docs/boids_gif.py). Моделі ONNX літають через engine.fly; фізика (Simulator) і обрізання команди |v| ≤ 8 м/с
    тут ті самі, тож Рейнольдс-формула тут і Рейнольдс-модель у fly дають однакові траєкторії (тест паритету).

    controller(state: SwarmState) -> v_cmd (E, N, 3) у світовій системі.
    callback(step, state, v_cmd) — викликається перед кожним кроком (v_cmd — уже обрізана команда).
    """
    sim = Simulator(scenarios, params)
    T = n_steps(duration, sim.p.dt)
    E, N = sim.E, sim.N
    rec = dict(pos=[], vel=[], yaw=[], alive=[], act=[])
    for k in range(T):
        st = sim.state
        if not st.alive.any():
            break
        u = clip_norm(np.asarray(controller(st), dtype=np.float64))     # як engine: |v_cmd| ≤ 8 м/с
        if callback is not None:
            callback(k, st, u)
        if record:
            rec["pos"].append(st.pos.copy()); rec["vel"].append(st.vel.copy())
            rec["yaw"].append(st.yaw.copy()); rec["alive"].append(st.alive.copy())
            rec["act"].append(np.where(st.alive[..., None], u, 0.0))
        sim.step(u)
    stack = {k: (np.stack(v) if v else np.zeros((0, E, N) + ((3,) if k in ("pos", "vel", "act") else ())))
             for k, v in rec.items()}
    return Trajectory(stack["pos"], stack["vel"], stack["yaw"], stack["alive"].astype(bool), stack["act"],
                      sim.state.goal.copy(), sim.first_event.copy(), sim.state.status.copy(),
                      np.array([s.n_agents for s in scenarios]), min_sep=sim.min_sep.copy())
