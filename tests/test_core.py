"""Фізика (simulator.core): 20 Гц, детермінізм, незалежність від пакета, зіткнення, вихід за зону, контур швидкості.
Контролер для rollout — вбудована модель Рейнольдса (ONNX) через той самий шлях, що й engine.fly."""
import numpy as np

from simulator import REYNOLDS
from simulator.core import CRASHED, DT, LOST, Scenario, SimParams, Simulator, rollout
from simulator.engine import model_actions
from simulator.modelfile import Model
from simulator.scenarios import make_scenarios

KW = dict(goal_dist=(150.0, 150.0), v0_max=6.0)
_REY = Model(REYNOLDS)


def reynolds_controller(st):
    """Світові команди Рейнольдса для живих дронів стану st (E, N, 3)."""
    u = np.zeros_like(st.pos)
    u[st.alive] = model_actions(_REY, st, st.alive, {})
    return u


def test_rate_20hz():
    assert DT == 0.05
    tr = rollout(make_scenarios(1, 2, n_fixed=3, **KW), reynolds_controller, 1.0)
    assert tr.pos.shape[0] == 20


def test_deterministic():
    sc = make_scenarios(5, 4, **KW)
    a = rollout(sc, reynolds_controller, 5.0)
    b = rollout(make_scenarios(5, 4, **KW), reynolds_controller, 5.0)
    np.testing.assert_array_equal(a.pos, b.pos)


def test_batch_independent_of_composition():
    """Результат епізоду не залежить від того, з якими епізодами він у пакеті (окремі потоки зерен)."""
    sc = make_scenarios(6, 3, **KW)
    alone = rollout(sc[:1], reynolds_controller, 3.0)
    n = sc[0].n_agents
    together = rollout(sc, reynolds_controller, 3.0)
    np.testing.assert_allclose(alone.pos[:, 0, :n], together.pos[:, 0, :n], atol=1e-12)


def test_crash_detection():
    s = Scenario(2, np.array([[0, 0, 20.0], [3.0, 0, 20.0]]), np.zeros(2), np.array([100.0, 0, 20]),
                 np.zeros(3), 0.0, 1)
    sim = Simulator([s], SimParams())
    for _ in range(40):
        sim.step(np.array([[[4.0, 0, 0], [-4.0, 0, 0]]]))
    assert (sim.state.status[0] == CRASHED).all()
    assert np.isfinite(sim.first_event[0])


def test_zone_exit():
    r0 = SimParams().zone_radius - 10.0                     # за 10 м до межі зони, 100 кроків по 8 м/с — за межею
    s = Scenario(1, np.array([[r0, 0, 20.0]]), np.zeros(1), np.zeros(3), np.zeros(3), 0.0, 1)
    sim = Simulator([s], SimParams())
    for _ in range(100):
        sim.step(np.array([[[8.0, 0, 0]]]))
    assert sim.state.status[0, 0] == LOST


def test_reynolds_never_crashes():
    sc = make_scenarios(11, 60, n_range=(3, 10), **KW)
    tr = rollout(sc, reynolds_controller, 30.0)
    assert not np.isfinite(tr.first_event).any()


def test_velocity_tracks_command():
    s = Scenario(1, np.array([[0, 0, 20.0]]), np.zeros(1), np.zeros(3), np.zeros(3), 0.0, 1)
    sim = Simulator([s], SimParams())
    for _ in range(60):
        sim.step(np.array([[[3.0, 0, 0]]]))
    np.testing.assert_allclose(sim.state.vel[0, 0], [3.0, 0, 0], atol=0.05)
    assert abs(sim.state.yaw[0, 0]) < 1e-6


def test_scenario_json_roundtrip_is_exact():
    """Сценарій із JSON (eval_scenarios.json) дає ту саму траєкторію, що й оригінал: без округлення, зі start_vel."""
    import json
    sc = make_scenarios(17, 4, **KW)
    back = [Scenario.from_dict(json.loads(json.dumps(s.to_dict()))) for s in sc]
    assert all(b.start_vel is not None for b in back)
    a = rollout(sc, reynolds_controller, 5.0)
    b = rollout(back, reynolds_controller, 5.0)
    np.testing.assert_array_equal(a.pos, b.pos)


def test_gust_stream_differs_from_scenario_streams():
    """Пориви епізоду з зерном s не беруться з потоку генерації сценарію s + 1 (вільний режим пульта — зерна
    підряд), тож сусідні сценарії незалежні."""
    from simulator.core import GUST_STREAM
    from simulator.scenarios import make_scenario
    s = make_scenario(41, 3, gust_range=(1.0, 1.0))
    sim = Simulator([s])
    xi0 = sim.gust[0, :3] / s.gust_std                       # перший стандартний шум поривів (z × 0,3)
    for seed in (s.seed, s.seed + 1):
        z = np.random.default_rng(seed).normal(size=(3, 3)) * np.array([1, 1, 0.3])
        assert not np.allclose(xi0, z)
    z = np.random.default_rng([s.seed, GUST_STREAM]).normal(size=(3, 3)) * np.array([1, 1, 0.3])
    np.testing.assert_allclose(xi0, z)


def test_rollout_clips_command_like_engine():
    """rollout обрізає команду |v| ≤ 8 м/с, як engine.fly."""
    sc = make_scenarios(2, 1, n_fixed=2, **KW)
    tr = rollout(sc, lambda st: np.full(st.pos.shape, 100.0), 0.5)
    assert np.linalg.norm(tr.act, axis=-1).max() <= 8.0 + 1e-9


def test_nan_command_crashes_drone():
    """NaN у команді — аварія; позиції не отруєні NaN."""
    def ctrl(st):
        u = np.zeros_like(st.pos)
        u[:, 1] = np.nan
        return u
    tr = rollout(make_scenarios(7, 3, n_fixed=4), ctrl, 2.0)
    assert np.isfinite(tr.first_event).all()
    assert (tr.final_status[:, 1] == CRASHED).all()
    assert np.isfinite(tr.pos).all()
