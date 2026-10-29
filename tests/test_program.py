"""Симулятор як програма: формат моделі ONNX, вбудований Рейнольдс, політ і запис, CLI, візуальний режим."""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch

from simulator import REYNOLDS, check, fly, metrics, stamp
from simulator.modelfile import (Model, ModelFormatError, behavior_fingerprint, export_sklearn, export_torch)
from simulator.record import load
from simulator.scenarios import make_scenarios, save_scenarios

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def scen():
    return make_scenarios(11, 6, n_range=(3, 8))


@pytest.fixture(scope="module")
def mlp_b(tmp_path_factory):
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(63, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))
    return export_torch(net, "B", "body", tmp_path_factory.mktemp("m") / "mlp_b.onnx", "mlp")


def test_teacher_labels_and_ladder(scen, mlp_b):
    """Вчитель пише дію Рейнольдса в станах, які відвідала модель; драбина — перші k дронів за моделлю."""
    rec = fly(mlp_b, scen[:3], teacher=REYNOLDS, nn_agents=2, duration=3.0, hz=False)
    df = rec.frame
    assert set(df.model[df.agent_id < 2]) == {0} and set(df.model[df.agent_id >= 2]) == {1}
    nb = df[[c for c in df.columns if c.startswith("nb") and not c.endswith("mask")]].to_numpy().reshape(-1, 8, 6)
    mask = df[[f"nb{k}_mask" for k in range(8)]].to_numpy()
    ss = df[["self_vx", "self_vy", "self_vz", "self_gx", "self_gy", "self_gz", "self_dgoal"]].to_numpy()
    rey = Model(REYNOLDS).act([nb.astype(np.float64), mask.astype(np.float64), ss.astype(np.float64)])
    np.testing.assert_allclose(df[["label_x", "label_y", "label_z"]].to_numpy(), rey, atol=2e-3)
    rest = df[df.model == 1]                      # дрони «решти рою» летять за Рейнольдсом: act = label
    np.testing.assert_allclose(rest[["act_x", "act_y", "act_z"]].to_numpy(),
                               rest[["label_x", "label_y", "label_z"]].to_numpy(), atol=1e-5)
    m = metrics(rec)
    assert m["summary"]["online_mse"] is not None and m["nn_agents"] == 2


def test_noise_and_batch_independence(scen):
    """Шум дії — у виконаних командах, мітки чисті; епізод не залежить від складу пакета."""
    a = fly(REYNOLDS, scen, teacher=REYNOLDS, action_noise=2.0, duration=4.0, hz=False).frame
    assert np.abs(a[["act_x", "act_y"]].to_numpy() - a[["label_x", "label_y"]].to_numpy()).mean() > 0.5
    b = fly(REYNOLDS, scen[2:4], teacher=REYNOLDS, action_noise=2.0, duration=4.0, episode_offset=2, hz=False).frame
    x = a[a.episode_id == 3].reset_index(drop=True)
    y = b[b.episode_id == 3].reset_index(drop=True)
    assert x.equals(y)


def test_sklearn_models_fly(tmp_path, scen):
    """Будь-яка модель scikit-learn у форматі: поліноміальна регресія і випадковий ліс."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    df = fly(REYNOLDS, scen[:2], teacher=REYNOLDS, duration=5.0, hz=False).frame
    X = np.concatenate([df[[c for c in df.columns if c.startswith("nb")]].to_numpy(),
                        df[["self_vx", "self_vy", "self_vz", "self_gx", "self_gy", "self_gz", "self_dgoal"]].to_numpy()],
                       axis=1).astype(np.float32)                                   # спостереження B: 63 числа
    y = df[["label_x", "label_y", "label_z"]].to_numpy(np.float32)
    for est, name in ((make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=10.0)), "poly"),
                      (RandomForestRegressor(n_estimators=5, max_depth=6, random_state=0), "forest")):
        est.fit(X, y)
        path = export_sklearn(est, "B", "body", tmp_path / f"{name}.onnx", name)
        assert check(path)["ok"]
        np.testing.assert_allclose(Model(path).act([X[:200].astype(np.float64)]), est.predict(X[:200]), atol=1e-3)
        assert len(fly(path, scen[:2], duration=2.0, hz=False).frame) > 0


def test_model_without_metadata_rejected(tmp_path):
    torch.manual_seed(0)
    p = export_torch(torch.nn.Linear(63, 3), "B", "body", tmp_path / "x.onnx")
    import onnx
    m = onnx.load(str(p))
    del m.metadata_props[:]
    onnx.save(m, str(tmp_path / "raw.onnx"))
    with pytest.raises(ModelFormatError):
        Model(tmp_path / "raw.onnx")
    with pytest.raises(ModelFormatError):
        stamp(p, observation="D", frame="body")


def test_fingerprint_survives_resave(tmp_path, mlp_b):
    """Відбиток поведінки однаковий для перезбереженої з іншою назвою моделі; sha256 файлу — ні."""
    from simulator.modelfile import sha256_file
    p2 = stamp(mlp_b, observation="B", frame="body", name="інша назва", out=tmp_path / "copy.onnx")
    assert sha256_file(p2) != sha256_file(mlp_b)
    assert behavior_fingerprint(p2) == behavior_fingerprint(mlp_b)
    assert behavior_fingerprint(REYNOLDS) != behavior_fingerprint(mlp_b)


def test_cli_equals_api(tmp_path, scen, mlp_b):
    """swarm-sim fly (окремий процес) пише той самий запис, що й fly()."""
    sp = save_scenarios(scen[:2], tmp_path / "s.json")
    out = tmp_path / "cli.parquet"
    subprocess.run([sys.executable, "-m", "simulator", "fly", "--model", str(mlp_b), "--teacher", str(REYNOLDS),
                    "--scenarios", str(sp), "-o", str(out), "--duration", "3", "--quiet"], check=True, cwd=ROOT)
    api = fly(mlp_b, sp, teacher=REYNOLDS, duration=3.0, hz=False).frame
    assert load(out).frame.equals(api)
    r = subprocess.run([sys.executable, "-m", "simulator", "metrics", str(out)], cwd=ROOT, capture_output=True,
                       text=True, check=True)
    assert "success_rate" in r.stdout


def _live_variant() -> int:
    """Варіант для живого режиму — variant_01 з фікстури conftest (поточна тека)."""
    return 1


def _page_data(html: str) -> dict:
    return json.loads(html.split("const D = ")[1].split(";</script>")[0])


@pytest.fixture(scope="module")
def live(mlp_b):
    """Живий сервер пульта на вільному порту: (base_url, token, server)."""
    from simulator.view import server as V
    v = _live_variant()
    srv = V.view(variant=v, model=mlp_b, port=0, open_browser=False, block=False)
    base = f"http://127.0.0.1:{srv.server_address[1]}/"
    token = _page_data(urllib.request.urlopen(base).read().decode())["token"]
    yield base, token, srv
    srv.shutdown()
    srv.server_close()
    srv.app.close()


def _post(base, path, obj=None, raw=None, token=None, headers=None):
    """POST до пульта → (код, JSON)."""
    import urllib.error
    data = raw if raw is not None else json.dumps(obj).encode()
    h = dict(headers or {})
    if token:
        h["X-Lab4-Token"] = token
    try:
        r = urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=h))
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_view_record_page_and_server(tmp_path, mlp_b, scen, live):
    """Візуальний режим: сторінка запису (самодостатня) і живий сервер — епізод, пакет, завантаження моделі."""
    from simulator.view import server as V
    rec_path = tmp_path / "r.parquet"
    fly(mlp_b, scen[:2], out=rec_path, teacher=REYNOLDS, duration=3.0, hz=False)
    page = V.view(record=rec_path, export=tmp_path / "r.html")
    html = page.read_text(encoding="utf-8")
    assert "makeFX" not in html and '"mode":"record"' in html and "three" in html.lower()
    assert "token" not in _page_data(html)                     # самодостатня сторінка запису POST не робить
    base, token, srv = live
    D = _page_data(urllib.request.urlopen(base).read().decode())
    assert D["controllers"] and D["P"]["duration"] == 30.0 and D["P"]["goal_r_max"] > 0

    def post(path, obj=None, raw=None):
        code, res = _post(base, path, obj, raw, token)
        assert code == 200, res
        return res
    ep = post("api/episode", dict(set="eval", idx=0, ctrl="reynolds", k=None, goalD=[0, 0, 0], plan=[],
                                  until=False))
    assert ep["T"] == 601 and ep["m"]["success"] is True and ep["m"]["no_event"] is True
    assert ep["m"]["survival_time"] == 30.0 and ep["m"]["goal_reach_rate"] == 1.0
    up = post("api/model?name=x.onnx", raw=Path(mlp_b).read_bytes())
    ep2 = post("api/episode", dict(set="eval", idx=1, ctrl=up["id"], k=2, goalD=[0, 0, 0],
                                   plan=[{"f": 20, "g": [30.0, 40.0, 25.0]}], until=False))
    assert ep2["edited"] and len(ep2["goals"]) == 2
    b = post("api/batch", dict(set="eval", ctrl="reynolds", k=None))
    assert len(b["res"]) == 30 and all(x["success"] and x["no_event"] for x in b["res"])
    assert set(b["res"][0]) == {"survival_time", "success", "no_event", "goal_reach_rate", "online_mse"}


def test_live_server_security(live):
    """Токен сесії, Host, ліміт тіла, валідація запитів, помилки без шляхів (PSE-6, ARCH-12, QA-13)."""
    import http.client
    base, token, srv = live
    port = srv.server_address[1]
    req = dict(set="eval", idx=0, ctrl="reynolds", k=None, goalD=[0, 0, 0], plan=[], until=False)
    assert _post(base, "api/episode", req)[0] == 403                          # без токена
    assert _post(base, "api/episode", req, token="wrong-token")[0] == 403
    assert _post(base, "api/episode", req, token=token, headers={"Host": f"evil.example:{port}"})[0] == 403
    c = http.client.HTTPConnection("127.0.0.1", port)                            # DNS rebinding: чужий Host у GET
    c.request("GET", "/", headers={"Host": "evil.example"})
    assert c.getresponse().status == 403
    c.close()
    assert _post(base, "api/episode", req, token=token, headers={"Host": f"localhost:{port}"})[0] == 200
    c = http.client.HTTPConnection("127.0.0.1", port)                            # тіло > 64 МБ — 413, не читаємо
    c.putrequest("POST", "/api/model")
    c.putheader("X-Lab4-Token", token)
    c.putheader("Content-Length", str(200 * 2 ** 20))
    c.endheaders()
    assert c.getresponse().status == 413
    c.close()
    bad = [dict(req, idx=-1), dict(req, idx=10 ** 6), dict(req, idx="0"), dict(req, set="nope"),
           dict(req, ctrl="zzz"), dict(req, goalD=[float("nan"), 0, 0]), dict(req, goalD=[1, 2]),
           dict(req, plan=[{"f": 3, "g": [1, 2, float("inf")]}]), dict(req, ctrl=srv.app.default, k=0)]
    for r in bad:
        code, res = _post(base, "api/episode", raw=json.dumps(r).encode(), token=token)
        assert code == 400 and res["error"] and "Traceback" not in res["error"], (r, res)
    code, res = _post(base, "api/episode", raw=b"not json", token=token)
    assert code == 400
    code, res = _post(base, "api/model?name=my%20secret.onnx", raw=b"garbage", token=token)
    assert code == 400 and "my secret.onnx" in res["error"] and "000.onnx" not in res["error"]
    assert str(srv.app.upload_dir) not in res["error"] and "/tmp/" not in res["error"]
    # вільний режим: зерна не підряд; ціль далеко за зоною — обмежується, NaN не потрапляє у JSON
    code, ep = _post(base, "api/episode", dict(set="free", free=dict(n=4, seed=1, wind=3, gust=1), ctrl="reynolds",
                                                goalD=[5000.0, 0, 0], plan=[], until=False), token=token)
    assert code == 200 and np.hypot(*ep["goals"][0][:2]) <= srv.app.payload()["P"]["goal_r_max"] + 1e-6
    from simulator.view.server import free_seed
    assert ep["sc"]["seed"] == free_seed(1) and abs(free_seed(2) - free_seed(1)) > 1


def test_upload_dir_removed_on_close(mlp_b):
    """Тимчасова тека завантажених моделей видаляється (atexit / закриття сервера)."""
    from simulator.view import server as V
    v = _live_variant()
    app = V.LiveApp(v)
    app.upload(Path(mlp_b).read_bytes(), "m.onnx")
    assert any(app.upload_dir.iterdir())
    app.close()
    assert not app.upload_dir.exists()


def test_record_view_uses_record_params_and_full_duration(tmp_path, scen, mlp_b):
    """Перегляд запису: dt, тривалість і зона — з метаданих запису; епізод без подій — 30,0 с (а не 29,95);
    метрики епізоду — ті самі, що в swarm-sim metrics (ARCH-5, ARCH-6, PSE-12)."""
    from simulator.core import SimParams
    from simulator.view.server import record_payload
    rec = tmp_path / "r.parquet"
    fly(REYNOLDS, scen[:2], out=rec, teacher=REYNOLDS, duration=4.0, hz=False,
        params=SimParams(zone_radius=300.0))
    d = record_payload(rec)
    assert d["P"]["duration"] == 4.0 and d["P"]["zone_radius"] == 300.0
    per = metrics(rec, by_episode=True)["per_episode"]
    for i, ep in enumerate(d["static"]["episodes"]):
        assert ep["dur"] == pytest.approx(4.0) and ep["m"]["survival_time"] == 4.0
        for k in ("success", "no_event", "goal_reach_rate", "formation_std", "min_separation", "online_mse"):
            assert ep["m"][k] == per[k][i]
    assert set(d["static"]["batch"]["res"][0]) == {"survival_time", "success", "no_event", "goal_reach_rate",
                                                  "online_mse"}


def test_live_episode_equals_headless_metrics(scen):
    """Пульт і swarm-sim metrics дають ті самі числа — зокрема online MSE для B/world (σ_y за frame: ARCH-2,
    PSE-7) і радіус інерції formation_std."""
    from simulator.view.episode import run_episode
    torch.manual_seed(3)
    net = torch.nn.Sequential(torch.nn.Linear(63, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3))
    import tempfile
    path = export_torch(net, "B", "world", Path(tempfile.mkdtemp()) / "bw.onnx")
    sig = {"body": [3.0, 0.9, 0.6], "world": [4.1, 4.1, 0.6]}
    t = Model(REYNOLDS)
    for sc in scen[:2]:
        rec = fly(path, [sc], teacher=REYNOLDS, sigma_y=sig, hz=False)
        per = metrics(rec, by_episode=True)["per_episode"]
        m = run_episode(sc, Model(path), None, t, t, sig)["m"]
        for k in ("survival_time", "success", "no_event", "min_separation", "order_parameter", "goal_reach_rate",
                  "formation_std", "online_mse", "online_mse_pre"):
            if isinstance(m[k], bool) or m[k] is None:
                assert m[k] == per[k][0], k
            else:
                assert m[k] == pytest.approx(per[k][0], abs=2e-4), k


def test_page_has_effects_controls():
    """Сторінка пульта містить ефекти вибуху й феєрверка та перемикач режимів."""
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "simulator" / "view" / "page.html").read_text(encoding="utf-8")
    assert 'id="fxSel"' in html
    assert 'id="fxCrash"' in html and 'id="fxDrone"' in html and "function fxDrone" in html
    assert 'id="lyMarkers"' in html and "st.layers.markers" in html       # маркери дронів вимикаються
    assert all(f'<cs-info slot="toolbar" id="tip{k}"' in html for k in ("Ms", "Mse", "Phi"))   # підказки до графіків
    for mode in ("fire", "boom", "off"):
        assert f'<option value="{mode}">' in html
    assert "function fxExplosion" in html and "function fxShell" in html


def test_page_inlines_core_system():
    """Пульт — один рядок HTML: Core System (стилі, компоненти, шрифти) і three.js вбудовані, жодного зовнішнього
    файлу чи CDN; копія набору в пакеті — з ліцензіями шрифтів та іконок."""
    from simulator.view import server
    html = server.render({"P": {}, "sets": {}, "controllers": []})
    assert "<!--__CORE__-->" not in html and "/*__THREE__*/" not in html
    assert "customElements" in html and "--surface-lab" in html and "@font-face" in html
    assert 'data-style="instrument"' in html and '<html lang="uk">' in html
    assert re.search(r'<(?:link|script)[^>]+(?:href|src)="https?://', html) is None
    core = server.CORE
    for f in ("tokens-embedded-lab.css", "components/bundle.css", "components/bundle.js", "LICENSE-lucide.txt",
              "fonts/OFL-GolosText.txt", "fonts/OFL-IBMPlexMono.txt"):
        assert (core / f).is_file(), f
    stamp = (core / "tokens.css").read_text(encoding="utf-8").splitlines()[0]
    assert stamp.startswith("/* Core System v") and stamp == (core / "tokens-embedded-lab.css").read_text(
        encoding="utf-8").splitlines()[0]


def test_reference_map_keys_and_teacher_group():
    """Еталонні числа для сторінки: no_event_rate і success_ci95 (None, якщо в старому JSON їх немає);
    Рейнольдс у списку моделей — група «Вчитель»."""
    from simulator.view.server import LiveApp, reference_map
    x = dict(success_rate=0.5, success_ci95=[0.3, 0.7], no_event_rate=0.9, survival_time=20.0, online_mse=0.1,
             survival_per_episode=[20.0])
    r = reference_map({"closed_loop": {"BC_C": x, "expert": dict(success_rate=1.0)}})
    assert r["bc_C"]["eval"]["all"]["no_event_rate"] == 0.9 and r["bc_C"]["eval"]["all"]["success_ci95"] == [0.3, 0.7]
    assert r["reynolds"]["eval"]["all"]["no_event_rate"] is None
    app = LiveApp(_live_variant())
    assert app.models["reynolds"]["group"] == "Вчитель"
    app.close()
