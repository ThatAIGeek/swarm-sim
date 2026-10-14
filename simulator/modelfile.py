"""Формат моделі симулятора: файл ONNX (формат «lab4-swarm-model/1»).

Симулятор приймає будь-яку модель, збережену в ONNX: поліноміальну регресію, випадковий ліс чи бустинг
(scikit-learn → skl2onnx), нейромережу (PyTorch → torch.onnx.export). Він не виконує Python-код моделі,
лише граф ONNX (onnxruntime). Контролер Рейнольдса — такий самий файл (simulator/assets/reynolds.onnx, simulator.REYNOLDS).

Метадані (metadata_props), які ставить stamp():
    lab4.format      = lab4-swarm-model/1
    lab4.observation = A | B | C      — яке спостереження подає симулятор (simulator/sensors/observation.py)
    lab4.frame       = world | body   — у якій системі модель видає дію
    lab4.name        = назва для записів і таблиць

Входи (B — кількість дронів у батчі; вісь B має бути динамічною — torch.onnx.export з dynamic_axes):
    A, B — один вхід x (B, 63);
    C    — три входи nb (B, 8, 6), mask (B, 8), self_state (B, 7) (за назвами, інакше — за порядком).
    Тип float32 або float64 — як у графі; симулятор приводить дані до нього.
Вихід — v_cmd (B, 3), бажана швидкість у м/с (нормалізацію модель несе в собі): вихід з ім'ям v_cmd, а якщо такого
немає — перший вихід графа. Інша форма (транспонована (3, B), (B, 1, 3), …) — помилка формату.
Симулятор повертає дію frame=body у світову систему (Rz(ψ)·v) і обрізає |v| ≤ 8 м/с.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

FORMAT = "lab4-swarm-model/1"
FRAMES = ("world", "body")
C_INPUTS = ("nb", "mask", "self_state")
SHAPES = {"A": [(63,)], "B": [(63,)], "C": [(8, 6), (8,), (7,)]}
_ORT_TYPES = {"tensor(float)": np.float32, "tensor(double)": np.float64}


class ModelFormatError(ValueError):
    """Файл не відповідає контракту lab4-swarm-model."""


def stamp(path, observation: str, frame: str, name: str | None = None, out=None) -> Path:
    """Записати метадані контракту в ONNX-файл (на місці або в out). Повертає шлях."""
    import onnx

    if observation not in SHAPES:
        raise ModelFormatError(f"observation має бути A, B або C, отримано {observation!r}")
    if frame not in FRAMES:
        raise ModelFormatError(f"frame має бути world або body, отримано {frame!r}")
    path = Path(path)
    m = onnx.load(str(path))
    meta = {p.key: p.value for p in m.metadata_props}
    meta.update({"lab4.format": FORMAT, "lab4.observation": observation, "lab4.frame": frame,
                 "lab4.name": name or meta.get("lab4.name") or path.stem})
    del m.metadata_props[:]
    for k, v in sorted(meta.items()):
        m.metadata_props.add(key=k, value=v)
    out = Path(out) if out else path
    onnx.save(m, str(out))
    return out


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Model:
    """Завантажена модель: act(inputs) -> v_cmd (B, 3) float64 у системі self.frame."""

    def __init__(self, path, threads: int | None = None):
        import onnxruntime as ort

        # глобальний журнал onnxruntime — лише помилки: інакше на деяких машинах (раннери CI, ВМ) він пише в stderr
        # службові попередження на кшталт «Skipping pci_bus_id …», які студента лише плутають
        ort.set_default_logger_severity(3)
        self.path = Path(path)
        if self.path.is_dir():
            raise IsADirectoryError(f"{self.path}: це тека, а не файл моделі *.onnx")
        if not self.path.exists():
            raise FileNotFoundError(f"немає файлу моделі {self.path}")
        self.sha256 = sha256_file(self.path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads or os.environ.get("SWARM_SIM_THREADS", 1))
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        try:
            self.sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        except Exception as e:                                         # noqa: BLE001 — показати причину студенту
            raise ModelFormatError(f"{self.path.name}: onnxruntime не може відкрити модель: {e}") from e
        meta = self.sess.get_modelmeta().custom_metadata_map
        if meta.get("lab4.format") != FORMAT:
            raise ModelFormatError(f"{self.path.name}: немає метаданих {FORMAT} — викличте "
                                   "simulator.modelfile.stamp(path, observation=..., frame=...)")
        self.observation = meta.get("lab4.observation")
        self.frame = meta.get("lab4.frame")
        self.name = meta.get("lab4.name") or self.path.stem
        if self.observation not in SHAPES:
            raise ModelFormatError(f"{self.path.name}: lab4.observation = {self.observation!r}, очікується A, B або C")
        if self.frame not in FRAMES:
            raise ModelFormatError(f"{self.path.name}: lab4.frame = {self.frame!r}, очікується world або body")
        ins = self.sess.get_inputs()
        want = SHAPES[self.observation]
        # C: входи за назвами; модель може не використовувати частину з них (експортер ONNX викидає невживані)
        if self.observation == "C" and ins and {i.name for i in ins} <= set(C_INPUTS):
            slots = [C_INPUTS.index(i.name) for i in ins]
        elif len(ins) == len(want):
            slots = list(range(len(want)))
        else:
            raise ModelFormatError(f"{self.path.name}: спостереження {self.observation} має {len(want)} вхід(и) "
                                   f"{list(C_INPUTS) if self.observation == 'C' else ['x']}, а в моделі "
                                   f"{len(ins)}: {[i.name for i in ins]}")
        self.slots = slots
        self.inputs = []
        for i, shape in zip(ins, [want[k] for k in slots]):
            if i.type not in _ORT_TYPES:
                raise ModelFormatError(f"{self.path.name}: вхід {i.name} має тип {i.type}, потрібен float або double")
            if not i.shape or isinstance(i.shape[0], int):
                raise ModelFormatError(_fixed_batch_msg(self.path.name, f"вхід {i.name}", i.shape, ins))
            tail = tuple(i.shape[1:])
            if len(tail) != len(shape) or any(isinstance(a, int) and a != b for a, b in zip(tail, shape)):
                raise ModelFormatError(f"{self.path.name}: вхід {i.name} має форму {i.shape}, очікується "
                                       f"[B, {', '.join(map(str, shape))}]")
            self.inputs.append((i.name, _ORT_TYPES[i.type]))
        outs = self.sess.get_outputs()
        out = next((o for o in outs if o.name == "v_cmd"), outs[0])        # перевага — виходу з ім'ям v_cmd
        # форму виходу перевіряє act() на фактичному виході: оголошена в графі буває неточною (skl2onnx для лісу
        # пише [None, 1] при виході (B, 3)), а check() проганяє модель на рої й пробних спостереженнях
        self.output = out.name

    def info(self) -> dict:
        return dict(name=self.name, file=self.path.name, sha256=self.sha256, observation=self.observation,
                    frame=self.frame)

    def act(self, inputs: list[np.ndarray]) -> np.ndarray:
        """inputs — масиви спостереження (float64) у порядку контракту. Повертає (B, 3) float64."""
        feed = {n: np.ascontiguousarray(inputs[k], dtype=t) for (n, t), k in zip(self.inputs, self.slots)}
        B = len(inputs[0])
        try:
            y = self.sess.run([self.output], feed)[0]
        except Exception as e:                                         # noqa: BLE001 — помилки onnxruntime (Fail, …)
            raise ModelFormatError(f"{self.path.name}: onnxruntime не виконав модель на батчі з {B} дронів: "
                                   f"{_short(e)}") from e
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (B, 3):
            raise ModelFormatError(f"{self.path.name}: вихід {self.output} має форму {y.shape}, очікується ({B}, 3) "
                                   "— рядок на кожного дрона батча (транспонований вихід (3, B) не приймається)")
        return y


def _fixed_batch_msg(file: str, what: str, shape, ins) -> str:
    names = [i.name for i in ins]
    axes = ", ".join(f"'{n}': {{0: 'B'}}" for n in names + ["v_cmd"])
    return (f"{file}: {what} має фіксовану вісь батча (форма {list(shape)}), а симулятор подає моделі всіх дронів "
            f"одним батчем змінного розміру. Експортуйте з dynamic_axes: torch.onnx.export(..., input_names={names}, "
            f"output_names=['v_cmd'], dynamic_axes={{{axes}}}) — або готовою simulator.modelfile.export_torch")


def _short(e: Exception) -> str:
    """Перший змістовний рядок повідомлення onnxruntime (без службового префікса [ONNXRuntimeError] : 2 : …)."""
    msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return msg.split(" : ", 3)[-1] if msg.startswith("[ONNXRuntimeError]") else msg


def load_model(model) -> Model:
    """Шлях або вже завантажена Model."""
    return model if isinstance(model, Model) else Model(model)


def check(path, n_agents: int = 10, steps: int = 200, seed: int = 0) -> dict:
    """Перевірка файлу перед польотом: контракт, скінченність виходу, частота інференсу для рою з n_agents дронів."""
    from .core import Simulator
    from .scenarios import make_scenario
    from .sensors import observation_inputs

    m = Model(path)
    sim = Simulator([make_scenario(seed, n_agents)])
    st, sel = sim.state, sim.state.alive
    y = m.act(observation_inputs(m.observation, st, sel, {}))
    rng = np.random.default_rng(seed)
    probe = probe_inputs(m.observation, 256, rng)
    yp = m.act(probe)
    finite = bool(np.isfinite(y).all() and np.isfinite(yp).all())
    for _ in range(5):
        m.act(observation_inputs(m.observation, st, sel, {}))
    t0 = time.perf_counter()
    for _ in range(steps):
        m.act(observation_inputs(m.observation, st, sel, {}))
    hz = steps / (time.perf_counter() - t0)
    ok = finite and hz >= 20.0
    return dict(m.info(), ok=ok, finite=finite, inference_hz=round(hz, 1),
                inputs=[f"{n}: {np.dtype(t).name}" for n, t in m.inputs])


def probe_inputs(kind: str, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Випадкові, але правдоподібні спостереження (для перевірки й відбитка поведінки моделі)."""
    from .sensors import observation_B

    nbm = rng.integers(0, 9, n)
    mask = (np.arange(8)[None, :] < nbm[:, None]).astype(np.float64)
    nb = np.concatenate([rng.normal(0, 6, (n, 8, 3)), rng.normal(0, 2, (n, 8, 3))], axis=-1) * mask[..., None]
    g = rng.normal(size=(n, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    ss = np.concatenate([rng.normal(0, 3, (n, 3)), g, rng.uniform(0, 1, (n, 1))], axis=1)
    if kind == "C":
        return [nb, mask, ss]
    if kind == "B":
        return [observation_B(nb, mask, ss)]
    # A — як справжнє спостереження: рій (центр c, спільна швидкість) у світовій системі, слоти інших дронів
    # за номерами підряд — p(3), v(3) кожного, порожні (дронів менше 10) і вибулі — нулі; ціль — за ~150 м
    from .sensors import N_MAX_A
    k = N_MAX_A - 1
    c = np.concatenate([rng.normal(0, 60, (n, 2)), rng.uniform(10, 50, (n, 1))], axis=1)
    v0 = rng.normal(0, 3, (n, 1, 3)) * np.array([1.0, 1.0, 0.3])
    p = c[:, None, :] + rng.normal(0, 6, (n, N_MAX_A, 3))                          # (n, 10, 3): свій + 9 інших
    v = v0 + rng.normal(0, 1.5, (n, N_MAX_A, 3))
    n_others = rng.integers(1, N_MAX_A, n)                                           # у рої 2…10 дронів
    slot = (np.arange(k)[None, :] < n_others[:, None]) & (rng.uniform(size=(n, k)) > 0.1)   # ~10 % вибулих
    pv = np.concatenate([p, v], axis=-1)
    pv[:, 1:] *= slot[..., None]
    az = rng.uniform(-np.pi, np.pi, n)
    goal = np.stack([c[:, 0] + 150 * np.cos(az), c[:, 1] + 150 * np.sin(az), rng.uniform(20, 60, n)], axis=1)
    return [np.concatenate([pv.reshape(n, 6 * N_MAX_A), goal], axis=1)]


def behavior_fingerprint(path, n: int = 512, seed: int = 20261006) -> str:
    """sha256 виходів моделі на фіксованому наборі спостережень, округлених до 1e-4: та сама модель, перезбережена
    з іншими метаданими чи назвою, дає той самий відбиток (на відміну від sha256 файлу)."""
    m = Model(path)
    y = m.act(probe_inputs(m.observation, n, np.random.default_rng(seed)))
    q = np.round(y, 4) + 0.0                                           # + 0.0: −0.0 → 0.0
    return hashlib.sha256((m.observation + m.frame).encode() + q.astype("<f8").tobytes()).hexdigest()


# ---------------------------------------------------------------- експорт у формат симулятора


def export_torch(model, observation: str, frame: str, path, name: str | None = None, scaler: dict | None = None,
                 opset: int = 17) -> Path:
    """PyTorch-модель → ONNX у форматі симулятора (метадані ставить stamp).

    model.forward приймає входи спостереження (A, B: x; C: nb, mask, self_state) і повертає дію. scaler — якщо модель
    навчена на нормованих даних: {x_mean, x_std} (A, B) або {nb_mean, nb_std, self_mean, self_std} (C), і
    {y_mean, y_std}; тоді нормалізація вшивається в граф, і файл видає дію в м/с (C: сусіди нормуються й
    множаться на маску, як під час навчання).
    """
    import torch
    import torch.nn as nn

    if observation not in SHAPES:
        raise ModelFormatError(f"observation має бути A, B або C, отримано {observation!r}")
    s = {k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in (scaler or {}).items()}

    class Deployed(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            for k, v in s.items():
                self.register_buffer(k, v)

        def _out(self, y):
            return y * self.y_std + self.y_mean if s else y

        def forward(self, *xs):
            if observation == "C":
                nb, mask, ss = xs
                if s:
                    nb = (nb - self.nb_mean) / self.nb_std * mask.unsqueeze(-1)
                    ss = (ss - self.self_mean) / self.self_std
                return self._out(self.model(nb, mask, ss))
            x = xs[0]
            if s:
                x = (x - self.x_mean) / self.x_std
            return self._out(self.model(x))

    dep = Deployed().eval()
    if observation == "C":
        ex, names = (torch.zeros(2, 8, 6), torch.ones(2, 8), torch.zeros(2, 7)), list(C_INPUTS)
    else:
        ex, names = (torch.zeros(2, 63),), ["x"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(dep, ex, str(path), input_names=names, output_names=["v_cmd"],
                          dynamic_axes={**{n: {0: "B"} for n in names}, "v_cmd": {0: "B"}}, opset_version=opset,
                          dynamo=False)
    return stamp(path, observation=observation, frame=frame, name=name or path.stem)


def export_sklearn(estimator, observation: str, frame: str, path, name: str | None = None) -> Path:
    """Модель scikit-learn (Pipeline зі StandardScaler, PolynomialFeatures, Ridge, ліс, …) → ONNX у форматі
    симулятора. Підтримуються спостереження A і B (один вхід x (B, 63))."""
    from skl2onnx import to_onnx

    if observation not in ("A", "B"):
        raise ModelFormatError("scikit-learn: лише плаский вхід x (спостереження A або B)")
    onx = to_onnx(estimator, np.zeros((1, 63), dtype=np.float32), target_opset=17)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())
    return stamp(path, observation=observation, frame=frame, name=name or path.stem)
