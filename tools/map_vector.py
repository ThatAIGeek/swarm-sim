"""Векторна 3D-карта для сцени «Кремль» Пульта рою: дані Overture Maps (OpenStreetMap) → підкладка + будівлі.

    python -m tools.map_vector kremlin       # → simulator/view/assets/maps/kremlin_ground.jpg, kremlin_city.json
    python -m tools.map_vector kremlin --cache work/overture_kremlin   # шари — у теку parquet (повторний запуск
                                                                       # без мережі)

Overture Maps Foundation публікує дані OSM (будівлі з висотами й формою даху, дороги, воду, землекористування) у
відкритому бакеті S3 у форматі GeoParquet. Скрипт читає лише потрібні групи рядків (HTTP Range за статистикою
bbox), переводить координати в локальні метри навколо точки місця (x — схід, y — північ) і зберігає:
- підкладку JPEG (вода, парки, дерева, площі, дороги, контури будівель), квадрат size_m;
- будівлі: контури (дм), висоти основи/верху, висота й форма даху, кольори фасаду й даху — пульт видавлює їх у 3D.
Ліцензія даних: © учасники OpenStreetMap (ODbL), Overture Maps Foundation — див. simulator/view/assets/maps/NOTICE.
Кеш шарів — parquet (дані, а не код): на відміну від pickle, чужий кеш не може виконати код під час читання.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import math
import multiprocessing as mp
import re
import struct
import time
import urllib.parse
from pathlib import Path

from .map_tiles import MAPS_DIR, PLACES

BUCKET = "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com"
RELEASE = "release/2026-09-23.1"
LAYERS = {
    "building": "theme=buildings/type=building",
    "part": "theme=buildings/type=building_part",
    "water": "theme=base/type=water",
    "land_use": "theme=base/type=land_use",
    "land": "theme=base/type=land",
    "infra": "theme=base/type=infrastructure",
    "segment": "theme=transportation/type=segment",
}
GROUND_PX = 2048              # ≈ 0,8 м/піксель
ATTRIBUTION = "© учасники OpenStreetMap (ODbL), Overture Maps Foundation"


# ======================================================================= читання GeoParquet через HTTP Range
_session = None


def _http():
    global _session
    if _session is None:                       # окрема сесія на процес (успадкована через fork — падає)
        import requests
        _session = requests.Session()
    return _session


class RangeFile(io.RawIOBase):
    """Файл на HTTP-сервері, що читається шматками (pyarrow сам просить потрібні діапазони)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        r = _http().head(url, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        r = _http().get(self.url, headers={"Range": f"bytes={self.pos}-{min(self.size, self.pos + n) - 1}"}, timeout=120)
        r.raise_for_status()
        if r.status_code != 206:          # сервер віддав увесь файл або сторінку помилки — не годувати це pyarrow
            raise IOError(f"{self.url}: очікувалась часткова відповідь 206, отримано {r.status_code}")
        self.pos += len(r.content)
        return r.content

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def _list(prefix: str) -> list[str]:
    keys, tok = [], None
    while True:
        u = f"{BUCKET}/?list-type=2&prefix={prefix}"
        if tok:
            u += "&continuation-token=" + urllib.parse.quote(tok)
        r = _http().get(u, timeout=60)
        r.raise_for_status()
        x = r.text
        keys += re.findall(r"<Key>([^<]+)</Key>", x)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m:
            return [k for k in keys if k.endswith(".parquet")]
        tok = m.group(1)


def _row_groups(args):
    """Номери груп рядків файла, bbox яких перетинає наш квадрат (за статистикою стовпців bbox.*)."""
    import pyarrow.parquet as pq
    key, (x0, x1, y0, y1) = args
    pf = pq.ParquetFile(RangeFile(f"{BUCKET}/{key}"))      # pf живий, поки читаємо metadata (інакше segfault)
    md = pf.metadata
    names = [md.schema.column(i).path for i in range(md.num_columns)]
    ix = {n: names.index(f"bbox.{n}") for n in ("xmin", "xmax", "ymin", "ymax")}
    hits = []
    for r in range(md.num_row_groups):
        rg, st = md.row_group(r), {}
        for n, i in ix.items():                            # min/max — поки живі rg і cc (тимчасові об'єкти pyarrow)
            cc = rg.column(i)
            s = cc.statistics
            st[n] = (s.min, s.max) if s is not None and s.has_min_max else None
        if None in st.values() or (st["xmin"][0] <= x1 and st["xmax"][1] >= x0 and st["ymin"][0] <= y1
                                   and st["ymax"][1] >= y0):
            hits.append(r)
    return key, hits


def _read(args):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    key, groups, (x0, x1, y0, y1) = args
    pf = pq.ParquetFile(RangeFile(f"{BUCKET}/{key}"))
    cols = [c for c in pf.schema_arrow.names if c not in ("sources", "version")]
    t = pf.read_row_groups(groups, columns=cols, use_threads=False)
    b = t.column("bbox")
    f = lambda n: pc.struct_field(b, n)  # noqa: E731
    t = t.filter(pc.and_(pc.and_(pc.less_equal(f("xmin"), x1), pc.greater_equal(f("xmax"), x0)),
                         pc.and_(pc.less_equal(f("ymin"), y1), pc.greater_equal(f("ymax"), y0))))
    return t.to_pylist()


def fetch_layer(layer: str, bbox, workers: int = 12) -> list[dict]:
    ctx = mp.get_context("spawn")                         # не fork: pyarrow і HTTPS-сесії після fork падають
    keys = _list(f"{RELEASE}/{LAYERS[layer]}/")
    if not keys:                          # реліз Overture прибрали з S3 — інакше тихо вийшло б місто без будівель
        raise RuntimeError(f"у {BUCKET} немає файлів {RELEASE}/{LAYERS[layer]} — оновіть RELEASE у tools/map_vector.py")
    with cf.ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        hits = [(k, h) for k, h in ex.map(_row_groups, [(k, bbox) for k in keys], chunksize=4) if h]
    with cf.ProcessPoolExecutor(min(workers, 6), mp_context=ctx) as ex:
        rows = [r for part in ex.map(_read, [(k, h, bbox) for k, h in hits]) for r in part]
    return rows


# ======================================================================= геометрія
def wkb(b: bytes):
    """WKB → ('Polygon', [кільця]) | ('LineString', [точки]) | Multi* (список); кільце — список (lon, lat)."""
    def rd(o):
        bo = "<" if b[o] == 1 else ">"
        t = struct.unpack_from(bo + "I", b, o + 1)[0] % 1000
        o += 5
        if t == 1:
            return ("Point", struct.unpack_from(bo + "2d", b, o)), o + 16
        if t == 2:
            n = struct.unpack_from(bo + "I", b, o)[0]
            pts = list(struct.iter_unpack(bo + "2d", b[o + 4:o + 4 + 16 * n]))
            return ("LineString", pts), o + 4 + 16 * n
        if t == 3:
            nr = struct.unpack_from(bo + "I", b, o)[0]
            o += 4
            rings = []
            for _ in range(nr):
                n = struct.unpack_from(bo + "I", b, o)[0]
                rings.append(list(struct.iter_unpack(bo + "2d", b[o + 4:o + 4 + 16 * n])))
                o += 4 + 16 * n
            return ("Polygon", rings), o
        if t in (4, 5, 6, 7):
            n = struct.unpack_from(bo + "I", b, o)[0]
            o += 4
            parts = []
            for _ in range(n):
                g, o = rd(o)
                parts.append(g)
            return ("Multi", parts), o
        raise ValueError(f"WKB тип {t}")
    return rd(0)[0]


def flat(g):
    """Розгорнути Multi* у список простих геометрій."""
    return [x for p in g[1] for x in flat(p)] if g[0] == "Multi" else [g]


class Proj:
    """Локальна площина навколо точки (lat0, lon0): метри, x — схід, y — північ (похибка < 0,1 % на 1 км)."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        p = math.radians(lat0)
        self.kx = 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p)
        self.ky = 111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p)

    def __call__(self, pts):
        return [((lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky) for lon, lat in pts]

    def bbox(self, half):
        return (self.lon0 - half / self.kx, self.lon0 + half / self.kx, self.lat0 - half / self.ky,
                self.lat0 + half / self.ky)


def simplify(pts, tol):
    """Дуглас–Пекер."""
    if len(pts) < 4:
        return pts
    (ax, ay), (bx, by) = pts[0], pts[-1]
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    dmax, imax = 0.0, 0
    for i in range(1, len(pts) - 1):
        d = (abs(dy * (pts[i][0] - ax) - dx * (pts[i][1] - ay)) / L if L > 1e-6     # замкнений контур: кінці збігаються
             else math.hypot(pts[i][0] - ax, pts[i][1] - ay))
        if d > dmax:
            dmax, imax = d, i
    if dmax <= tol:
        return [pts[0], pts[-1]]
    return simplify(pts[:imax + 1], tol)[:-1] + simplify(pts[imax:], tol)


def ring_area(r):
    return 0.5 * sum(r[i][0] * r[(i + 1) % len(r)][1] - r[(i + 1) % len(r)][0] * r[i][1] for i in range(len(r)))


def clean_ring(r, tol=0.25):
    if len(r) > 1 and r[0] == r[-1]:
        r = r[:-1]
    if len(r) < 3:
        return None
    s = simplify(r + [r[0]], tol)[:-1]
    return s if len(s) >= 3 and abs(ring_area(s)) > 0.5 else None


def hexcol(s, default):
    if isinstance(s, str) and re.fullmatch(r"#?[0-9a-fA-F]{6}", s.strip()):
        return int(s.strip().lstrip("#"), 16)
    return default


# ======================================================================= підкладка (PIL)
GROUND_STYLE = {
    "base": "#e9e5dc",
    "land_use": {"grass": "#bcd5a3", "park": "#aacb8d", "garden": "#aacb8d", "flowerbed": "#c9d9a0",
                 "pedestrian": "#ded7c9", "plaza": "#ded7c9", "religious": "#e2d8ca", "playground": "#c6dcae",
                 "pitch": "#b7d49b", "construction": "#d8d0bf", "military": "#ddd3c6"},
    "land": {"wood": "#8fb57a", "forest": "#8fb57a", "scrub": "#a6c48c", "grassland": "#bcd5a3"},
    "water": "#8fbfe0",
    "road": {"motorway": (24, "#fbf6ea"), "trunk": (22, "#fbf6ea"), "primary": (18, "#fbf6ea"),
             "secondary": (14, "#fbf6ea"), "tertiary": (11, "#fbf6ea"), "unclassified": (8, "#f6f2e9"),
             "residential": (8, "#f6f2e9"), "service": (5, "#f3efe6"), "pedestrian": (7, "#efe9dc"),
             "living_street": (7, "#f3efe6"), "footway": (2.2, "#f1ebde"), "path": (1.8, "#efe7d6"),
             "cycleway": (2, "#efe7d6"), "steps": (2.5, "#e6dccb")},
    "road_edge": "#c9c0b0",
    "building": "#c2b9ad",
}


def draw_ground(layers, proj, size_m, px):
    from PIL import Image, ImageDraw
    S = GROUND_STYLE
    im = Image.new("RGB", (px, px), S["base"])
    d = ImageDraw.Draw(im)
    k = px / size_m
    to = lambda pts: [((x + size_m / 2) * k, (size_m / 2 - y) * k) for x, y in pts]  # noqa: E731 — північ угорі

    def polys(rows, color_of):
        for r in rows:
            c = color_of(r)
            if not c:
                continue
            for g in flat(wkb(r["geometry"])):
                if g[0] == "Polygon":
                    d.polygon(to(proj(g[1][0])), fill=c)
                    for hole in g[1][1:]:
                        d.polygon(to(proj(hole)), fill=S["base"])
    polys(layers["land_use"], lambda r: S["land_use"].get(r.get("class")) or S["land_use"].get(r.get("subtype")))
    polys(layers["land"], lambda r: S["land"].get(r.get("class")))
    for r in layers["land"]:                                   # окремі дерева — кола крони
        if r.get("class") == "tree":
            g = wkb(r["geometry"])
            if g[0] == "Point":
                (x, y), = to(proj([g[1]]))
                rr = 2.6 * k
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill="#86ad72")
    polys(layers["water"], lambda r: S["water"])
    for r in layers["water"]:                                  # річки-лінії
        for g in flat(wkb(r["geometry"])):
            if g[0] == "LineString":
                d.line(to(proj(g[1])), fill=S["water"], width=max(1, int(6 * k)))
    roads = [(r, S["road"].get(r.get("class"))) for r in layers["segment"] if r.get("subtype") == "road"]
    roads = [(r, st) for r, st in roads if st]
    for widen, color in ((1.6, S["road_edge"]), (0, None)):    # обвідка, потім заливка
        for r, (w, c) in sorted(roads, key=lambda x: x[1][0]):
            for g in flat(wkb(r["geometry"])):
                if g[0] == "LineString":
                    d.line(to(proj(g[1])), fill=color or c, width=max(1, int((w + widen) * k)), joint="curve")
    polys(layers["building"], lambda r: S["building"])
    return im


# ======================================================================= будівлі
DOME = {"onion", "dome", "round"}
PYRAMID = {"pyramidal", "hipped", "gabled", "skillion", "half-hipped", "gambrel", "mansard", "tented"}
FLOOR_H = 3.3


def building_rows(layers, proj, half):
    """Частини будівель + будівлі без частин → [основа, верх, висота даху, форма, фасад, дах, контур, діри] (дм)."""
    out = []
    items = list(layers["part"]) + [b for b in layers["building"] if not b.get("has_parts")]
    for r in items:
        if r.get("is_underground"):
            continue
        h = r.get("height") or ((r.get("num_floors") or 0) * FLOOR_H) or 9.0
        base = r.get("min_height") or ((r.get("min_floor") or 0) * FLOOR_H) or 0.0
        if h <= base + 0.3:
            continue
        shape = (r.get("roof_shape") or "").lower()
        form = 1 if shape in DOME else 2 if shape in PYRAMID else 0
        facade = hexcol(r.get("facade_color"), 0xD8D0C2)
        roof = hexcol(r.get("roof_color"), facade if form == 0 else 0x6F7A70)
        for g in flat(wkb(r["geometry"])):
            if g[0] != "Polygon":
                continue
            rings = [clean_ring(proj(rg)) for rg in g[1]]
            if not rings[0]:
                continue
            outer = rings[0]
            if any(abs(x) > half or abs(y) > half for x, y in outer):
                continue
            rad = math.sqrt(abs(ring_area(outer)) / math.pi)
            roof_h = r.get("roof_height") or (min(rad * (1.7 if shape == "onion" else 1.0), 0.6 * (h - base))
                                              if form else 0.0)
            roof_h = min(roof_h, h - base)
            q = lambda pts: [int(round(c * 10)) for p in pts for c in p]  # noqa: E731 — дециметри
            out.append([int(round(base * 10)), int(round(h * 10)), int(round(roof_h * 10)), form, facade, roof,
                        q(outer), [q(x) for x in rings[1:] if x]])
    return out


def save_cache(layers: dict, d: Path):
    """Шари → тека d: <шар>.parquet (порожній шар — <шар>.empty)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    d.mkdir(parents=True, exist_ok=True)
    for name, rows in layers.items():
        if rows:
            pq.write_table(pa.Table.from_pylist(rows), d / f"{name}.parquet", compression="zstd")
        else:
            (d / f"{name}.empty").write_text("", encoding="utf-8")


def load_cache(d: Path) -> dict | None:
    """Тека кешу → шари; None, якщо якогось шару немає (тоді завантажити заново)."""
    import pyarrow.parquet as pq

    if not d.is_dir():
        return None
    layers = {}
    for name in LAYERS:
        if (d / f"{name}.parquet").exists():
            layers[name] = pq.read_table(d / f"{name}.parquet").to_pylist()
        elif (d / f"{name}.empty").exists():
            layers[name] = []
        else:
            return None
    return layers


def build(place_id: str, size_m: float | None = None, px: int = GROUND_PX, verbose=True, cache: Path | None = None):
    p = PLACES[place_id]
    size_m = size_m or p["size_m"]          # той самий квадрат, що в схеми й супутника (simulator/view/maps.py)
    proj = Proj(p["lat"], p["lon"])
    bbox = proj.bbox(size_m / 2)
    layers = load_cache(cache) if cache else None
    if layers is None:
        layers = {}
        for name in LAYERS:
            t0 = time.time()
            layers[name] = fetch_layer(name, bbox)
            if verbose:
                print(f"  {name}: {len(layers[name])} об'єктів, {time.time() - t0:.0f} с", flush=True)
        if cache:
            save_cache(layers, cache)
    im = draw_ground(layers, proj, size_m, px)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80, optimize=True, progressive=True)
    city = {"place": place_id, "label": p["label"], "lat": p["lat"], "lon": p["lon"], "size_m": size_m,
            "attribution": ATTRIBUTION, "release": RELEASE,
            "buildings": building_rows(layers, proj, size_m / 2)}
    return buf.getvalue(), city


def paths(place_id: str):
    return MAPS_DIR / f"{place_id}_ground.jpg", MAPS_DIR / f"{place_id}_city.json"


def main():
    ap = argparse.ArgumentParser(description="3D-карта місця з Overture Maps (OSM) для Пульта рою")
    ap.add_argument("place", choices=sorted(PLACES))
    ap.add_argument("--cache", default=None, help="тека кешу завантажених шарів (parquet; повторний запуск без мережі)")
    a = ap.parse_args()
    jpg, city = build(a.place, cache=Path(a.cache) if a.cache else None)
    if not city["buildings"]:
        raise SystemExit("0 будівель — закомічену карту не перезаписую (перевірте RELEASE і мережу)")
    fg, fc = paths(a.place)
    fg.parent.mkdir(parents=True, exist_ok=True)
    fg.write_bytes(jpg)
    fc.write_text(json.dumps(city, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Збережено {fg} ({len(jpg) / 1e3:.0f} КБ) і {fc} ({fc.stat().st_size / 1e3:.0f} КБ, "
          f"{len(city['buildings'])} будівель). Пульт (python -m simulator view …) підхопить її сам")


if __name__ == "__main__":
    main()
