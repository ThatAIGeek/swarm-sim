"""Підкладка-карта для Пульта рою: місця, джерела тайлів і знімок для вбудовування.

    python -m tools.map_tiles kremlin                 # → simulator/view/assets/maps/kremlin_esri.jpg (+ .json)
    python -m tools.map_tiles kremlin --provider osm  # схема OpenStreetMap

Пульт має перемикач сцени: «3D-простір» (сітка) або «Кремль» — карта під полігоном у масштабі 1:1 (1 м
симулятора = 1 м на місцевості, x — схід, y — північ), прив'язана так, що ціль рою лежить у точці місця:
рій у кожному сценарії летить до Кремля. Без вбудованого знімка браузер довантажує тайли сам (потрібен
інтернет); зі знімком (simulator/view/assets/maps/<місце>_<джерело>.jpg) карта працює й з диска без мережі.
Знімок робиться один раз цією командою там, де є доступ до сервера тайлів.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import urllib.request
from pathlib import Path

from simulator.view.maps import DEFAULT_PLACE, DEFAULT_PROVIDER, MAPS_DIR, PLACES, PROVIDERS, TILE, player_maps  # noqa: F401

EARTH = 156543.03392804097          # м/піксель на екваторі при z = 0 (Web Mercator)
USER_AGENT = "ml-lab4-swarm-player/1.0 (навчальний симулятор, НУ «Львівська політехніка»)"


def meters_per_pixel(lat: float, zoom: int) -> float:
    return EARTH * math.cos(math.radians(lat)) / 2 ** zoom


def tile_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Дробові координати тайла (x, y) точки у Web Mercator."""
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile_plan(lat: float, lon: float, size_m: float, zoom: int) -> dict:
    """Які тайли потрібні і як вирізати з них квадрат size_m × size_m з центром у (lat, lon)."""
    mpp = meters_per_pixel(lat, zoom)
    side = int(math.ceil(size_m / mpp))                       # пікселів у знімку
    cx, cy = tile_xy(lat, lon, zoom)
    x0, y0 = cx * TILE - side / 2, cy * TILE - side / 2       # лівий верхній кут у глобальних пікселях
    tx = range(int(math.floor(x0 / TILE)), int(math.floor((x0 + side) / TILE)) + 1)
    ty = range(int(math.floor(y0 / TILE)), int(math.floor((y0 + side) / TILE)) + 1)
    return {"zoom": zoom, "mpp": mpp, "side": side, "x0": x0, "y0": y0, "tiles_x": list(tx), "tiles_y": list(ty)}


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def snapshot(place: str, provider: str = DEFAULT_PROVIDER, fetch=_fetch, quality: int = 82) -> tuple[bytes, dict]:
    """Склеїти тайли в один JPEG size_m × size_m; повертає (jpeg, метадані)."""
    from PIL import Image
    p, pr = PLACES[place], PROVIDERS[provider]
    zoom = p.get("zoom", pr["zoom"])
    plan = tile_plan(p["lat"], p["lon"], p["size_m"], zoom)
    tx, ty = plan["tiles_x"], plan["tiles_y"]
    mosaic = Image.new("RGB", (len(tx) * TILE, len(ty) * TILE), (40, 48, 60))
    for j, y in enumerate(ty):
        for i, x in enumerate(tx):
            img = Image.open(io.BytesIO(fetch(pr["url"].format(z=zoom, x=x, y=y)))).convert("RGB")
            mosaic.paste(img, (i * TILE, j * TILE))
    ox, oy = plan["x0"] - tx[0] * TILE, plan["y0"] - ty[0] * TILE
    crop = mosaic.crop((round(ox), round(oy), round(ox) + plan["side"], round(oy) + plan["side"]))
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    meta = {"place": place, "provider": provider, "label": p["label"], "lat": p["lat"], "lon": p["lon"],
            "size_m": plan["side"] * plan["mpp"], "zoom": zoom, "attribution": pr["attribution"]}
    return buf.getvalue(), meta


def snapshot_path(place: str, provider: str) -> Path:
    return MAPS_DIR / f"{place}_{provider}.jpg"


def main():
    ap = argparse.ArgumentParser(description="Знімок карти для Пульта рою")
    ap.add_argument("place", choices=sorted(PLACES))
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default=DEFAULT_PROVIDER)
    a = ap.parse_args()
    jpg, meta = snapshot(a.place, a.provider)
    out = snapshot_path(a.place, a.provider)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(jpg)
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Збережено {out} ({len(jpg) / 1e3:.0f} КБ, {meta['size_m']:.0f} м). Пульт підхопить його сам: "
          f"python -m simulator view …")


if __name__ == "__main__":
    main()
