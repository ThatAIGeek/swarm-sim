"""Карти сцени «Кремль» для пульта: місця, джерела тайлів і вбудовані ресурси (simulator/view/assets/maps).

Ресурси готують інструменти викладача tools/map_tiles.py (знімок тайлів) і tools/map_vector.py (3D-місто з OSM).
Карта 1:1 (1 м симулятора = 1 м на місцевості, x — схід, y — північ), прив'язана так, що ціль рою лежить у точці
місця. Будівель фізика не знає — це лише підкладка.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

MAPS_DIR = Path(__file__).resolve().parent / "assets" / "maps"
TILE = 256

# місця: точка-якір (до неї прив'язується ціль рою), розмір квадрата карти і масштаб тайлів
PLACES = {
    # Соборна площа біля дзвіниці Івана Великого; 1600 м — той самий квадрат, що й 3D-карта OSM
    # (tools/map_vector.py), тож схема й супутник покривають її повністю, а не лише центр.
    # goal_m — де на карті лежить стандартна ціль рою: зсув (схід, північ), м від центру місця (lat, lon).
    # [346, 157] — над Покровським собором (храм Василя Блаженного) на Червоній площі. Це лише прив'язка карти:
    # фізика, сценарії й метрики не змінюються (ціль, як і раніше, за 150 м від центру старту).
    "kremlin": {"label": "Московський Кремль", "lat": 55.75105, "lon": 37.61750, "size_m": 1600.0, "zoom": 17,
                "goal_m": [346.0, 157.0]},
}
DEFAULT_PLACE = "kremlin"

PROVIDERS = {
    "esri": {"label": "супутник", "zoom": 18,
             "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
             "attribution": "Знімки © Esri, Maxar, Earthstar Geographics"},
    "osm": {"label": "схема", "zoom": 18, "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attribution": "© учасники OpenStreetMap (ODbL)"},
}
DEFAULT_PROVIDER = "esri"


def player_maps() -> dict:
    """Опис карт для сторінки пульта: місця, джерела і вбудовані знімки (data URI), якщо вони є."""
    snaps = {}
    for f in sorted(MAPS_DIR.glob("*.jpg")):
        mf = f.with_suffix(".json")
        if mf.exists():
            meta = json.loads(mf.read_text(encoding="utf-8"))
            snaps[f"{meta['place']}:{meta['provider']}"] = dict(
                meta, src="data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode())
    city = None
    fg, fc = MAPS_DIR / f"{DEFAULT_PLACE}_ground.jpg", MAPS_DIR / f"{DEFAULT_PLACE}_city.json"
    if fg.exists() and fc.exists():
        city = json.loads(fc.read_text(encoding="utf-8"))
        city["ground"] = "data:image/jpeg;base64," + base64.b64encode(fg.read_bytes()).decode()
    return {"places": [dict(id=k, **v) for k, v in PLACES.items()], "default_place": DEFAULT_PLACE, "city": city,
            "providers": [dict(id=k, **v) for k, v in PROVIDERS.items()], "default_provider": DEFAULT_PROVIDER,
            "tile": TILE, "snapshots": snaps}
