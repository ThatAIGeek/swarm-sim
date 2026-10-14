"""tools/ — ресурси сцени «Кремль»: кеш шарів карти (parquet) читається назад без втрат."""


def test_map_cache_parquet_roundtrip(tmp_path):
    from tools.map_vector import LAYERS, load_cache, save_cache
    layers = {n: [] for n in LAYERS}
    layers["building"] = [{"id": "a", "geometry": b"\x01\x02", "bbox": {"xmin": 1.0, "xmax": 2.0}, "height": None}]
    save_cache(layers, tmp_path / "c")
    got = load_cache(tmp_path / "c")
    assert got["building"][0]["geometry"] == b"\x01\x02" and got["water"] == []
    assert load_cache(tmp_path / "немає") is None
