import json

from scripts import parse_demos as parser


def test_completion_requires_intact_current_bundle(tmp_path):
    marker = tmp_path / "demo_parse.json"
    identity = {"version": 2, "source_sha256": "source"}
    assert not parser.parse_complete(marker, identity)
    files = {}
    for suffix in (*parser.EVENT_TABLES, "header"):
        path = tmp_path / f"demo_{suffix}.json"
        path.write_text("[]")
        files[path.name] = parser.sha256(path)
    for suffix in ("ticks", *parser.EXTRA_TABLES):
        path = tmp_path / f"demo_{suffix}.parquet"
        path.write_bytes(b"fixture")
        files[path.name] = parser.sha256(path)
    marker.write_text(json.dumps({"identity": identity, "files": files}))
    assert parser.parse_complete(marker, identity)
    assert not parser.parse_complete(marker, {**identity, "version": 1})
    (tmp_path / "demo_shots.parquet").write_bytes(b"truncated")
    assert not parser.parse_complete(marker, identity)


def test_source1_is_rejected_before_parsing(tmp_path):
    demo = tmp_path / "old.dem"
    demo.write_bytes(b"HL2DEMO\x00")
    _, ok, message = parser.parse_one(demo)
    assert not ok and "Not a Source 2 demo" in message
