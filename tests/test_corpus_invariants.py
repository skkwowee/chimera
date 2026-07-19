"""Corpus invariant checks — infra-plan §1 item 2, written BEFORE runbook [1].

Six pure check functions, each `(blob, schema=None, ...) -> list[str]` of
violation strings (empty == pass). `scripts/patch_corpus.py` imports these BY
NAME and runs them post-write; the pytest wrappers below run them on the real
val blobs when present (CI has no blobs -> skip via fixture).

The invariants encode the FIXED (v2.1) semantics from
build_tick_sequences.py / build_v3_features.py after the D3/D4/D6 fixes
(commit fc1efb5, datasheet §5):

  D3 fix — bomb_state one-hot is LIVE: every frame is exactly one of
      {none, carried, planted_a, planted_b}; "carried" iff some player's
      has_c4 bit (per-player dim 14) is set (planted overrides carried at/after
      the plant tick); planted_* implies bomb_x/y stamped from the plant
      position (never both zero — no bombsite sits at the map origin); site
      bits appear only on post-plant frames (phase in {post_plant, end}).
  D4 fix — round_time is anchored at freeze_end and clamped at 0: freeze
      frames are exactly 0, values live in [0, ~2] (normalized /115s), and the
      clock is monotone non-decreasing within a round.
  D6 fix — v3 derived dim 7 (dist_to_bomb, per-player index 56+7=63) is
      plant-gated: wherever bomb_x AND bomb_y are 0 (pre-plant) it carries the
      1.0 sentinel, never distance-to-origin.

On PRE-patch blobs (no "patch_lineage" key) the D3/D4/D6 checks FAIL — that is
the point: the validator pre-exists the mutation, so "patched correctly" is
defined against pre-patch state instead of tautologically blessing the patch
output. Wrappers xfail (non-strict, imperative) on pre-patch blobs and flip
green after runbook [1].

Layout facts these checks rely on (feature_schema_v1.json + builder code):
  global block starts at 10*per_player_dim; within it:
    map(7) | phase(4: freeze,live,post_plant,end) | score(2) | round_num(1)
    | round_time(1) | bomb_state(4: none,carried,planted_a,planted_b)
    | bomb_x | bomb_y | bomb_age | ... = 37 dims total.
  per-player: has_c4 at index 14; v3 appends 9 derived dims at 56..64.

Run: .venv/bin/python -m pytest tests/test_corpus_invariants.py -q
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
SEQ_DIR = REPO / "data" / "processed" / "tick_sequences"
MANIFEST_PATH = SEQ_DIR / "split_manifest_v2.json"

BIT = 0.5          # one-hot bit threshold
EPS = 1e-6
N_PLAYERS = 10
HAS_C4_IDX = 14    # per-player index of has_c4
RAW_PPD = 56       # v2 per-player dim; v3 derived block starts here
DIM7_IDX = RAW_PPD + 7   # dist_to_bomb within a v3 per-player block
GLOBAL_DIM = 37
# offsets within the global block
G_PHASE = 7        # ..10: freeze, live, post_plant, end
G_ROUND_TIME = 14
G_BOMB_STATE = 15  # ..18: none, carried, planted_a, planted_b
G_BOMB_X = 19
G_BOMB_Y = 20

ROUND_TIME_MAX = 2.0     # normalized /115s; live+bomb+end tail stays well under
MAX_MSGS = 20            # cap per check so a broken corpus doesn't OOM the report

REQUIRED_LINEAGE_KEYS = {"script", "script_sha", "transforms",
                         "sha256_pre", "sha256_post", "date"}


# ---------------------------------------------------------------- layout helper

def _layout(blob: dict, schema: dict | None = None) -> dict:
    """Column indices for the dims the checks touch, derived from the blob
    (per_player_dim: v2=56, v3=65) with the schema JSON as fallback."""
    ppd = int(blob.get("per_player_dim",
                       (schema or {}).get("per_player_dim", RAW_PPD)))
    fd = int(blob["feature_dim"])
    gs = N_PLAYERS * ppd
    return {
        "ppd": ppd, "feature_dim": fd, "gs": gs,
        "ok": fd == gs + GLOBAL_DIM,
        "c4_cols": [p * ppd + HAS_C4_IDX for p in range(N_PLAYERS)],
        "dim7_cols": [p * ppd + DIM7_IDX for p in range(N_PLAYERS)],
    }


def _cap(msgs: list[str]) -> list[str]:
    if len(msgs) > MAX_MSGS:
        return msgs[:MAX_MSGS] + [f"... {len(msgs) - MAX_MSGS} more violations suppressed"]
    return msgs


# ---------------------------------------------------------------- the six checks

def check_bomb_bits_consistent(blob: dict, schema: dict | None = None) -> list[str]:
    """D3 fixed semantics: bomb_state rows valid; carried<->has_c4; planted
    implies nonzero plant pos; site bits only on post-plant frames."""
    lay = _layout(blob, schema)
    if not lay["ok"]:
        return [f"feature_dim {lay['feature_dim']} != 10*{lay['ppd']}+{GLOBAL_DIM}"]
    gs = lay["gs"]
    # rule -> [n_bad_frames, n_bad_rounds, first_example]
    agg: dict[str, list] = {}

    def tally(rule: str, bad: torch.Tensor, rnd: int):
        n = int(bad.sum())
        if n == 0:
            return
        a = agg.setdefault(rule, [0, 0, f"first at round {rnd} frame {int(bad.nonzero()[0])}"])
        a[0] += n
        a[1] += 1

    for i, t in enumerate(blob["tensors"]):
        bs = t[:, gs + G_BOMB_STATE:gs + G_BOMB_STATE + 4]
        hot = bs > BIT
        nhot = hot.sum(dim=1)
        none_b, carried, pa, pb = hot[:, 0], hot[:, 1], hot[:, 2], hot[:, 3]
        planted = pa | pb
        any_c4 = t[:, lay["c4_cols"]].max(dim=1).values > BIT
        post = (t[:, gs + G_PHASE + 2] > BIT) | (t[:, gs + G_PHASE + 3] > BIT)  # post_plant|end
        pos_zero = (t[:, gs + G_BOMB_X].abs() <= EPS) & (t[:, gs + G_BOMB_Y].abs() <= EPS)

        tally("non-binary bomb_state values", ((bs - bs.round()).abs() > 1e-3).any(dim=1), i)
        tally("bomb_state multi-hot (>1 bit set)", nhot > 1, i)
        tally("bomb_state all-zero (dead bits — D3)", nhot == 0, i)
        tally("carried bit set but no player has_c4", carried & ~any_c4, i)
        tally("has_c4 set but bomb_state not carried/planted", any_c4 & (none_b | (nhot == 0)), i)
        tally("planted bit set but bomb_x/y both zero", planted & pos_zero, i)
        tally("site bit set outside post-plant/end phase", planted & ~post, i)
    return _cap([f"check_bomb_bits_consistent: {rule}: {n} frames in {r} rounds ({ex})"
                 for rule, (n, r, ex) in agg.items()])


def check_round_time(blob: dict, schema: dict | None = None) -> list[str]:
    """D4 fixed semantics: round_time in [0, ~2], exactly 0 on freeze frames,
    monotone non-decreasing within a round."""
    lay = _layout(blob, schema)
    if not lay["ok"]:
        return [f"feature_dim {lay['feature_dim']} != 10*{lay['ppd']}+{GLOBAL_DIM}"]
    gs = lay["gs"]
    agg: dict[str, list] = {}
    for i, t in enumerate(blob["tensors"]):
        rt = t[:, gs + G_ROUND_TIME]
        freeze = t[:, gs + G_PHASE] > BIT
        checks = {
            "round_time < 0": rt < -EPS,
            f"round_time > {ROUND_TIME_MAX}": rt > ROUND_TIME_MAX + EPS,
            "round_time != 0 on freeze frame (pause-anchored clock — D4)":
                freeze & (rt.abs() > EPS),
            "round_time decreases within round":
                torch.cat([torch.zeros(1, dtype=torch.bool),
                           (rt[1:] - rt[:-1]) < -EPS]),
        }
        for rule, bad in checks.items():
            n = int(bad.sum())
            if n:
                a = agg.setdefault(rule, [0, 0, f"first at round {i} frame {int(bad.nonzero()[0])}"])
                a[0] += n
                a[1] += 1
    return _cap([f"check_round_time: {rule}: {n} frames in {r} rounds ({ex})"
                 for rule, (n, r, ex) in agg.items()])


def check_dim7_plant_gated(blob: dict, schema: dict | None = None) -> list[str]:
    """D6 fixed semantics (v3 only): derived dim 7 (dist_to_bomb) == 1.0
    sentinel on every frame where bomb_x/y are both zero (pre-plant)."""
    lay = _layout(blob, schema)
    if lay["ppd"] <= RAW_PPD:
        return ["check_dim7_plant_gated: blob has no derived block "
                f"(per_player_dim={lay['ppd']}) — v3 blobs only"]
    if not lay["ok"]:
        return [f"feature_dim {lay['feature_dim']} != 10*{lay['ppd']}+{GLOBAL_DIM}"]
    gs = lay["gs"]
    n_frames = n_rounds = 0
    example = ""
    for i, t in enumerate(blob["tensors"]):
        pre_plant = (t[:, gs + G_BOMB_X].abs() <= EPS) & (t[:, gs + G_BOMB_Y].abs() <= EPS)
        if not bool(pre_plant.any()):
            continue
        d7 = t[:, lay["dim7_cols"]]                     # [T, 10]
        bad = pre_plant & ((d7 - 1.0).abs() > 1e-4).any(dim=1)
        n = int(bad.sum())
        if n:
            if not example:
                example = f"first at round {i} frame {int(bad.nonzero()[0])}"
            n_frames += n
            n_rounds += 1
    if n_frames:
        return [f"check_dim7_plant_gated: dist_to_bomb != 1.0 sentinel on "
                f"{n_frames} pre-plant frames in {n_rounds} rounds ({example}) — D6"]
    return []


def check_no_nan_inf(blob: dict, schema: dict | None = None, *,
                     sample_every: int = 8) -> list[str]:
    """No NaN/Inf: full scan of the global block on every round; full-tensor
    scan on every `sample_every`-th round (keeps IO bounded on 1.9 GB blobs)."""
    lay = _layout(blob, schema)
    gs = lay["gs"] if lay["ok"] else 0
    msgs = []
    for i, t in enumerate(blob["tensors"]):
        block = t if (i % sample_every == 0) else t[:, gs:]
        if not bool(torch.isfinite(block).all()):
            n = int((~torch.isfinite(block)).sum())
            msgs.append(f"check_no_nan_inf: round {i} "
                        f"({'full' if i % sample_every == 0 else 'global block'}): "
                        f"{n} non-finite values")
    return _cap(msgs)


def check_match_ids(blob: dict, schema: dict | None = None, *,
                    manifest: dict | None = None, split: str = "val") -> list[str]:
    """Every meta carries match_id; the set of match_ids equals the manifest's
    match keys for this split side (after the D1/D2 map exclusions, which drop
    rounds but — verified — no whole match)."""
    if manifest is None:
        if not MANIFEST_PATH.exists():
            return [f"check_match_ids: manifest not found at {MANIFEST_PATH}"]
        manifest = json.loads(MANIFEST_PATH.read_text())
    msgs = []
    missing = [i for i, m in enumerate(blob["metas"]) if not m.get("match_id")]
    if missing:
        msgs.append(f"check_match_ids: {len(missing)} metas missing match_id "
                    f"(first at round {missing[0]})")
    blob_ids = {str(m["match_id"]) for m in blob["metas"] if m.get("match_id")}
    want = {str(k) for k, v in manifest["matches"].items() if v == split}
    extra, absent = blob_ids - want, want - blob_ids
    if extra:
        msgs.append(f"check_match_ids: {len(extra)} match_ids not in manifest "
                    f"'{split}' side: {sorted(extra)[:5]}")
    if absent:
        msgs.append(f"check_match_ids: {len(absent)} manifest '{split}' matches "
                    f"absent from blob: {sorted(absent)[:5]}")
    return _cap(msgs)


def check_lineage(blob: dict, schema: dict | None = None) -> list[str]:
    """Blob carries patch_lineage: a non-empty list of entries, each with
    {script, script_sha, transforms, sha256_pre, sha256_post, date}."""
    lin = blob.get("patch_lineage")
    if lin is None:
        return ["check_lineage: blob has no 'patch_lineage' key"]
    if not isinstance(lin, list) or not lin:
        return [f"check_lineage: patch_lineage must be a non-empty list, got {type(lin).__name__}"]
    msgs = []
    for j, entry in enumerate(lin):
        if not isinstance(entry, dict):
            msgs.append(f"check_lineage: entry {j} is {type(entry).__name__}, not dict")
            continue
        miss = REQUIRED_LINEAGE_KEYS - set(entry)
        if miss:
            msgs.append(f"check_lineage: entry {j} missing keys {sorted(miss)}")
        for k in REQUIRED_LINEAGE_KEYS & set(entry):
            if entry[k] in (None, "", []):
                msgs.append(f"check_lineage: entry {j} key '{k}' is empty")
    return _cap(msgs)


ALL_CHECKS = [check_bomb_bits_consistent, check_round_time, check_dim7_plant_gated,
              check_no_nan_inf, check_match_ids, check_lineage]


# ---------------------------------------------------------------- blob loading

def load_blob(path: Path) -> dict:
    """Load a corpus blob via _corpus.load_corpus (the canonical mmap reader,
    infra-plan item 1). Falls back to torch.load(mmap=True) + clean_blob so the
    suite works both before and after load_corpus lands."""
    try:
        from _corpus import load_corpus  # type: ignore[attr-defined]
        return load_corpus(str(path))
    except (ImportError, AttributeError, TypeError):
        from _corpus import clean_blob
        blob = torch.load(str(path), map_location="cpu",
                          weights_only=False, mmap=True)
        clean_blob(blob, tag=Path(path).name)
        return blob


# ---------------------------------------------------------------- pytest wrappers

PRE_PATCH_REASON = "pre-patch corpus, flips green after runbook [1]"

# prefer the patched blobs when present (runbook [1]); fall back pre-patch
_TS = "data/processed/tick_sequences"
import os as _os
VAL_BLOBS = [n if not _os.path.exists(f"{_TS}/{n[:-3]}_p1.pt") else f"{n[:-3]}_p1.pt"
             for n in ["val_v2m.pt", "val_v3m.pt"]]


@pytest.fixture(scope="module", params=VAL_BLOBS)
def val_blob(request):
    path = SEQ_DIR / request.param
    if not path.exists():
        pytest.skip(f"corpus blob absent (CI has no blobs): {path}")
    return request.param, load_blob(path)


def _is_pre_patch(blob: dict) -> bool:
    return "patch_lineage" not in blob


def _assert_clean(name: str, blob: dict, violations: list[str]):
    if violations and _is_pre_patch(blob):
        pytest.xfail(f"{name}: {PRE_PATCH_REASON}")
    assert not violations, f"{name}:\n" + "\n".join(violations)


def test_bomb_bits_consistent(val_blob):
    name, blob = val_blob
    _assert_clean(name, blob, check_bomb_bits_consistent(blob))


def test_round_time(val_blob):
    name, blob = val_blob
    _assert_clean(name, blob, check_round_time(blob))


def test_dim7_plant_gated(val_blob):
    name, blob = val_blob
    if int(blob.get("per_player_dim", RAW_PPD)) <= RAW_PPD:
        pytest.skip(f"{name}: v2 blob has no derived block (v3-only check)")
    _assert_clean(name, blob, check_dim7_plant_gated(blob))


def test_no_nan_inf(val_blob):
    name, blob = val_blob
    # not a patch-fixed defect: must hold pre- AND post-patch, no xfail
    violations = check_no_nan_inf(blob)
    assert not violations, f"{name}:\n" + "\n".join(violations)


def test_match_ids(val_blob):
    name, blob = val_blob
    # match_id is already stamped in every meta pre-patch: must hold now too
    violations = check_match_ids(blob, split="val")
    assert not violations, f"{name}:\n" + "\n".join(violations)


def test_lineage(val_blob):
    name, blob = val_blob
    if _is_pre_patch(blob):
        pytest.skip(f"{name}: no lineage until [1]")
    violations = check_lineage(blob)
    assert not violations, f"{name}:\n" + "\n".join(violations)


# ------------------------------------------------- toy patch -> mmap round-trip
# (absorbs the blob-format dimension's pytest — infra-plan item 2)

def _toy_frame(ppd: int, *, phase: str, rt: float, bomb: str,
               c4_player: int | None, bomb_xy=(0.0, 0.0)) -> torch.Tensor:
    fd = N_PLAYERS * ppd + GLOBAL_DIM
    f = torch.zeros(fd)
    gs = N_PLAYERS * ppd
    f[gs + 3] = 1.0                                     # map one-hot: de_mirage
    f[gs + G_PHASE + ["freeze", "live", "post_plant", "end"].index(phase)] = 1.0
    f[gs + G_ROUND_TIME] = rt
    f[gs + G_BOMB_STATE + ["none", "carried", "planted_a", "planted_b"].index(bomb)] = 1.0
    f[gs + G_BOMB_X], f[gs + G_BOMB_Y] = bomb_xy
    if c4_player is not None:
        f[c4_player * ppd + HAS_C4_IDX] = 1.0
    if ppd > RAW_PPD:  # v3 derived: dist_to_bomb sentinel pre-plant
        pre = bomb_xy == (0.0, 0.0)
        for p in range(N_PLAYERS):
            f[p * ppd + DIM7_IDX] = 1.0 if pre else 0.42
    return f


def make_toy_blob(ppd: int = RAW_PPD, *, patched: bool = True) -> dict:
    """A minimal blob obeying the FIXED (post-patch) semantics."""
    rows = [
        _toy_frame(ppd, phase="freeze", rt=0.0, bomb="carried", c4_player=2),
        _toy_frame(ppd, phase="freeze", rt=0.0, bomb="carried", c4_player=2),
        _toy_frame(ppd, phase="live", rt=0.1, bomb="carried", c4_player=2),
        _toy_frame(ppd, phase="live", rt=0.3, bomb="carried", c4_player=2),
        _toy_frame(ppd, phase="post_plant", rt=0.6, bomb="planted_a",
                   c4_player=None, bomb_xy=(0.3, -0.5)),
        _toy_frame(ppd, phase="end", rt=0.9, bomb="planted_a",
                   c4_player=None, bomb_xy=(0.3, -0.5)),
    ]
    blob = {
        "tensors": [torch.stack(rows)],
        "metas": [{"round_num": 1, "map_name": "de_mirage", "winner": "t",
                   "match_id": "9000001", "demo_stem": "toy-m1-mirage"}],
        "feature_dim": N_PLAYERS * ppd + GLOBAL_DIM,
        "per_player_dim": ppd,
        "downsample": 8,
        "schema_version": "feature_schema_v2_toy",
    }
    if patched:
        blob["patch_lineage"] = [{
            "script": "patch_corpus.py", "script_sha": "deadbeef",
            "transforms": ["bomb_bits", "round_time"],
            "sha256_pre": "a" * 64, "sha256_post": "b" * 64,
            "date": "2026-07-19",
        }]
    return blob


TOY_MANIFEST = {"matches": {"9000001": "val"}}


def test_toy_patch_mmap_roundtrip(tmp_path):
    """Post-patch-shaped toy blob survives torch.save -> mmap load -> all six
    checks clean, for both the v2 and v3 layouts."""
    for ppd in (RAW_PPD, RAW_PPD + 9):
        p = tmp_path / f"toy_ppd{ppd}.pt"
        torch.save(make_toy_blob(ppd), p)
        blob = load_blob(p)
        assert blob["tensors"][0].shape == (6, N_PLAYERS * ppd + GLOBAL_DIM)
        for chk in ALL_CHECKS:
            if chk is check_dim7_plant_gated and ppd == RAW_PPD:
                continue
            kw = {"manifest": TOY_MANIFEST, "split": "val"} if chk is check_match_ids else {}
            assert chk(blob, **kw) == [], f"{chk.__name__} on clean toy (ppd={ppd})"


def test_checks_catch_defects():
    """Each check fires on a toy blob carrying its defect (the D3/D4/D6
    pre-patch shapes), so a green run means something."""
    gs = N_PLAYERS * RAW_PPD

    b = make_toy_blob()
    b["tensors"][0][:, gs + G_BOMB_STATE:gs + G_BOMB_STATE + 4] = 0.0  # D3 dead bits
    assert any("all-zero" in v for v in check_bomb_bits_consistent(b))

    b = make_toy_blob()
    b["tensors"][0][0, gs + G_ROUND_TIME] = 1.2                        # D4 pause-anchored
    assert any("freeze" in v for v in check_round_time(b))

    b = make_toy_blob(RAW_PPD + 9)
    b["tensors"][0][0, 0 * (RAW_PPD + 9) + DIM7_IDX] = 0.7             # D6 dist-to-origin
    assert check_dim7_plant_gated(b)

    b = make_toy_blob()
    b["tensors"][0][0, gs + 11] = math.nan
    assert check_no_nan_inf(b)

    b = make_toy_blob()
    del b["metas"][0]["match_id"]
    assert check_match_ids(b, manifest=TOY_MANIFEST, split="val")
    b = make_toy_blob()
    assert check_match_ids(b, manifest={"matches": {"other": "val"}}, split="val")

    b = make_toy_blob(patched=False)
    assert check_lineage(b)
    b = make_toy_blob()
    del b["patch_lineage"][0]["script_sha"]
    assert any("script_sha" in v for v in check_lineage(b))
