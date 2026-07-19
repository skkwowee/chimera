"""Chimera 2 — invariant rails that exist from day one.

These encode the build discipline as executable checks. The suite is GREEN on the
pristine scaffold (unimplemented stages skip via the NOT YET IMPLEMENTED sentinel)
so that any red is a real regression. Rails must never be deleted to make a stage
pass — fix the stage instead.

Run: pytest tests/ -v
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import gate  # noqa: E402

SENTINEL = "NOT YET IMPLEMENTED"


def implemented(relpath: str) -> bool:
    p = ROOT / relpath
    return p.exists() and SENTINEL not in p.read_text()


def _load_module(relpath: str, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- Rail 1: split leakage — one contract PER manifest --------------------------
# match_ids.json: match_id overlap must be 0; team/map overlap across splits is
# EXPECTED (a ~5-6-team pro corpus cannot avoid it) — that leakage is handled by
# the separate LOTO/LOMO manifests, each with its own contract.

@pytest.mark.skipif(not (ROOT / "data/manifest/match_ids.json").exists(),
                    reason="L0 not built yet")
def test_main_split_has_no_match_id_overlap():
    m = json.load(open(ROOT / "data/manifest/match_ids.json"))
    seen = {}
    for split in ("train", "val", "test"):
        for item in m.get(split, []):
            v = item if isinstance(item, str) else item.get("match_id")
            if v is None:
                continue
            assert v not in seen or seen[v] == split, (
                f"match_id={v!r} appears in both {seen.get(v)} and {split}")
            seen[v] = split


@pytest.mark.skipif(not (ROOT / "data/manifest/loto.json").exists(),
                    reason="LOTO manifest not built yet")
def test_loto_held_out_team_in_exactly_one_split():
    m = json.load(open(ROOT / "data/manifest/loto.json"))
    team = m["held_out_team"]
    holding = [s for s in ("train", "val", "test")
               if any(team in (i.get("teams", []) if isinstance(i, dict) else [i])
                      for i in m.get(s, []))]
    assert holding == [m["held_out_split"]], (
        f"held-out team {team!r} leaks into {holding}")


@pytest.mark.skipif(not (ROOT / "data/manifest/lomo.json").exists(),
                    reason="LOMO manifest not built yet")
def test_lomo_held_out_map_in_exactly_one_split():
    m = json.load(open(ROOT / "data/manifest/lomo.json"))
    map_name = m["held_out_map"]
    holding = [s for s in ("train", "val", "test")
               if any((i.get("map") if isinstance(i, dict) else i) == map_name
                      for i in m.get(s, []))]
    assert holding == [m["held_out_split"]], (
        f"held-out map {map_name!r} leaks into {holding}")


# ---- Rail 2: NO outcome gradient reaches the trunk (C1 exogeneity, machine-checked)
# The locked design (chimera Knob 5) KEEPS a value head but DETACHES it — the trunk
# must be gradient-identical to value_weight=0. A string-grep would both forbid the
# locked design and prove nothing (rename the head and it passes). The canonical
# guarantee is gradient-based: backprop the value loss ALONE; every parameter
# outside the value head must have grad None.

@pytest.mark.skipif(not implemented("L1_trunk/model.py"),
                    reason="L1 trunk not implemented yet")
def test_no_value_leak():
    torch = pytest.importorskip("torch")
    mod = _load_module("L1_trunk/model.py", "wm_model")
    model = mod.build_model()
    batch = mod.dummy_batch()
    out = model(batch) if not isinstance(batch, (tuple, list)) else model(*batch)
    assert isinstance(out, dict) and "value" in out, (
        "model forward must return a dict including 'value' (see model.py contract)")
    v = out["value"].float()
    v_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        v, torch.zeros_like(v))
    model.zero_grad(set_to_none=True)
    v_loss.backward()
    head_params = {id(p) for p in model.value_head.parameters()}
    leaks = [n for n, p in model.named_parameters()
             if id(p) not in head_params and p.grad is not None]
    assert not leaks, (
        f"outcome gradient reached non-value-head params (trunk is NOT "
        f"gradient-identical to value_weight=0): {leaks[:10]}")


# ---- Rail 3: baseline and test row use the SAME metric function -----------------
@pytest.mark.skipif(not implemented("L2_probe/metrics.py"),
                    reason="L2 metrics not implemented yet")
def test_one_metric_harness():
    mod = _load_module("L2_probe/metrics.py", "metrics")
    assert hasattr(mod, "roc_auc_ci"), (
        "metrics.py must expose ONE roc_auc_ci() used for BOTH baseline and test rows")


# ---- Rail 4: the green contract (mirrors tools/gate.py exactly) -----------------
STAGES = ["L0", "L1e", "L1", "L1b", "L1c", "L2a", "L2", "OOD", "L3"]


@pytest.mark.parametrize("ln", STAGES)
def test_green_rows_satisfy_the_gate_contract(ln):
    f = ROOT / "results" / f"{ln}.json"
    if not f.exists():
        pytest.skip(f"{ln} not produced yet")
    d = json.load(open(f))
    if d.get("green"):
        reason = gate.green_reason(ln)
        assert reason is None, f"{ln}.json claims green but: {reason}"


def test_gate_rejects_forged_green(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "L0.json").write_text('{"green": true}')
    assert gate.green_reason("L0") is not None, (
        "a bare {'green': true} row must NOT count as green (script/seeds required)")
    with pytest.raises(SystemExit):
        gate.require_green("L0")


# ---- Rail 5: corpus certification contract (two-stage counts, zero-overpass) ----
def test_l0_green_row_carries_certified_counts():
    f = ROOT / "results" / "L0.json"
    if not f.exists() or not json.load(open(f)).get("green"):
        pytest.skip("L0 not green yet")
    d = json.load(open(f))
    assert d.get("counts", {}).get("clean") == [3876, 705], (
        "L0 green requires the 6-map clean invariant 3,876/705 (datasheet D1+D2)")
    assert d.get("counts", {}).get("five_map") == [3573, 641], (
        "L0 green requires the canonical 5-map counts 3,573/641 (Knob 4)")
    assert d.get("overpass_in_train") == 0, (
        "L0 green requires zero de_overpass rounds in any training loader (Knob 4)")


# ---- Rail 6: end-phase masking on any value metric ------------------------------
def test_l1_green_row_declares_end_phase_masked_value():
    f = ROOT / "results" / "L1.json"
    if not f.exists() or not json.load(open(f)).get("green"):
        pytest.skip("L1 not green yet")
    d = json.load(open(f))
    assert d.get("value_metric_end_phase_masked") is True, (
        "any reported value metric must be end-phase-masked (Knob 5a; post-round "
        "frames leak the winner)")


# ---- Rail 7: corpus facts stay single-sourced from the canonical repo -----------
def test_corpus_facts_parity_with_chimera():
    try:
        cf = _load_module("L0_data/corpus_facts.py", "corpus_facts")
    except ImportError as e:
        pytest.skip(f"canonical chimera repo not available: {e}")
    assert cf.EXCLUDED_MAPS == frozenset({"de_anubis", "de_train"})
    assert callable(cf.clean_blob)
    assert cf.CLEAN_COUNTS == (3876, 705)
    assert cf.FIVE_MAP_COUNTS == (3573, 641)
    assert cf.OOD_MAP == "de_overpass" and cf.OOD_MAP not in cf.CANONICAL_MAPS


# ---- Rail 8: direct script invocation cannot bypass the stage gate --------------
@pytest.mark.skipif(gate.is_green("L1e"), reason="L1e green — gate legitimately open")
def test_direct_invocation_is_gated():
    r = subprocess.run([sys.executable, str(ROOT / "L1_trunk" / "train.py")],
                       capture_output=True, text=True)
    assert r.returncode == 1, f"expected blocked exit 1, got {r.returncode}"
    assert "BLOCKED" in r.stderr


@pytest.mark.skipif(implemented("L0_data/build_corpus.py"),
                    reason="L0 implemented — stub exit code no longer applies")
def test_stub_exits_with_not_implemented_code():
    r = subprocess.run([sys.executable, str(ROOT / "L0_data" / "build_corpus.py")],
                       capture_output=True, text=True)
    assert r.returncode == gate.NOT_IMPLEMENTED_RC
    assert SENTINEL in r.stdout
