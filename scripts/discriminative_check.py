#!/usr/bin/env python3
"""Verbalization discriminative check: does tactical LANGUAGE carry the
outcome signal that's in the structured state?

Three classifiers predict round_won on a demo-disjoint validation split:

  1. MAJORITY        — predict the most common class. The floor.
  2. STRUCTURED      — logistic regression on numeric features from the
                       egocentric game_state (man-advantage, hp, money,
                       phase, bomb). The CEILING: how much outcome signal
                       is in the state at all, at this single tick.
  3. CAPTION (TFIDF) — logistic regression on TF-IDF of Claude's captions.
                       The TEST: does natural language retain that signal?

Interpretation:
  - caption ≈ structured, both >> majority  → language preserves the signal.
        Green-light the L3 verbalization layer.
  - structured >> majority, caption ≈ majority → verbalization drops signal
        (large gap). Rethink caption content before building the bridge.
  - structured ≈ majority → single-tick state itself barely predicts outcome
        (needs temporal context). Inconclusive for language; rerun with
        windowed captions.

gap = structured_acc - caption_acc is the headline: signal lost in the trip
through language.

Split is demo-disjoint (val = one held-out demo) so no same-round leakage.

Usage:
    python scripts/discriminative_check.py
    python scripts/discriminative_check.py --val-demo furia-vs-vitality-m3-nuke
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
CAPTIONS = REPO / "data" / "captions" / "captions.jsonl"
SOURCE = REPO / "data" / "training" / "grpo" / "smoke_test_source.jsonl"
DATA = REPO / "data" / "training" / "grpo" / "smoke_test.jsonl"

N_STRUCT_FEATS = 11
_PHASE = {"freezetime": 0, "buy": 0, "pistol_round": 1, "playing": 2,
          "post-plant": 3, "post_plant": 3}
_BOMB = {"carried": 0, "dropped": 1, "planted": 2, "defused": 3, "exploded": 4}


def load_captions(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def load_structured_index() -> dict[tuple, dict]:
    """Map (demo, round, tick) -> egocentric game_state from ground_truth."""
    idx = {}
    src = [json.loads(l) for l in SOURCE.open() if l.strip()]
    dat = [json.loads(l) for l in DATA.open() if l.strip()]
    for s, d in zip(src, dat):
        gs = (d.get("ground_truth", {}) or {}).get("game_state")
        if gs is not None:
            idx[(s["demo_stem"], s["round_num"], s["tick"])] = gs
    return idx


def structured_features(gs: dict) -> list[float]:
    """Numeric features from a single-tick EGOCENTRIC game_state."""
    mates = float(gs.get("alive_teammates", 0) or 0)   # includes self
    enemies = float(gs.get("alive_enemies", 0) or 0)
    return [
        mates, enemies, mates - enemies,
        mates / max(enemies, 1.0),
        float(gs.get("player_health", 0) or 0),
        float(gs.get("player_armor", 0) or 0),
        float(gs.get("player_money", 0) or 0),
        float(gs.get("visible_enemies", 0) or 0),
        1.0 if gs.get("player_side") == "T" else 0.0,
        float(_PHASE.get(str(gs.get("round_phase", "")).lower(), 2)),
        float(_BOMB.get(str(gs.get("bomb_status", "")).lower(), 0)),
    ]


def fit_eval(Xtr, ytr, Xva, yva, is_text: bool):
    if is_text:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
        Xtr_m = vec.fit_transform(Xtr)
        Xva_m = vec.transform(Xva)
    else:
        Xtr_m = np.asarray(Xtr, dtype=np.float64)
        Xva_m = np.asarray(Xva, dtype=np.float64)
        mu, sd = Xtr_m.mean(0), Xtr_m.std(0) + 1e-8
        Xtr_m = (Xtr_m - mu) / sd
        Xva_m = (Xva_m - mu) / sd
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xtr_m, ytr)
    pred = clf.predict(Xva_m)
    proba = clf.predict_proba(Xva_m)[:, 1]
    acc = accuracy_score(yva, pred)
    try:
        auc = roc_auc_score(yva, proba)
    except ValueError:
        auc = float("nan")
    return acc, auc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--captions", type=Path, default=CAPTIONS)
    ap.add_argument("--val-demo", default="furia-vs-vitality-m3-nuke")
    ap.add_argument("--out", type=Path,
                    default=REPO / "data" / "captions" / "discriminative_check.json")
    args = ap.parse_args()

    caps = load_captions(args.captions)
    struct_idx = load_structured_index()
    print(f"Loaded {len(caps)} captions")

    by_demo = Counter(c["demo_stem"] for c in caps)
    print(f"Per-demo: {dict(by_demo)}")
    if args.val_demo not in by_demo:
        args.val_demo = by_demo.most_common(1)[0][0]
        print(f"  default val demo absent; using {args.val_demo}")

    train = [c for c in caps if c["demo_stem"] != args.val_demo]
    val = [c for c in caps if c["demo_stem"] == args.val_demo]
    print(f"Split: train={len(train)} val={len(val)} (val demo={args.val_demo})\n")

    ytr = np.array([1 if c["round_won"] else 0 for c in train])
    yva = np.array([1 if c["round_won"] else 0 for c in val])

    maj_class = Counter(yva.tolist()).most_common(1)[0][0]
    maj_acc = float(np.mean(yva == maj_class))

    def feats(rows):
        out = []
        for c in rows:
            gs = struct_idx.get((c["demo_stem"], c["round_num"], c["tick"]))
            out.append(structured_features(gs) if gs else [0.0] * N_STRUCT_FEATS)
        return out
    s_acc, s_auc = fit_eval(feats(train), ytr, feats(val), yva, is_text=False)
    c_acc, c_auc = fit_eval([c["caption"] for c in train], ytr,
                            [c["caption"] for c in val], yva, is_text=True)

    gap = s_acc - c_acc
    print("=" * 64)
    print(f"{'classifier':18s} {'val_acc':>9s} {'val_auc':>9s} {'vs_majority':>12s}")
    print("=" * 64)
    print(f"{'majority':18s} {maj_acc:>9.4f} {'—':>9s} {'—':>12s}")
    print(f"{'structured':18s} {s_acc:>9.4f} {s_auc:>9.4f} {s_acc-maj_acc:>+12.4f}")
    print(f"{'caption (TFIDF)':18s} {c_acc:>9.4f} {c_auc:>9.4f} {c_acc-maj_acc:>+12.4f}")
    print("=" * 64)
    print(f"\nVERBALIZATION GAP (structured - caption): {gap:+.4f}\n")

    if s_acc - maj_acc < 0.03:
        verdict = ("INCONCLUSIVE: single-tick structured state barely beats "
                   "majority. Outcome needs temporal context — rerun with "
                   "windowed captions.")
    elif c_acc - maj_acc < 0.03:
        verdict = ("LANGUAGE DROPS SIGNAL: structured predicts outcome but "
                   "captions don't. Large gap — rethink caption content "
                   "before building the bridge.")
    elif gap < 0.05:
        verdict = ("GREEN LIGHT: captions retain ~all structured signal. "
                   "Language is a near-sufficient statistic here. Proceed to "
                   "the encoder->language bridge.")
    else:
        verdict = (f"PARTIAL: captions carry real signal (+{c_acc-maj_acc:.3f} "
                   f"over majority) but lose {gap:.3f} vs structured. Usable, "
                   f"with a measurable language bottleneck.")
    print("VERDICT:", verdict)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "val_demo": args.val_demo,
        "n_train": len(train), "n_val": len(val),
        "majority_acc": maj_acc,
        "structured_acc": s_acc, "structured_auc": s_auc,
        "caption_acc": c_acc, "caption_auc": c_auc,
        "verbalization_gap": gap,
        "verdict": verdict,
    }, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
