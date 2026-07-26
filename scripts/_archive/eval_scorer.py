#!/usr/bin/env python3
"""Score the candidate GRPO strategy-rewards against the pseudo-gold benchmark.

This is the offline gating test from docs/methodology.md axis 3. Reads a
hand-authored pseudo-gold JSONL (built via scripts/build_pseudo_gold.py and
then filled in by hand) where each record has 4 advices labeled by
construction:

    A_correct          — should rank HIGH
    B_anti_pro         — should rank LOW (when round_won=true)
    C_generic          — should rank MEDIUM-LOW
    D_plausible_wrong  — should rank LOW; the polish-vs-tactics canary

Runs each requested scorer against all advices and reports:

    overall pairwise AUC vs construction labels
    A>B / A>C / A>D rates (the D rate is the Goodhart canary)
    per-stratum AUC (map × side)
    per-state failure dump (which states a scorer got wrong, for inspection)

Output:
    data/eval/scorer_<name>.json    raw per-pair scores per scorer
    stdout summary table comparing all scorers

Usage:
    python scripts/eval_scorer.py data/eval/pseudo_gold_authored.jsonl
    python scripts/eval_scorer.py data/eval/pseudo_gold_authored.jsonl --scorers judge,recall_mask
    python scripts/eval_scorer.py data/eval/pseudo_gold_authored.jsonl --no-judge

methodology.md gate: AUC ≥ 0.70 required before any 100-step GRPO run is
launched with this scorer. Below 0.60: scorer is at chance, do not run.

See docs/methodology.md (axis 3) and docs/reward-candidates.md.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys
from pathlib import Path

# Ensure src/ is importable when run as a script
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_module(name: str, path: Path):
    """Load a module file directly, bypassing parent package __init__.py.

    Needed because src/training/__init__.py pulls in PIL/torch transitively;
    this script only needs the FAISS-CPU recall plumbing.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Heavy deps (numpy, faiss, anthropic) are deferred to the scorer functions
# so the script's argument parsing and graceful-exit paths work in any env.
RECALLIndex = None
_extract_action_from_text = None


def _load_recall_deps() -> None:
    global RECALLIndex, _extract_action_from_text
    if RECALLIndex is not None:
        return
    mod = _load_module(
        "_chimera_recall", _REPO_ROOT / "src" / "training" / "recall.py",
    )
    RECALLIndex = mod.RECALLIndex
    _extract_action_from_text = mod._extract_action_from_text

CONSTRUCTION_LABELS = ["A_correct", "B_anti_pro", "C_generic", "D_plausible_wrong"]
DEFAULT_GOLD = Path("data/eval/pseudo_gold_stub.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/eval")
DEFAULT_SMOKE = Path("data/training/grpo/smoke_test.jsonl")
DEFAULT_SMOKE_SOURCE = Path("data/training/grpo/smoke_test_source.jsonl")
GATE_THRESHOLD = 0.70  # methodology.md gate for proceeding to pod time


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_authored(record: dict) -> bool:
    """A record is authored when none of the 4 candidates start with 'TODO'."""
    cands = record.get("candidates", {})
    if not all(k in cands for k in CONSTRUCTION_LABELS):
        return False
    return not any(str(cands[k]).startswith("TODO") for k in CONSTRUCTION_LABELS)


# ----------------------------------------------------------------------
# Scorers
# ----------------------------------------------------------------------

def score_with_judge(records: list[dict]) -> list[list[float]]:
    """Score every record's 4 candidates with the Claude judge.

    Returns a list of [s_A, s_B, s_C, s_D] per record. One API call per
    record (cached across the 4 per-completion lookups).
    """
    _judge = _load_module(
        "_chimera_judge_reward", _REPO_ROOT / "src" / "training" / "judge_reward.py",
    )
    judge_reward = _judge.judge_reward

    out = []
    for i, rec in enumerate(records):
        siblings = [rec["candidates"][k] for k in CONSTRUCTION_LABELS]
        gt = {
            "game_state": rec.get("game_state", {}),
            "pro_action": rec.get("pro_action", {}),
            "round_won": rec.get("round_won"),
        }
        scores = [
            judge_reward(response=resp, ground_truth=gt, siblings=siblings)
            for resp in siblings
        ]
        out.append(scores)
        print(f"  judge: {i + 1}/{len(records)} done", end="\r", flush=True)
    print()
    return out


def score_with_recall_mask(
    records: list[dict],
    smoke_path: Path,
    source_path: Path,
) -> list[list[float]]:
    """Score every record's 4 candidates with RECALL + same-round mask.

    Builds the index once from smoke_test.jsonl with merged source keys,
    then for each gold record, queries with the gold record's
    (demo_stem, round_num) as the mask key. Per-advice scores come from
    extracting an action vec from the advice text and computing
    Q̂(s,a) − V̂(s).
    """
    _load_recall_deps()
    print("  building RECALL index from smoke_test.jsonl ...")
    smoke = load_jsonl(smoke_path)
    sources = load_jsonl(source_path) if source_path.exists() else []
    if sources:
        if len(sources) != len(smoke):
            raise ValueError(
                f"smoke ({len(smoke)}) and source ({len(sources)}) row counts disagree"
            )
        for s, src in zip(smoke, sources):
            gt = s.setdefault("ground_truth", {})
            gt["source"] = {
                "demo_stem": src.get("demo_stem"),
                "round_num": src.get("round_num"),
            }

    samples = [s for s in smoke if "ground_truth" in s]
    index = RECALLIndex()
    index.build_from_samples(samples)
    print(f"  RECALL index built: {index.size} samples")

    out = []
    for rec in records:
        state = rec.get("game_state", {})
        src_key: tuple[str, int] | None = None
        if rec.get("demo_stem") is not None and rec.get("round_num") is not None:
            src_key = (str(rec["demo_stem"]), int(rec["round_num"]))
        scores = []
        for k in CONSTRUCTION_LABELS:
            advice = rec["candidates"][k]
            action = _extract_action_from_text(advice)
            adv = index.recall_advantage(
                state, action, query_source_key=src_key,
            )
            scores.append(float(adv))
        out.append(scores)
    return out


def score_with_bt_head(records: list[dict]) -> list[list[float]] | None:
    """Score with the trained BT head if CHIMERA_BT_HEAD_PATH is set.

    Returns None if no head is available. The BT path has zero labels
    collected as of this commit; this is a stub that activates when a
    head exists.
    """
    head_path = os.environ.get("CHIMERA_BT_HEAD_PATH")
    if not head_path or not Path(head_path).exists():
        return None
    try:
        _bt = _load_module(
            "_chimera_bt_reward", _REPO_ROOT / "src" / "training" / "bt_reward.py",
        )
        bt_head_score = _bt.bt_head_score
    except Exception as e:
        print(f"  bt_head: import failed ({e}); skipping")
        return None
    out = []
    for rec in records:
        state = rec.get("game_state", {})
        scores = [
            float(bt_head_score(state, rec["candidates"][k]))
            for k in CONSTRUCTION_LABELS
        ]
        out.append(scores)
    return out


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def pairwise_auc_breakdown(
    scores: list[list[float]],
) -> dict:
    """Compute pairwise win-rate of A over each of {B, C, D} and overall AUC.

    Tied scores count as 0.5 (standard AUC convention).
    """
    n = len(scores)
    if n == 0:
        return {"n_states": 0, "auc": 0.0, "a_gt_b": 0.0, "a_gt_c": 0.0, "a_gt_d": 0.0}

    def beats(a: float, x: float) -> float:
        if a > x:
            return 1.0
        if a == x:
            return 0.5
        return 0.0

    ab = [beats(s[0], s[1]) for s in scores]
    ac = [beats(s[0], s[2]) for s in scores]
    ad = [beats(s[0], s[3]) for s in scores]
    auc = (sum(ab) + sum(ac) + sum(ad)) / (3 * n)
    return {
        "n_states": n,
        "auc": auc,
        "a_gt_b": sum(ab) / n,
        "a_gt_c": sum(ac) / n,
        "a_gt_d": sum(ad) / n,
    }


def per_stratum_auc(
    records: list[dict], scores: list[list[float]],
) -> dict[str, float]:
    """AUC split by (map, side). Stratum format from build_pseudo_gold.py."""
    by_stratum: dict[tuple, list[list[float]]] = collections.defaultdict(list)
    for rec, sc in zip(records, scores):
        s = rec.get("stratum") or ["unknown", "unknown", "unknown"]
        key = (s[0], s[1])  # map, side (drop phase to avoid singleton strata)
        by_stratum[key].append(sc)
    out = {}
    for key, group in sorted(by_stratum.items()):
        m = pairwise_auc_breakdown(group)
        out[f"{key[0]}/{key[1]}"] = {"n": m["n_states"], "auc": m["auc"]}
    return out


def failures(records: list[dict], scores: list[list[float]]) -> list[dict]:
    """Per-state cases where A failed to outrank one of B/C/D."""
    out = []
    for rec, s in zip(records, scores):
        bad = []
        for label, idx in (("B", 1), ("C", 2), ("D", 3)):
            if s[0] <= s[idx]:
                bad.append(label)
        if bad:
            out.append({
                "id": rec.get("id"),
                "stratum": rec.get("stratum"),
                "failed_against": bad,
                "scores": dict(zip(["A", "B", "C", "D"], s)),
            })
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def render_summary(results: dict[str, dict]) -> str:
    """Side-by-side summary table for stdout."""
    headers = ["Scorer", "AUC", "A>B", "A>C", "A>D", "n", "verdict"]
    rows = []
    for name, r in results.items():
        if r is None:
            rows.append([name, "--", "--", "--", "--", "--", "no head trained"])
            continue
        m = r["overall"]
        verdict = (
            "PASS" if m["auc"] >= GATE_THRESHOLD
            else "MARGINAL" if m["auc"] >= 0.60
            else "FAIL (chance)"
        )
        rows.append([
            name,
            f"{m['auc']:.3f}",
            f"{m['a_gt_b']:.2f}",
            f"{m['a_gt_c']:.2f}",
            f"{m['a_gt_d']:.2f}",
            str(m["n_states"]),
            verdict,
        ])

    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-" * len(line)
    out = [line, sep]
    for row in rows:
        out.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    out.append(sep)
    out.append(f"methodology.md gate: AUC ≥ {GATE_THRESHOLD:.2f} required for pod time")
    return "\n".join(out)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gold", type=Path, nargs="?", default=DEFAULT_GOLD,
                    help="hand-authored pseudo-gold JSONL")
    ap.add_argument("--scorers", default="judge,recall_mask,bt_head",
                    help="comma-separated subset")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip judge (avoids API cost)")
    ap.add_argument("--smoke", type=Path, default=DEFAULT_SMOKE,
                    help="smoke_test.jsonl for RECALL index build")
    ap.add_argument("--source", type=Path, default=DEFAULT_SMOKE_SOURCE,
                    help="smoke_test_source.jsonl for same-round mask keys")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    if not args.gold.exists():
        print(f"Error: {args.gold} not found. Run scripts/build_pseudo_gold.py first.",
              file=sys.stderr)
        sys.exit(1)

    all_records = load_jsonl(args.gold)
    records = [r for r in all_records if is_authored(r)]
    if not records:
        print(f"No authored records in {args.gold}. Author the candidates first.",
              file=sys.stderr)
        print(f"  ({len(all_records)} stub records present, all unfilled)")
        sys.exit(1)
    if len(records) < len(all_records):
        print(f"Note: {len(records)}/{len(all_records)} records authored; "
              "scoring authored subset only.")

    requested = set(args.scorers.split(","))
    if args.no_judge:
        requested.discard("judge")

    results: dict[str, dict | None] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "judge" in requested:
        print("Scoring with judge (Claude API) ...")
        try:
            scores = score_with_judge(records)
            results["judge"] = {
                "overall": pairwise_auc_breakdown(scores),
                "per_stratum": per_stratum_auc(records, scores),
                "failures": failures(records, scores),
                "scores": [{"id": r.get("id"), "scores": dict(zip(CONSTRUCTION_LABELS, s))}
                           for r, s in zip(records, scores)],
            }
            with open(args.output_dir / "scorer_judge.json", "w") as f:
                json.dump(results["judge"], f, indent=2)
        except Exception as e:
            print(f"  judge failed: {e}")
            results["judge"] = None

    if "recall_mask" in requested:
        print("Scoring with RECALL + same-round mask ...")
        try:
            scores = score_with_recall_mask(records, args.smoke, args.source)
            results["recall_mask"] = {
                "overall": pairwise_auc_breakdown(scores),
                "per_stratum": per_stratum_auc(records, scores),
                "failures": failures(records, scores),
                "scores": [{"id": r.get("id"), "scores": dict(zip(CONSTRUCTION_LABELS, s))}
                           for r, s in zip(records, scores)],
            }
            with open(args.output_dir / "scorer_recall_mask.json", "w") as f:
                json.dump(results["recall_mask"], f, indent=2)
        except Exception as e:
            print(f"  recall_mask failed: {e}")
            results["recall_mask"] = None

    if "bt_head" in requested:
        print("Scoring with BT head (if available) ...")
        scores = score_with_bt_head(records)
        if scores is None:
            print("  CHIMERA_BT_HEAD_PATH not set or path missing; skipping.")
            results["bt_head"] = None
        else:
            results["bt_head"] = {
                "overall": pairwise_auc_breakdown(scores),
                "per_stratum": per_stratum_auc(records, scores),
                "failures": failures(records, scores),
                "scores": [{"id": r.get("id"), "scores": dict(zip(CONSTRUCTION_LABELS, s))}
                           for r, s in zip(records, scores)],
            }
            with open(args.output_dir / "scorer_bt_head.json", "w") as f:
                json.dump(results["bt_head"], f, indent=2)

    print()
    print(render_summary(results))

    # Stratum + failure detail per scorer
    for name, r in results.items():
        if r is None:
            continue
        print()
        print(f"=== {name}: per-stratum AUC ===")
        for k, v in r["per_stratum"].items():
            print(f"  {k:>20}  n={v['n']}  auc={v['auc']:.3f}")
        if r["failures"]:
            print(f"=== {name}: {len(r['failures'])} failures ===")
            for f in r["failures"][:5]:  # cap at 5 for readability
                print(f"  state {f['id']:>4} {f['stratum']}  failed vs {f['failed_against']}  "
                      f"scores={f['scores']}")
            if len(r["failures"]) > 5:
                print(f"  ... ({len(r['failures']) - 5} more in {args.output_dir}/scorer_{name}.json)")


if __name__ == "__main__":
    main()
