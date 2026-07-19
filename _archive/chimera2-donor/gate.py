"""Shared stage gate — ONE green contract, enforced everywhere.

A stage row results/<STAGE>.json is green iff ALL of:
  - "green" == true
  - non-empty "script" naming a repo-relative path that EXISTS
  - non-empty "seeds"
run.sh and every stage script enforce this identically (run.sh delegates to
`python tools/gate.py check <STAGE>`; scripts call require_green() first thing
in main(), so direct `python LN_*/x.py` invocation cannot bypass the gate).
tests/test_rails.py Rail 4 asserts the same contract — one contract, three
enforcers.

Stubs signal "contract only, not implemented" by exiting with code 3
(not_implemented()) so run.sh can distinguish "stop here, nothing to do yet"
(exit 0) from a real failure.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

NOT_IMPLEMENTED_RC = 3


def green_reason(stage: str) -> str | None:
    """None if the stage row is green; otherwise the human-readable reason."""
    f = ROOT / "results" / f"{stage}.json"
    if not f.exists():
        return f"results/{stage}.json missing"
    try:
        d = json.load(open(f))
    except Exception as e:  # noqa: BLE001
        return f"results/{stage}.json unreadable ({e})"
    if not d.get("green"):
        return f"results/{stage}.json has green != true"
    script = d.get("script")
    if not script:
        return f"results/{stage}.json is green but cites no regenerating script"
    if not (ROOT / script).exists():
        return f"results/{stage}.json cites script {script!r} which does not exist in the repo"
    if not d.get("seeds"):
        return f"results/{stage}.json is green but reports no seeds"
    return None


def is_green(stage: str) -> bool:
    return green_reason(stage) is None


def require_green(stage: str, for_stage: str = "") -> None:
    """SystemExit(1) with the block message unless `stage` is green."""
    reason = green_reason(stage)
    if reason is not None:
        who = f"{for_stage} " if for_stage else ""
        print(
            f"BLOCKED — {who}requires {stage} green, but: {reason}.\n"
            "  Build discipline: no stage N+1 until stage N's row is green. "
            "See docs/thesis.md.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def not_implemented(doc: str | None) -> None:
    """Stub exit: print the contract, exit with the reserved 'not implemented' code."""
    print((doc or "").strip())
    print("\nNOT YET IMPLEMENTED — this file is a contract; run.sh gates the build.")
    raise SystemExit(NOT_IMPLEMENTED_RC)


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "check":
        reason = green_reason(argv[2])
        if reason is None:
            print(f"{argv[2]}: green")
            return 0
        print(f"{argv[2]}: NOT green — {reason}")
        return 1
    if len(argv) == 2 and argv[1] == "status":
        for f in sorted((ROOT / "results").glob("*.json")):
            stage = f.stem
            reason = green_reason(stage)
            print(f"{stage}: {'green' if reason is None else 'NOT green — ' + reason}")
        return 0
    print("usage: gate.py check <STAGE> | gate.py status", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
