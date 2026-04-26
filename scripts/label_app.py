#!/usr/bin/env python3
"""Streamlit-based labeling tool for pairwise CS2 strategic-advice comparisons.

An expert player reads two completions side-by-side for the same game state and
picks the better one (or ties / skips). Output is appended to a JSONL that the
preference trainer consumes downstream.

INSTALL:
    pip install streamlit
    # or: uv pip install streamlit

RUN:
    streamlit run scripts/label_app.py -- \\
        --candidates outputs/labels/candidate_pairs.jsonl \\
        --preferences outputs/labels/preferences.jsonl \\
        --labeler davidzeng

NOTES:
- Position of A/B is randomized per pair to avoid left-bias. The original
  identity of "ui_a" is recorded in the output so the trainer can recover the
  canonical mapping.
- Judge scores from the input are intentionally NOT shown to the labeler.
- Saves on every click (append-only JSONL); the app is fully resumable -- it
  scans preferences.jsonl on start and skips already-labeled pair_ids.
- Streamlit doesn't expose real keyboard shortcuts cleanly. Closest hack is to
  add an HTML/JS component that listens for keypresses and clicks the matching
  button by aria-label, but it is fragile across Streamlit versions; we keep
  it button-only for reliability. (See st.components.v1.html if you want to try.)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import streamlit as st


# --------------------------------------------------------------------------- #
# Pretty-print + dataset/radar helpers
# --------------------------------------------------------------------------- #
def pretty_json(text: str) -> str:
    """Indent JSON if parseable, else return text unchanged."""
    if not text:
        return text
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except Exception:
        return text


@st.cache_data(show_spinner=False)
def load_smoke_dataset(path: str) -> dict[int, dict]:
    """Load smoke_test.jsonl indexed by sample_idx for round-context lookup.

    Returns {sample_idx: sample_dict}. Empty dict if file not present.
    """
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[int, dict] = {}
    with p.open("r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                out[i] = json.loads(line)
            except Exception:
                continue
    return out


@st.cache_data(show_spinner=False)
def get_radar_png_bytes(map_name: str) -> bytes | None:
    """Return PNG bytes for the map's radar image from awpy, or None.

    Reuses the same data source as cs2-tools/export_viewer_data.py
    (awpy ships radar PNGs for all active CS2 maps).
    """
    try:
        from awpy.data import MAPS_DIR  # type: ignore
    except Exception:
        return None
    candidates = [
        Path(MAPS_DIR) / f"{map_name}.png",
        Path(MAPS_DIR) / f"{map_name}_light.png",
    ]
    for p in candidates:
        if p.exists():
            return p.read_bytes()
    return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse args from sys.argv after the streamlit `--` separator."""
    p = argparse.ArgumentParser(description="CS2 advice pairwise labeling UI")
    p.add_argument(
        "--candidates",
        type=Path,
        default=Path("outputs/labels/candidate_pairs.jsonl"),
        help="Input JSONL of candidate pairs (read-only).",
    )
    p.add_argument(
        "--preferences",
        type=Path,
        default=Path("outputs/labels/preferences.jsonl"),
        help="Output JSONL; appended to on every click.",
    )
    p.add_argument(
        "--labeler",
        type=str,
        default=os.environ.get("USER", "anonymous"),
        help="Labeler identity recorded in each output row.",
    )
    p.add_argument(
        "--smoke-dataset",
        type=Path,
        default=Path("data/training/grpo/smoke_test.jsonl"),
        help="Path to smoke_test.jsonl, used to fetch the original prompt text "
             "(round events, full HUD context) by sample_idx.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        st.error(f"Candidates file not found: {path}")
        st.stop()
    pairs: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def load_labeled_ids(path: Path) -> tuple[set[str], int]:
    """Return (set of labeled pair_ids, count labeled today)."""
    if not path.exists():
        return set(), 0
    labeled: set[str] = set()
    today_iso = date.today().isoformat()
    today_count = 0
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = row.get("pair_id")
            if pid:
                labeled.add(pid)
            ts = row.get("labeled_at", "")
            if ts.startswith(today_iso):
                today_count += 1
    return labeled, today_count


def append_preference(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #
def select_next_pair(
    pairs: list[dict[str, Any]], labeled: set[str]
) -> dict[str, Any] | None:
    """Highest informativeness_score among unlabeled."""
    unlabeled = [p for p in pairs if p["pair_id"] not in labeled]
    if not unlabeled:
        return None
    unlabeled.sort(
        key=lambda p: p.get("informativeness_score", 0.0), reverse=True
    )
    return unlabeled[0]


def shuffle_for_display(pair: dict[str, Any]) -> tuple[str, str, str]:
    """Randomly swap A/B. Returns (left_text, right_text, ui_a_was_originally)."""
    if random.random() < 0.5:
        return pair["completion_a"], pair["completion_b"], "A"
    return pair["completion_b"], pair["completion_a"], "B"


def canonical_choice(ui_choice: str, ui_a_was: str) -> str:
    """Map UI-relative click ('A'|'B'|'tie'|'skip') to canonical A/B."""
    if ui_choice in ("tie", "skip"):
        return ui_choice
    # ui_choice is "A" or "B" (which side they clicked).
    if ui_a_was == "A":
        return ui_choice  # no swap
    # ui_a_was == "B": clicking left ("A") means original B was preferred.
    return "B" if ui_choice == "A" else "A"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_round_context(sample_idx: int, smoke_path: str) -> None:
    """Show the full prompt text from smoke_test.jsonl for this sample.

    The prompt has rich context (round events timeline, economy, current state
    block) that the high-level state card can't fit. Reuses the same input the
    GRPO trainer feeds the model, no recomputation.
    """
    ds = load_smoke_dataset(smoke_path)
    sample = ds.get(sample_idx)
    if not sample:
        return
    prompt = sample.get("prompt", [])
    text_blocks: list[str] = []
    for b in prompt:
        if isinstance(b, dict) and b.get("type") == "text":
            t = b.get("text") or ""
            if t:
                text_blocks.append(t)
        elif isinstance(b, str):
            text_blocks.append(b)
    if not text_blocks:
        return
    with st.expander("Round context (full prompt — round events, economy, HUD)", expanded=True):
        st.code("\n\n".join(text_blocks), language="text")


def render_map(map_name: str) -> None:
    """Show the radar image for this map. Player-position dots require demo
    parquets locally — added when those are pulled down (export_viewer_data.py
    in cs2-tools handles that pipeline)."""
    png = get_radar_png_bytes(map_name)
    if png is None:
        st.caption(
            f"_(radar for `{map_name}` not available locally — "
            "install `awpy` or run `awpy get maps`)_"
        )
        return
    st.image(png, caption=f"{map_name} radar (positions pending demo parquets)", width=420)


def render_state_card(pair: dict[str, Any], smoke_path: str) -> None:
    s = pair.get("state", {})
    c = pair.get("context", {})
    util = s.get("utility") or []
    util_str = ", ".join(util) if util else "none"
    visible = s.get("visible_enemies")
    visible_str = (
        ", ".join(visible) if isinstance(visible, list) and visible else str(visible)
    )

    st.markdown("### Game state")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Map:** `{s.get('map_name', '?')}`")
        st.markdown(f"**Phase:** `{s.get('round_phase', '?')}`")
        st.markdown(f"**Side:** `{s.get('player_side', '?')}`")
        st.markdown(f"**Bomb:** `{s.get('bomb_status', '?')}`")
    with col2:
        st.markdown(
            f"**HP/Armor:** `{s.get('player_health', '?')} / "
            f"{s.get('player_armor', '?')}`"
        )
        st.markdown(f"**Money:** `${s.get('player_money', '?')}`")
        st.markdown(
            f"**Alive (T/E):** "
            f"`{s.get('alive_teammates', '?')} / {s.get('alive_enemies', '?')}`"
        )
        st.markdown(f"**Visible enemies:** `{visible_str}`")
    with col3:
        st.markdown(f"**Primary:** `{s.get('weapon_primary', '?')}`")
        st.markdown(f"**Secondary:** `{s.get('weapon_secondary', '?')}`")
        st.markdown(f"**Utility:** `{util_str}`")

    won = c.get("round_won")
    won_icon = "win" if won is True else ("loss" if won is False else "?")
    won_emoji = "✅" if won is True else ("❌" if won is False else "")
    cats = c.get("pro_action_categories") or []
    cat_str = ", ".join(cats) if cats else "(none)"
    desc = c.get("pro_action_description") or "(none)"
    st.markdown("### Pro player context")
    st.markdown(
        f"**Demo:** `{c.get('demo_stem', '?')}`  "
        f"**Round:** `{c.get('round_num', '?')}`  "
        f"**Tick:** `{c.get('tick', '?')}`  "
        f"**Player:** `{c.get('player_name', '?')}`  "
        f"**Outcome:** {won_emoji} ({won_icon})"
    )
    st.markdown(f"**Categories:** `{cat_str}`")
    st.markdown(f"**Pro action description:** {desc}")

    # Map + full prompt context.
    map_col, ctx_col = st.columns([1, 2])
    with map_col:
        render_map(s.get("map_name", "") or "")
    with ctx_col:
        sidx = pair.get("sample_idx")
        if isinstance(sidx, int):
            render_round_context(sidx, smoke_path)


def render_pair_panel(
    pair: dict[str, Any], labeler: str, prefs_path: Path, smoke_path: str
) -> None:
    pid = pair["pair_id"]

    # Re-shuffle only when the displayed pair_id changes. Pretty-print JSON
    # bodies once at shuffle time so diffing the two completions is bearable.
    if st.session_state.get("displayed_pair_id") != pid:
        left, right, ui_a_was = shuffle_for_display(pair)
        st.session_state["displayed_pair_id"] = pid
        st.session_state["left_text"] = pretty_json(left)
        st.session_state["right_text"] = pretty_json(right)
        st.session_state["ui_a_was_originally"] = ui_a_was
        st.session_state["render_ts"] = time.time()
        st.session_state["notes"] = ""

    render_state_card(pair, smoke_path)
    st.markdown("---")
    st.markdown("### Compare completions")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Left  (press *A is better* if you prefer this)")
        st.code(st.session_state["left_text"], language="json")
    with col_b:
        st.markdown("#### Right  (press *B is better* if you prefer this)")
        st.code(st.session_state["right_text"], language="json")

    notes = st.text_input(
        "Notes (optional)", value=st.session_state.get("notes", ""), key=f"notes_{pid}"
    )

    b1, b2, b3, b4 = st.columns(4)
    clicked: str | None = None
    with b1:
        if st.button("← A is better", key=f"btn_a_{pid}", use_container_width=True):
            clicked = "A"
    with b2:
        if st.button("B is better →", key=f"btn_b_{pid}", use_container_width=True):
            clicked = "B"
    with b3:
        if st.button("Tie / Equivalent", key=f"btn_t_{pid}", use_container_width=True):
            clicked = "tie"
    with b4:
        if st.button("Skip / Unsure", key=f"btn_s_{pid}", use_container_width=True):
            clicked = "skip"

    if clicked is not None:
        elapsed = time.time() - st.session_state.get("render_ts", time.time())
        ui_a_was = st.session_state["ui_a_was_originally"]
        choice = canonical_choice(clicked, ui_a_was)
        row = {
            "pair_id": pid,
            "choice": choice,
            "time_seconds": round(elapsed, 2),
            "labeled_at": datetime.now().isoformat(timespec="seconds"),
            "ui_a_was_originally": ui_a_was,
            "labeler": labeler,
            "notes": notes or "",
        }
        append_preference(prefs_path, row)
        # Force re-pick of next pair on next run.
        st.session_state["labeled_just_now"] = pid
        st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    st.set_page_config(
        page_title="CS2 advice pairwise labeling", layout="wide"
    )
    st.title("CS2 advice pairwise labeling")
    st.caption(
        f"labeler: `{args.labeler}` -- candidates: `{args.candidates}` -- "
        f"preferences: `{args.preferences}`"
    )

    pairs = load_candidates(args.candidates)
    labeled, today_count = load_labeled_ids(args.preferences)
    total = len(pairs)
    done = len(labeled)
    remaining = total - done

    # Progress strip.
    pct = (done / total) if total else 0.0
    st.progress(min(max(pct, 0.0), 1.0))
    st.markdown(
        f"**Progress:** {done} / {total} labeled "
        f"({remaining} remaining)  --  **Labeled today:** {today_count}"
    )

    if remaining == 0:
        st.success("All pairs are labeled. Nothing left to do here.")
        return

    next_pair = select_next_pair(pairs, labeled)
    if next_pair is None:
        st.success("All pairs are labeled. Nothing left to do here.")
        return

    info = next_pair.get("informativeness_score", 0.0)
    st.markdown(
        f"**Pair `{next_pair['pair_id']}`**  --  "
        f"informativeness: `{info:.3f}`  --  "
        f"sample_idx: `{next_pair.get('sample_idx', '?')}`  "
        f"source_run: `{next_pair.get('source_run', '?')}`"
    )

    render_pair_panel(
        next_pair,
        labeler=args.labeler,
        prefs_path=args.preferences,
        smoke_path=str(args.smoke_dataset),
    )


if __name__ == "__main__":
    # Streamlit invokes this module as a script; argparse reads sys.argv after `--`.
    main()
