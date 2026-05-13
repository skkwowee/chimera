#!/usr/bin/env python3
"""Streamlit UI for authoring the D_plausible_wrong slot in pseudo-gold.

The pseudo-gold offline benchmark needs four advice candidates per state.
A_correct (parrots pro), B_anti_pro (inverted), and C_generic (vague) are
filled mechanically by scripts/draft_pseudo_gold_abc.py. The fourth —
D_plausible_wrong — needs CS2 domain knowledge: confident-sounding advice
that's tactically WRONG for THIS state.

This app walks you through every record still needing a D, shows the full
state and the existing A_correct as a reference for what "right" looks like,
and lets you type the D in a textarea. Save & Next moves on. Resumable.

INSTALL:
    pip install streamlit

RUN:
    streamlit run scripts/label_d_app.py
    # or with custom path:
    streamlit run scripts/label_d_app.py -- --path data/eval/pseudo_gold_stub.jsonl

After authoring: python scripts/eval_scorer.py data/eval/pseudo_gold_stub.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import streamlit as st


DEFAULT_PATH = Path("data/eval/pseudo_gold_stub.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args, _ = parser.parse_known_args()
    return args


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_records(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def is_todo(value: str) -> bool:
    return str(value).startswith("TODO")


def render_state(rec: dict) -> None:
    gs = rec.get("game_state", {})
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"**Map:** `{gs.get('map_name', '?')}`")
        st.markdown(f"**Side:** `{gs.get('player_side', '?')}`")
        st.markdown(f"**Phase:** `{gs.get('round_phase', '?')}`")
        st.markdown(f"**Bomb:** `{gs.get('bomb_status', '?')}`")
    with cols[1]:
        st.markdown(f"**HP/Armor:** `{gs.get('player_health', '?')} / {gs.get('player_armor', '?')}`")
        st.markdown(f"**Money:** `${gs.get('player_money', '?')}`")
        st.markdown(
            f"**Alive:** `{gs.get('alive_teammates', 0) + 1}T vs {gs.get('alive_enemies', '?')}CT`"
            if gs.get('player_side', '').upper() == 'T'
            else f"**Alive:** `{gs.get('alive_enemies', '?')}T vs {gs.get('alive_teammates', 0) + 1}CT`"
        )
        st.markdown(f"**Visible enemies:** `{gs.get('visible_enemies', '?')}`")
    with cols[2]:
        st.markdown(f"**Primary:** `{gs.get('weapon_primary', '?')}`")
        st.markdown(f"**Secondary:** `{gs.get('weapon_secondary', '?')}`")
        util = gs.get("utility", []) or []
        util_str = ", ".join(util) if util else "(none)"
        st.markdown(f"**Utility:** `{util_str}`")
        st.markdown(f"**round_won:** `{rec.get('round_won')}`")


def render_pro_action(rec: dict) -> None:
    pa = rec.get("pro_action", {})
    st.markdown("**Pro action:**")
    st.markdown(f"> {pa.get('description', '(no description)')}")
    cols = st.columns(2)
    with cols[0]:
        st.caption(f"categories: {pa.get('categories', [])}")
    with cols[1]:
        b = pa.get("behavior", {})
        st.caption(
            f"movement={b.get('movement_direction', 0)}  "
            f"objective={b.get('objective_direction', 0)}  "
            f"util_used={b.get('utility_used', [])}  "
            f"initiated={b.get('initiated_engagement', False)}"
        )


def render_round_context(rec: dict) -> None:
    summary = rec.get("state_summary", "")
    if summary:
        with st.expander("Round context (events + economy)", expanded=False):
            st.code(summary, language="text")


def render_record(rec: dict, n: int, total_remaining: int, total_all: int, idx_in_full: int) -> str | None:
    """Render the full record UI. Returns new D text if Save clicked, else None."""
    cands = rec.get("candidates", {})

    st.subheader(
        f"Record {n + 1} of {total_remaining} remaining  "
        f"(global #{idx_in_full + 1} of {total_all})"
    )

    render_state(rec)
    st.divider()
    render_round_context(rec)
    render_pro_action(rec)
    st.divider()

    st.markdown("**Reference candidates** (already filled, just for context):")
    st.success(f"**A_correct (should rank HIGH):** {cands.get('A_correct', '<missing>')}")
    with st.expander("B_anti_pro (should rank LOW)", expanded=False):
        st.markdown(cands.get("B_anti_pro", "<missing>"))
    with st.expander("C_generic (should rank MID-LOW)", expanded=False):
        st.markdown(cands.get("C_generic", "<missing>"))
    st.divider()

    st.markdown("**Your job: write D_plausible_wrong**")
    st.caption(
        "Confident-sounding CS2 advice that's tactically WRONG for THIS state. "
        "Specific (not generic). Uses real CS2 vocabulary. Could plausibly come "
        "from a tier-2 IGL but is the wrong call for the state shown above. "
        "Not just the inverted action (that's B)."
    )

    existing_d = cands.get("D_plausible_wrong", "")
    placeholder = "" if is_todo(existing_d) else existing_d

    d_text = st.text_area(
        "D_plausible_wrong",
        value=placeholder,
        height=120,
        key=f"d_text_{idx_in_full}",
        label_visibility="collapsed",
    )

    col_save, col_skip, col_back = st.columns([1, 1, 1])
    saved_text = None
    with col_save:
        if st.button("💾 Save & Next", type="primary", use_container_width=True, key=f"save_{idx_in_full}"):
            if d_text.strip():
                saved_text = d_text.strip()
            else:
                st.warning("Type something before saving.")
    with col_skip:
        if st.button("⏭ Skip (leave TODO)", use_container_width=True, key=f"skip_{idx_in_full}"):
            st.session_state["nav"] = "next"
            st.rerun()
    with col_back:
        if st.button("⬅ Previous", use_container_width=True, key=f"back_{idx_in_full}"):
            st.session_state["nav"] = "prev"
            st.rerun()

    return saved_text


def main() -> None:
    args = parse_args()
    path = args.path

    st.set_page_config(page_title="Pseudo-gold D labeler", layout="wide")

    if not path.exists():
        st.error(f"{path} not found. Run scripts/build_pseudo_gold.py first.")
        st.stop()

    records = load_records(path)
    pending = [i for i, r in enumerate(records)
               if is_todo(r.get("candidates", {}).get("D_plausible_wrong", "TODO"))]
    done = len(records) - len(pending)

    # Sidebar: progress + nav
    with st.sidebar:
        st.title("Pseudo-gold D")
        st.metric("Authored", f"{done} / {len(records)}")
        st.progress(done / len(records) if records else 0)
        if not pending:
            st.success("All done.")
            st.info(
                "Next step:\n\n"
                f"```\npython scripts/eval_scorer.py {path}\n```"
            )
            st.stop()

        st.caption(f"Remaining: {len(pending)}")
        sel_label = st.selectbox(
            "Jump to remaining record",
            options=list(range(len(pending))),
            format_func=lambda i: (
                f"#{i + 1}  "
                f"({records[pending[i]].get('game_state', {}).get('map_name', '?')} "
                f"{records[pending[i]].get('game_state', {}).get('player_side', '?')})"
            ),
            index=st.session_state.get("cursor", 0),
        )
        st.session_state["cursor"] = sel_label

    # Handle nav from prior frame
    nav = st.session_state.pop("nav", None)
    cursor = st.session_state.get("cursor", 0)
    if nav == "next":
        cursor = min(cursor + 1, len(pending) - 1)
    elif nav == "prev":
        cursor = max(cursor - 1, 0)
    st.session_state["cursor"] = cursor

    idx_in_full = pending[cursor]
    rec = records[idx_in_full]

    saved = render_record(rec, cursor, len(pending), len(records), idx_in_full)

    if saved is not None:
        records[idx_in_full]["candidates"]["D_plausible_wrong"] = saved
        save_records(path, records)
        st.toast(f"Saved record #{idx_in_full + 1}.", icon="✅")
        # Advance to next remaining
        st.session_state["nav"] = "next"
        st.rerun()


if __name__ == "__main__":
    main()
