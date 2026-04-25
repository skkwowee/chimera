#!/usr/bin/env python3
"""Train a Bradley-Terry preference head on human-labeled CS2 advice pairs.

Head: small MLP over [state_emb || response_emb] from two FROZEN encoders.
  state encoder    -- outputs/embedding/learned_v3_alive (CS2 state -> 384-d)
  response encoder -- all-MiniLM-L6-v2 (advice text -> 384-d)
Loss: BT pairwise  -- -log sigma(r_chosen - r_rejected)

Inputs (jsonl):
  preferences.jsonl     : {pair_id, choice in {A,B,tie,skip}, ui_a_was_originally, ...}
  candidate_pairs.jsonl : {pair_id, state, completion_a, completion_b, ...}

ui_a_was_originally resolves the UI shuffle. If "B", the labeler's "A" choice
maps to the canonical completion_b. We translate every label back to canonical
(chosen, rejected) text before training.

Outputs (in --output):
  model.safetensors   -- MLP weights only (encoders referenced by path)
  config.json         -- arch, encoder paths, sizes, final val_acc, label mix
  training_log.jsonl  -- per-epoch loss + val_acc
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Heavy deps (torch / sentence_transformers) imported inside main() so
# --help and py_compile stay light.


# ---- state serialization (must match learned_v3_alive training) ------------
# Mirrors src/training/learned_state_embedder.py exactly; any drift here
# silently degrades the head. Keep in lockstep with that file.
_REDACT_KEYS = frozenset({
    "player_name", "player", "teammates", "team", "opponent_team",
    "round_number", "round",
    "time_remaining", "time", "round_time", "tick",
    "score", "current_player",
})


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _REDACT_KEYS}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def state_text(state: dict[str, Any], redact: bool = True) -> str:
    """JSON-serialize a game state the same way learned_v3_alive was trained on."""
    data = _strip(state) if redact else state
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


# ---- IO helpers ------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_pair(pref: dict, cand: dict) -> tuple[str, str] | None:
    """Translate a UI-space label to canonical (chosen_text, rejected_text).

    ui_a_was_originally == "A": UI showed canonical_a as "A" -- identity.
    ui_a_was_originally == "B": UI swapped them, so labeler's "A" -> canonical_b.
    """
    choice = pref.get("choice")
    if choice not in ("A", "B"):
        return None
    a, b = cand["completion_a"], cand["completion_b"]
    if pref.get("ui_a_was_originally", "A") == "B":
        a, b = b, a  # ui_a_text, ui_b_text
    return (a, b) if choice == "A" else (b, a)


# ---- model -----------------------------------------------------------------
def build_bt_head(state_dim: int, response_dim: int, hidden: int, dropout: float):
    import torch
    import torch.nn as nn

    class BTHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(state_dim + response_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden),                   nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            self._state_encoder = None  # attached at inference time, FROZEN
            self._response_encoder = None

        def attach_encoders(self, state_encoder, response_encoder):
            self._state_encoder = state_encoder
            self._response_encoder = response_encoder

        def score_embeddings(self, state_emb, response_emb):
            return self.mlp(torch.cat([state_emb, response_emb], dim=-1)).squeeze(-1)

        def forward(self, state_dict: dict, completion: str) -> float:
            assert self._state_encoder is not None and self._response_encoder is not None, (
                "BTHead.forward requires attach_encoders() first"
            )
            with torch.no_grad():
                s = self._state_encoder.encode(
                    [state_text(state_dict)], convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False,
                )
                r = self._response_encoder.encode(
                    [completion], convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False,
                )
                device = next(self.mlp.parameters()).device
                s_t = torch.from_numpy(s).float().to(device)
                r_t = torch.from_numpy(r).float().to(device)
            return float(self.score_embeddings(s_t, r_t).item())

    return BTHead()


def bt_head_score(model, state: dict, completion: str) -> float:
    """Importable convenience for GRPO integration."""
    return model(state, completion)


def load_bt_head(head_dir: str | Path):
    """Reconstruct a trained BTHead from disk (config.json + model.safetensors).

    Loads encoders from the paths recorded at training time, rebuilds the MLP
    with matching dims, restores weights, attaches the encoders, and returns a
    ready-to-score model in eval mode. The reward integration in
    src/training/bt_reward.py imports this.
    """
    import torch
    from safetensors.torch import load_file as st_load_file

    head_dir = Path(head_dir)
    cfg = json.loads((head_dir / "config.json").read_text(encoding="utf-8"))
    arch = cfg["arch"]
    encs = cfg["encoders"]

    state_encoder = _load_encoder(encs["state_encoder_path"], "state-encoder")
    response_encoder = _load_encoder(encs["response_encoder_path"], "response-encoder")

    model = build_bt_head(
        state_dim=int(arch["state_dim"]),
        response_dim=int(arch["response_dim"]),
        hidden=int(arch["hidden_dim"]),
        dropout=float(arch["dropout"]),
    )
    state_dict = st_load_file(str(head_dir / "model.safetensors"))
    model.mlp.load_state_dict(state_dict)
    model.attach_encoders(state_encoder, response_encoder)
    model.eval()
    return model


# ---- encoder loading -------------------------------------------------------
def _looks_like_st_dir(p: Path) -> bool:
    return p.is_dir() and (
        (p / "modules.json").exists() or (p / "config_sentence_transformers.json").exists()
    )


def _load_encoder(name_or_path: str, label: str):
    from sentence_transformers import SentenceTransformer

    p = Path(name_or_path)
    if p.exists():
        if not _looks_like_st_dir(p):
            print(f"ERROR: --{label} path {p} is not a saved sentence-transformer "
                  f"(no modules.json / config_sentence_transformers.json).", file=sys.stderr)
            sys.exit(2)
        return SentenceTransformer(str(p))
    if label == "state-encoder":
        # learned_v3_alive is locally trained; refuse to silently fall back to a hub model.
        print(f"ERROR: --state-encoder path {p} not found. Required: locally-trained "
              f"learned_v3_alive encoder.", file=sys.stderr)
        sys.exit(2)
    try:
        return SentenceTransformer(name_or_path)
    except Exception as e:
        print(f"ERROR: failed to load --{label} '{name_or_path}': {e}", file=sys.stderr)
        sys.exit(2)


# ---- main ------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a Bradley-Terry preference head on labeled CS2 advice pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--state-encoder", required=True,
                        help="Path to local sentence-transformer (e.g. learned_v3_alive)")
    parser.add_argument("--response-encoder", default="all-MiniLM-L6-v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=4,
                        help="Early-stop patience on val loss (epochs)")
    parser.add_argument("--tie-margin-weight", type=float, default=0.0,
                        help="If >0, include 'tie' pairs as a soft margin-to-zero loss")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --output if it already contains files")
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file as st_save_file

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    pref_path, cand_path, out_dir = Path(args.preferences), Path(args.candidates), Path(args.output)
    for p, label in [(pref_path, "preferences"), (cand_path, "candidates")]:
        if not p.exists():
            print(f"ERROR: {label} file not found: {p}", file=sys.stderr)
            return 2
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(f"ERROR: --output {out_dir} already non-empty. Pass --force to overwrite.",
              file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load encoders FIRST -- fail fast on missing models.
    print(f"[load] state encoder:    {args.state_encoder}")
    state_encoder = _load_encoder(args.state_encoder, "state-encoder")
    print(f"[load] response encoder: {args.response_encoder}")
    response_encoder = _load_encoder(args.response_encoder, "response-encoder")
    for enc in (state_encoder, response_encoder):
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()
    state_dim = int(state_encoder.get_sentence_embedding_dimension())
    response_dim = int(response_encoder.get_sentence_embedding_dimension())

    # Load preference + candidate data and tally label mix.
    prefs, cands = _load_jsonl(pref_path), _load_jsonl(cand_path)
    cands_by_id = {c["pair_id"]: c for c in cands}
    n_A = sum(1 for p in prefs if p.get("choice") == "A")
    n_B = sum(1 for p in prefs if p.get("choice") == "B")
    n_tie = sum(1 for p in prefs if p.get("choice") == "tie")
    n_skip = sum(1 for p in prefs if p.get("choice") == "skip")
    print(f"[data] labels: A={n_A} B={n_B} tie={n_tie} skip={n_skip}  prefs={len(prefs)}  cands={len(cands_by_id)}")

    # Resolve to canonical (chosen, rejected) form.
    examples: list[tuple[str, dict, str, str, str]] = []  # (pair_id, state, chosen, rejected, kind)
    missing = 0
    for p in prefs:
        pid = p.get("pair_id")
        c = cands_by_id.get(pid)
        if c is None:
            missing += 1
            continue
        choice = p.get("choice")
        if choice in ("A", "B"):
            chosen, rejected = _resolve_pair(p, c)  # type: ignore[misc]
            examples.append((pid, c["state"], chosen, rejected, "pref"))
        elif choice == "tie" and args.tie_margin_weight > 0:
            examples.append((pid, c["state"], c["completion_a"], c["completion_b"], "tie"))
        # "skip" / unknown: dropped.
    if missing:
        print(f"[warn] {missing} preference rows had no matching candidate pair_id")
    if len(examples) < 100:
        print(f"[warn] only {len(examples)} usable pairs -- data scarcity. Smoke run; trust val_acc weakly.")
    if not examples:
        print("ERROR: no usable training pairs after filtering.", file=sys.stderr)
        return 2

    # Split 80/20 by pair_id (sample-level overlap is fine; the head conditions on state).
    pids = sorted({ex[0] for ex in examples})
    random.Random(args.seed).shuffle(pids)
    n_val = max(1, int(round(len(pids) * args.val_split))) if len(pids) > 1 else 0
    val_pids = set(pids[:n_val])
    train_ex = [ex for ex in examples if ex[0] not in val_pids]
    val_ex = [ex for ex in examples if ex[0] in val_pids]
    print(f"[split] train={len(train_ex)} val={len(val_ex)} (val_split={args.val_split})")

    # Pre-encode once -- encoders are frozen, so re-encoding each epoch is waste.
    print("[encode] pre-computing frozen embeddings...")
    state_cache: dict[str, "torch.Tensor"] = {}
    resp_cache: dict[str, "torch.Tensor"] = {}

    def _enc_state(d: dict) -> "torch.Tensor":
        t = state_text(d)
        if t not in state_cache:
            v = state_encoder.encode([t], convert_to_numpy=True, normalize_embeddings=True,
                                     show_progress_bar=False)[0]
            state_cache[t] = torch.from_numpy(v).float()
        return state_cache[t]

    def _enc_resp(t: str) -> "torch.Tensor":
        if t not in resp_cache:
            v = response_encoder.encode([t], convert_to_numpy=True, normalize_embeddings=True,
                                        show_progress_bar=False)[0]
            resp_cache[t] = torch.from_numpy(v).float()
        return resp_cache[t]

    def _materialize(rows):
        if not rows:
            return (torch.empty(0, state_dim), torch.empty(0, response_dim),
                    torch.empty(0, response_dim), [])
        s = torch.stack([_enc_state(r[1]) for r in rows])
        c = torch.stack([_enc_resp(r[2]) for r in rows])
        rj = torch.stack([_enc_resp(r[3]) for r in rows])
        return s, c, rj, [r[4] for r in rows]

    s_train, c_train, r_train, k_train = _materialize(train_ex)
    s_val, c_val, r_val, k_val = _materialize(val_ex)
    print(f"[encode] cached states={len(state_cache)} responses={len(resp_cache)}")

    # Build head + optimizer (MLP only -- encoders frozen).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")
    model = build_bt_head(state_dim, response_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.mlp.parameters(), lr=args.lr)
    s_train, c_train, r_train = s_train.to(device), c_train.to(device), r_train.to(device)
    s_val, c_val, r_val = s_val.to(device), c_val.to(device), r_val.to(device)

    log_f = (out_dir / "training_log.jsonl").open("w", encoding="utf-8")

    def _eval():
        model.eval()
        if len(s_val) == 0:
            return 0.0, 0.0
        with torch.no_grad():
            diff = model.score_embeddings(s_val, c_val) - model.score_embeddings(s_val, r_val)
            mask = torch.tensor([k == "pref" for k in k_val], device=device)
            if not mask.any():
                return 0.0, 0.0
            d = diff[mask]
            return float(F.softplus(-d).mean().item()), float((d > 0).float().mean().item())

    best_val_loss, best_state, since_best, final_val_acc = float("inf"), None, 0, 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = torch.randperm(len(s_train), device=device)
        running, n_seen = 0.0, 0
        for start in range(0, len(idx), args.batch_size):
            b = idx[start:start + args.batch_size]
            kb = [k_train[i] for i in b.tolist()]
            diff = (model.score_embeddings(s_train[b], c_train[b])
                    - model.score_embeddings(s_train[b], r_train[b]))
            mask_pref = torch.tensor([k == "pref" for k in kb], device=device)
            mask_tie = torch.tensor([k == "tie" for k in kb], device=device)
            loss = torch.tensor(0.0, device=device)
            if mask_pref.any():
                loss = loss + F.softplus(-diff[mask_pref]).mean()
            if mask_tie.any() and args.tie_margin_weight > 0:
                loss = loss + args.tie_margin_weight * (diff[mask_tie] ** 2).mean()
            if loss.requires_grad:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += float(loss.item()) * len(b); n_seen += len(b)
        train_loss = running / max(1, n_seen)
        val_loss, val_acc = _eval()
        final_val_acc = val_acc
        log_f.write(json.dumps({"epoch": epoch, "train_loss": train_loss,
                                "val_loss": val_loss, "val_acc": val_acc}) + "\n")
        log_f.flush()
        print(f"[epoch {epoch:>2}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.mlp.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= args.patience and len(s_val) > 0:
                print(f"[early-stop] no improvement for {args.patience} epochs"); break
    log_f.close()

    if best_state is not None:
        model.mlp.load_state_dict(best_state)
    _, final_val_acc = _eval()

    # Calibration check: 5 confidence buckets by |r_chosen - r_rejected|.
    calibration: list[dict] = []
    if len(s_val) > 0:
        model.eval()
        with torch.no_grad():
            d = (model.score_embeddings(s_val, c_val) - model.score_embeddings(s_val, r_val))
            mask = torch.tensor([k == "pref" for k in k_val], device=device)
            if mask.any():
                diffs = d[mask].cpu().numpy()
                correct = (diffs > 0).astype("float32")
                conf = np.abs(diffs)
                order = np.argsort(conf)
                if len(order) >= 5:
                    edges = np.linspace(0, len(order), 6, dtype=int)
                    for i in range(5):
                        lo, hi = edges[i], edges[i + 1]
                        if hi > lo:
                            ix = order[lo:hi]
                            calibration.append({"bucket": i, "n": int(hi - lo),
                                                "avg_conf": float(conf[ix].mean()),
                                                "acc": float(correct[ix].mean())})

    # Save MLP weights + config.
    st_save_file({k: v.detach().cpu().contiguous() for k, v in model.mlp.state_dict().items()},
                 str(out_dir / "model.safetensors"))
    config = {
        "arch": {"state_dim": state_dim, "response_dim": response_dim,
                 "hidden_dim": args.hidden_dim, "dropout": args.dropout,
                 "n_hidden_layers": 2, "activation": "GELU"},
        "encoders": {"state_encoder_path": str(args.state_encoder),
                     "response_encoder_path": str(args.response_encoder),
                     "redact_state": True},
        "data": {"preferences_path": str(pref_path), "candidates_path": str(cand_path),
                 "n_train": len(train_ex), "n_val": len(val_ex),
                 "label_counts": {"A": n_A, "B": n_B, "tie": n_tie, "skip": n_skip},
                 "data_scarcity": len(examples) < 100},
        "training": {"lr": args.lr, "batch_size": args.batch_size, "epochs_cap": args.epochs,
                     "val_split": args.val_split, "seed": args.seed,
                     "tie_margin_weight": args.tie_margin_weight},
        "results": {"final_val_acc": final_val_acc,
                    "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
                    "calibration": calibration},
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Final stdout summary.
    print()
    print("=" * 60)
    print(f"BT head saved -> {out_dir}")
    print(f"  train pairs: {len(train_ex)}   val pairs: {len(val_ex)}")
    print(f"  label mix:   A={n_A} B={n_B} tie={n_tie} skip={n_skip}")
    print(f"  final val accuracy: {final_val_acc:.3f}")
    if calibration:
        print("  calibration (5 buckets, low->high confidence):")
        for b in calibration:
            print(f"    bucket {b['bucket']}: n={b['n']:>3}  "
                  f"avg_conf={b['avg_conf']:.3f}  acc={b['acc']:.3f}")
        accs = [b["acc"] for b in calibration]
        mono = all(accs[i] <= accs[i + 1] + 1e-6 for i in range(len(accs) - 1))
        print(f"  monotonic? {mono}  ({'well-calibrated' if mono else 'not strictly monotonic'})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
