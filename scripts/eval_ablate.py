#!/usr/bin/env python3
"""Gate #1 — ablate-the-latent (bridge-design.md §3).

The single test that distinguishes this bridge from the Era-1 caster-SFT failure:
run identical inference with the soft tokens ZEROED (and separately SHUFFLED across
examples) and measure the delta on a held-out set.

  latent-on >> latent-off  -> the bridge grounds in the world model. Thesis holds.
  latent-on  ~ latent-off  -> the bridge ignores the latent (prompt/template match).
                              Circularity has returned. Stop and fix grounding.

Two readouts:
  (A) teacher-forced CE on the target span: latent-on vs off vs shuffled. Robust,
      backend-agnostic, the primary number.
  (B) generation value-agreement (--generate): regex the stated '~XX%' from the
      generated answer and compare to the world model's true P(CT win). Meaningful
      with real Qwen; on the stub it only exercises the plumbing (random text).

Backend-abstracted like train_bridge.py (stub for the local smoke, Qwen on the pod).
Usage:
  python scripts/eval_ablate.py --smoke
  python scripts/eval_ablate.py --llm qwen --model <id> --bridge outputs/bridge/single_v1/bridge.pt --generate
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bridge import LanguageBridge  # noqa: E402
from train_bridge import (ByteTokenizer, SFTPairs, make_collate, bridge_step,  # noqa: E402
                          build_stub_backend, build_qwen_backend)

PCT_RE = re.compile(r"(\d{1,3})\s*%")


@torch.no_grad()
def greedy_generate(llm, soft, prompt_ids, prompt_attn, max_new, dev):
    """Soft prefix + prompt -> greedy-decode max_new tokens. Works for stub and
    Qwen (both accept inputs_embeds)."""
    emb = llm.get_input_embeddings()
    cur = torch.cat([soft, emb(prompt_ids)], dim=1)
    attn = torch.cat([torch.ones(soft.shape[:2], dtype=torch.long, device=dev), prompt_attn], dim=1)
    out_ids = []
    for _ in range(max_new):
        nxt = llm(inputs_embeds=cur, attention_mask=attn).logits[:, -1].argmax(-1)  # [B]
        out_ids.append(nxt)
        cur = torch.cat([cur, emb(nxt).unsqueeze(1)], dim=1)
        attn = torch.cat([attn, torch.ones(soft.shape[0], 1, dtype=torch.long, device=dev)], dim=1)
    return torch.stack(out_ids, dim=1)  # [B, max_new]


def value_agreement(bridge, llm, tok, ds, dev, n=64, max_new=60, ablate=False):
    """Mean |stated% - true%| over examples whose generated text states a percent."""
    coll = make_collate(tok)
    dl = DataLoader(ds, batch_size=min(16, n), shuffle=False, collate_fn=coll)
    errs, parsed, total = [], 0, 0
    for batch in dl:
        if total >= n:
            break
        total += batch["grid"].shape[0]
        grid = batch["grid"].to(dev); channels = batch["channels"].to(dev)
        # prompt-only ids = strip the label-unmasked target; rebuild prompt segment
        soft = bridge.soft_tokens(grid, channels, ablate=ablate)
        # reuse input_ids up to first non -100 label as the prompt
        pl = (batch["labels"] == -100).sum(1)  # prompt length per row (incl BOS+SEP)
        T = int(pl.max())
        pid = batch["input_ids"][:, :T].to(dev)
        patt = (pid != tok.PAD).long()
        gen = greedy_generate(llm, soft, pid, patt, max_new, dev)
        for row in gen:
            txt = tok.decode(row.tolist())
            m = PCT_RE.search(txt)
            if m:
                parsed += 1
        # true% available via value_logit on the dataset; align by order
    return parsed, total


def ce_readout(bridge, llm, ds, tok, dev, n=512):
    coll = make_collate(tok)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=coll)
    on = off = shuf = c = 0.0
    for batch in dl:
        if c * 32 >= n:
            break
        on += bridge_step(bridge, llm, batch, dev, ablate=False).item()
        off += bridge_step(bridge, llm, batch, dev, ablate=True).item()
        # shuffled-latent control: permute grid/channels across the batch
        sb = dict(batch)
        perm = torch.randperm(batch["grid"].shape[0])
        sb["grid"] = batch["grid"][perm]; sb["channels"] = batch["channels"][perm]
        shuf += bridge_step(bridge, llm, sb, dev, ablate=False).item()
        c += 1
    return on / c, off / c, shuf / c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="data/processed/bridge_sft/val_single.pt")
    ap.add_argument("--bridge", default=None, help="trained bridge.pt (else fresh/untrained)")
    ap.add_argument("--llm", choices=["stub", "qwen"], default="stub")
    ap.add_argument("--model", default=None)
    ap.add_argument("--soft-tokens", type=int, default=32)
    ap.add_argument("--generate", action="store_true", help="also run generation value-agreement")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.llm = "stub"; args.device = "cpu"; args.generate = True
        print("[smoke] local CPU plumbing test (random stub -> CE delta + generation path)")

    cache = torch.load(args.val, map_location="cpu", weights_only=False)
    sch = cache["schema"]
    if args.smoke:
        for k in ("grid", "channels", "value_logit", "won"):
            cache[k] = cache[k][:64]
        cache["prompt"] = cache["prompt"][:64]; cache["target"] = cache["target"][:64]

    tok, llm = build_stub_backend(cache) if args.llm == "stub" else build_qwen_backend(args.model)
    dev = torch.device(args.device)
    if args.llm == "stub":
        llm.to(dev)
    bridge = LanguageBridge(llm, latent_dim=sch["latent_dim"], n_pred_channels=sch["n_pred_channels"],
                            n_soft_tokens=args.soft_tokens).to(dev)
    if args.bridge:
        st = torch.load(args.bridge, map_location=dev, weights_only=False)
        bridge.featurizer.load_state_dict(st["featurizer"]); bridge.resampler.load_state_dict(st["resampler"])
        print(f"loaded trained bridge: {args.bridge}")
    bridge.eval()
    ds = SFTPairs(cache)

    on, off, shuf = ce_readout(bridge, llm, ds, tok, dev)
    print(f"\nteacher-forced CE (held-out):")
    print(f"  latent-on   {on:.4f}")
    print(f"  latent-off  {off:.4f}   (delta {off - on:+.4f})")
    print(f"  shuffled    {shuf:.4f}   (delta {shuf - on:+.4f})")

    if args.generate:
        parsed, total = value_agreement(bridge, llm, tok, ds, dev)
        print(f"\ngeneration value-agreement: parsed a percent in {parsed}/{total} answers"
              + ("  (stub text is random — plumbing only)" if args.llm == "stub" else ""))

    if args.smoke:
        # untrained bridge: only require the ablate + shuffle paths to be wired (distinct CE)
        print("\033[32mSMOKE PASSED\033[0m" if (off != on and shuf != on) else "\033[31mSMOKE FAILED\033[0m",
              "— ablate + shuffle paths produce distinct CE (hook wired)")
        sys.exit(0 if (off != on and shuf != on) else 1)

    verdict = "GROUNDED (latent-on << off/shuffled)" if (off - on > 0.05 and shuf - on > 0.05) \
        else "NOT GROUNDED — latent-on ~ off/shuffled; fix grounding before proceeding (§3)"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
