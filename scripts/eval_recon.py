#!/usr/bin/env python3
"""Gate #2 — recon-fidelity / NLA faithfulness (bridge-design.md §2b, §3 eval #2).

Ablate (#1) asks "does the bridge USE the latent?"; recon asks the orthogonal
question "is the latent FAITHFULLY RENDERED in the text, or hallucinated past?"
A SEPARATE text-only decoder reconstructs the world-model latent from the bridge's
*generated* answer string (ids only — the firewall), trained on FROZEN verbalizer
outputs (no gradient to the bridge — a jointly-trained decoder colludes into a
private cipher, the documented NLA failure mode).

Pipeline:
  1. generate frozen-bridge answers (latent-on and latent-off) for held-out moments
  2. train a fresh NLADecoder on (generated-ids -> world-model latent z), bridge frozen
  3. score, as fraction-of-variance-explained over the latent-mean floor (§3c):
       real text   vs   shuffled-text   vs   empty-text   vs   ablated-bridge text
  4. value-head-agreement: pooled z_hat through the FROZEN value_head -> P(CT win) AUC
       (the faithfulness-gibberish guard — recon alone can be satisfied by a cipher)

Green light (§3): recon(real) > recon(shuffled/ablated) above the floor, AND
value-head-agreement holds. Backend-abstracted (stub smoke / Qwen on the pod).
Usage: python scripts/eval_recon.py --smoke
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bridge import LanguageBridge, NLADecoder, recon_loss, fraction_variance_explained  # noqa: E402
from src.bridge.wm_interface import load_world_model  # noqa: E402
from train_bridge import (SFTPairs, make_collate, build_stub_backend, build_qwen_backend)  # noqa: E402
from eval_ablate import greedy_generate  # noqa: E402
from src.bridge.featurizer import N_TOKENS  # noqa: E402


@torch.no_grad()
def generate_all(bridge, llm, tok, ds, dev, max_new, ablate):
    """Frozen-bridge answers for every example -> [N, max_new] LongTensor ids."""
    from torch.utils.data import DataLoader
    out = []
    for batch in DataLoader(ds, batch_size=16, shuffle=False, collate_fn=make_collate(tok)):
        grid = batch["grid"].to(dev); channels = batch["channels"].to(dev)
        soft = bridge.soft_tokens(grid, channels, ablate=ablate)
        pl = (batch["labels"] == -100).sum(1); T = int(pl.max())
        pid = batch["input_ids"][:, :T].to(dev); patt = (pid != tok.PAD).long()
        out.append(greedy_generate(llm, soft, pid, patt, max_new, dev).cpu())
    return torch.cat(out)


def auc(scores, labels):
    order = torch.argsort(scores); ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    pos = labels > 0.5; npos = int(pos.sum()); nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum().item() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="data/processed/bridge_sft/val_single.pt")
    ap.add_argument("--bridge", default=None)
    ap.add_argument("--llm", choices=["stub", "qwen"], default="stub")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ckpt", default=None, help="WM ckpt for value_head (else from cache schema)")
    ap.add_argument("--soft-tokens", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--recon-steps", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.llm = "stub"; args.device = "cpu"; args.max_new = 24; args.recon_steps = 200
        print("[smoke] local CPU plumbing + firewall test (random stub text won't carry z)")

    cache = torch.load(args.val, map_location="cpu", weights_only=False)
    sch = cache["schema"]
    if args.smoke:
        for k in ("grid", "channels", "value_logit", "won"):
            cache[k] = cache[k][:128]
        cache["prompt"] = cache["prompt"][:128]; cache["target"] = cache["target"][:128]
    dev = torch.device(args.device)

    tok, llm = build_stub_backend(cache) if args.llm == "stub" else build_qwen_backend(args.model)
    if args.llm == "stub":
        llm.to(dev)
    bridge = LanguageBridge(llm, latent_dim=sch["latent_dim"], n_pred_channels=sch["n_pred_channels"],
                            n_soft_tokens=args.soft_tokens).to(dev)
    if args.bridge:
        st = torch.load(args.bridge, map_location=dev, weights_only=False)
        bridge.featurizer.load_state_dict(st["featurizer"]); bridge.resampler.load_state_dict(st["resampler"])
    bridge.eval()
    ds = SFTPairs(cache)

    # 1. frozen-bridge generations (latent-on and ablated)
    y_on = generate_all(bridge, llm, tok, ds, dev, args.max_new, ablate=False)
    y_off = generate_all(bridge, llm, tok, ds, dev, args.max_new, ablate=True)
    z = cache["grid"].float()                                   # [N,11,512] recon target
    won = cache["won"].float()
    N = len(z); ntr = int(0.8 * N)
    idx = torch.randperm(N); tr, te = idx[:ntr], idx[ntr:]
    z_mean = z[tr].mean(0, keepdim=True)

    # 2. train a SEPARATE decoder on frozen-bridge real text (no grad to bridge)
    d_txt = llm.config.hidden_size
    dec = NLADecoder(vocab_size=tok.vocab_size, d_txt=d_txt, target_tokens=N_TOKENS,
                     target_dim=sch["latent_dim"],
                     frozen_embedding=llm.get_input_embeddings().weight.detach()).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-3)
    y_on, z = y_on.to(dev), z.to(dev)
    for step in range(args.recon_steps):
        b = tr[torch.randint(0, len(tr), (min(32, len(tr)),))]
        loss, _, _ = recon_loss(dec(y_on[b]), z[b], beta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()

    # FIREWALL audit: recon loss must leak NO gradient into the bridge
    bridge.zero_grad()
    recon_loss(dec(y_on[te[:8]]), z[te[:8]])[0].backward()
    leak = any(p.grad is not None and p.grad.abs().sum() > 0
               for p in list(bridge.featurizer.parameters()) + list(bridge.resampler.parameters()))

    # 3. score real vs controls (fraction-variance-explained over the latent-mean floor)
    dec.eval()
    with torch.no_grad():
        zt = z[te]; zm = z_mean.to(dev)
        fve = lambda yids: fraction_variance_explained(dec(yids), zt, zm)
        r_real = fve(y_on[te])
        r_shuf = fve(y_on[te][torch.randperm(len(te))])
        r_empty = fve(torch.zeros(len(te), 1, dtype=torch.long, device=dev))
        r_abl = fve(y_off.to(dev)[te])
        # value-head-agreement: pooled z_hat -> frozen value_head -> AUC vs outcome
        ck = args.ckpt or sch["ckpt"]
        wm, _ = load_world_model(ck, device=str(dev))
        zhat_pool = dec(y_on[te]).mean(1)                       # [n,512]
        v_auc = auc(wm.value_head(zhat_pool).squeeze(1).cpu(), won[te])

    print(f"\nrecon-fidelity (fraction-variance-explained over latent-mean floor):")
    print(f"  real text     {r_real:+.3f}")
    print(f"  shuffled text {r_shuf:+.3f}   (must collapse to ~0)")
    print(f"  empty text    {r_empty:+.3f}   (firewall floor)")
    print(f"  ablated text  {r_abl:+.3f}   (delta real-ablated {r_real - r_abl:+.3f})")
    print(f"value-head-agreement: AUC(value_head(z_hat), outcome) = {v_auc:.3f}")
    print(f"firewall: recon loss leaks gradient into bridge? {'YES (BROKEN)' if leak else 'no'}")

    if args.smoke:
        # plumbing + firewall test. shuffled/empty must be AT OR BELOW the floor
        # (FVE <= ~0; negative = worse-than-mean = collapsed). Real ~ floor is
        # EXPECTED on the random stub (text carries no z); real fidelity needs Qwen.
        ok = (not leak) and r_shuf <= 0.05 and r_empty <= 0.05 and (r_real > r_shuf)
        print("\033[32mSMOKE PASSED\033[0m — firewall holds; shuffled/empty collapse below floor "
              "(stub text carries no z, as expected)" if ok else "\033[31mSMOKE FAILED\033[0m")
        sys.exit(0 if ok else 1)

    faithful = (r_real - r_shuf > 0.05) and (r_real - r_abl > 0.03) and not leak
    print(f"\nVERDICT: {'FAITHFUL (real > shuffled/ablated above floor)' if faithful else 'NOT FAITHFUL — recon does not beat controls (§3 eval #2)'}"
          + "  [pair with value-agreement + readability before claiming the gate]")


if __name__ == "__main__":
    main()
