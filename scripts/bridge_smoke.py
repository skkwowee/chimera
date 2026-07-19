#!/usr/bin/env python3
"""Phase-2 bridge scaffold — local CPU smoke (bridge-design.md §7 build order step 2).

Exercises the whole bridge on CPU against a tiny LLM stub (NOT Qwen) with zero pod
hours, and asserts the load-bearing invariants BEFORE any pod spend:

  A. FORWARD PATH  featurizer -> resampler -> soft-prefix -> stub LLM
     - shapes: grid [B,11,512] -> soft [B,M,d_llm] -> LM loss scalar
     - ablate hook wired: zeroing soft tokens changes the LM loss
     - gradients flow into featurizer + resampler from the LM CE loss

  B. NLA DECODER   text-only reverse-Perceiver (the faithfulness leg)
     - learnability: on a synthetic z=f(text) task, recon beats the latent-mean
       floor from REAL text
     - anti-gaming controls (§3): SHUFFLED-text and EMPTY-text recon collapse to
       the floor
     - FIREWALL (§2b): (1) float ids rejected; (2) recon loss produces NO gradient
       on featurizer/resampler — there is zero tensor path from text back to the latent

  C. REAL WORLD MODEL (optional, --ckpt): pull a real [B,11,512] grid + predictive
     channels from the frozen checkpoint and run one forward; shape + firewall only.

Usage:
  python scripts/bridge_smoke.py                 # synthetic, self-contained
  python scripts/bridge_smoke.py --ckpt outputs/wm_3map_dist_v3m/h8_mt/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bridge import LanguageBridge, NLADecoder, TinyLLMStub, fraction_variance_explained, recon_loss

OK, BAD = "  \033[32mPASS\033[0m", "  \033[31mFAIL\033[0m"
_fail = []


def check(name, cond):
    print((OK if cond else BAD) + f"  {name}")
    if not cond:
        _fail.append(name)


def part_a_forward(g):
    print("\nA. FORWARD PATH (featurizer -> resampler -> soft-prefix -> stub LLM)")
    B, M, d_proj = 4, 32, 512
    llm = TinyLLMStub(vocab_size=64, hidden_size=128, layers=2)
    bridge = LanguageBridge(llm, latent_dim=512, n_pred_channels=3, d_proj=d_proj,
                            n_soft_tokens=M)
    grid = torch.randn(B, 11, 512, generator=g)
    channels = torch.randn(B, 11, 3, generator=g)
    text_ids = torch.randint(0, 64, (B, 20), generator=g)

    soft = bridge.soft_tokens(grid, channels)
    check(f"soft-token shape == [B,M,d_llm] [{B},{M},{llm.config.hidden_size}]",
          tuple(soft.shape) == (B, M, llm.config.hidden_size))

    loss_on, _ = bridge.lm_forward(llm, text_ids, grid, channels, ablate=False)
    loss_off, _ = bridge.lm_forward(llm, text_ids, grid, channels, ablate=True)
    check("LM loss is a finite scalar", loss_on.dim() == 0 and torch.isfinite(loss_on))
    check("ablate hook wired (zeroed soft tokens change the loss)",
          abs(loss_on.item() - loss_off.item()) > 1e-6)

    bridge.zero_grad()
    loss_on.backward()
    feat_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in bridge.featurizer.parameters())
    res_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in bridge.resampler.parameters())
    check("CE loss produces gradient in featurizer", feat_grad)
    check("CE loss produces gradient in resampler", res_grad)
    return bridge, llm


def part_b_decoder(g, bridge, llm):
    print("\nB. NLA DECODER (text-only reverse-Perceiver + firewall)")
    V, d_txt, n_tok, dim = 64, 64, 11, 512
    # --- synthetic LEARNABLE task: z is a fixed function of the text ids, so a
    #     decoder CAN recover it from real text but NOT from shuffled/empty text. ---
    torch.manual_seed(0)
    E = torch.randn(V, d_txt) * 0.5                     # frozen text embedding
    W = torch.randn(d_txt, n_tok * dim) * 0.1           # frozen text-pool -> latent map
    def z_of(ids):
        return (E[ids].mean(1) @ W).reshape(-1, n_tok, dim)

    dec = NLADecoder(vocab_size=V, d_txt=d_txt, target_tokens=n_tok, target_dim=dim,
                     depth=3, frozen_embedding=E)

    # type-firewall: float ids must be rejected outright
    raised = False
    try:
        dec(torch.randn(2, 10))
    except AssertionError:
        raised = True
    check("FIREWALL #1: float ids rejected (ids-only assertion)", raised)

    # train the decoder briefly on the synthetic task
    Ntr = 2048
    train_ids = torch.randint(0, V, (Ntr, 16), generator=g)
    z_mean = z_of(train_ids).mean(0, keepdim=True)
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-3)
    for _step in range(1500):
        b = torch.randint(0, Ntr, (128,))
        ids = train_ids[b]
        # beta=1.0 here anchors magnitude: the cosine-dominated production default
        # (beta=0.1) leaves scale free, which is fine for the headline cosine metric
        # but undertrains the MSE-based R^2 we gate on in this synthetic check.
        loss, _, _ = recon_loss(dec(ids), z_of(ids), beta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()

    dec.eval()
    import torch.nn.functional as F
    def cos(a, b_):
        return F.cosine_similarity(a, b_, dim=-1).mean().item()
    with torch.no_grad():
        te = torch.randint(0, V, (512, 16), generator=g)
        z_te = z_of(te)
        zr = dec(te)
        r2_real, cos_real = fraction_variance_explained(zr, z_te, z_mean), cos(zr, z_te)
        sh = te[torch.randperm(len(te), generator=g)]
        r2_shuf, cos_shuf = fraction_variance_explained(dec(sh), z_te, z_mean), cos(dec(sh), z_te)
        empty = torch.zeros(len(te), 1, dtype=torch.long)
        r2_empty = fraction_variance_explained(dec(empty), z_te, z_mean)
    print(f"     R^2 over latent-mean floor:  real {r2_real:+.3f}   "
          f"shuffled {r2_shuf:+.3f}   empty {r2_empty:+.3f}")
    print(f"     cosine (headline metric):    real {cos_real:+.3f}   shuffled {cos_shuf:+.3f}")
    check("recon from REAL text beats the floor (R^2 > 0.3)", r2_real > 0.3)
    check("headline cosine high on real text (cos > 0.7)", cos_real > 0.7)
    check("SHUFFLED-text recon collapses to floor (R^2 < 0.05)", r2_shuf < 0.05)
    check("EMPTY-text recon collapses to floor (R^2 < 0.05)", r2_empty < 0.05)
    check("real >> shuffled (the latent-specific signal exists)", r2_real - r2_shuf > 0.3)

    # --- FIREWALL #2: recon loss must NOT reach the verbalizer (featurizer/resampler) ---
    grid = torch.randn(4, 11, 512, generator=g)
    channels = torch.randn(4, 11, 3, generator=g)
    with torch.no_grad():
        out = bridge.lm_forward(llm, torch.randint(0, V, (4, 16), generator=g),
                                grid, channels)[1]
        gen_ids = out.logits[:, 32:].argmax(-1)        # "generated" string (ids only)
    bridge.zero_grad(); dec.zero_grad()
    rloss, _, _ = recon_loss(dec(gen_ids), z_of(gen_ids))
    rloss.backward()
    leak = any(p.grad is not None and p.grad.abs().sum() > 0
               for p in list(bridge.featurizer.parameters()) + list(bridge.resampler.parameters()))
    check("FIREWALL #2: recon loss leaks NO gradient into featurizer/resampler", not leak)


def part_c_real_wm(args, g):
    print(f"\nC. REAL WORLD MODEL ({args.ckpt})")
    from src.bridge.wm_interface import N_PRED_CHANNELS, extract, load_world_model
    model, meta = load_world_model(args.ckpt, device="cpu")
    print(f"     loaded: window={meta['window']} d_model={meta['d_model']} "
          f"ppd={meta['ppd']} dist={meta['dist']}")
    x = torch.randn(2, meta["window"], meta["feature_dim"], generator=g)
    grid, z_pooled, channels = extract(model, x)
    check("grid shape [B,11,512]", tuple(grid.shape) == (2, 11, meta["d_model"]))
    check("z_pooled == grid.mean(dim=1)", torch.allclose(z_pooled, grid.mean(dim=1), atol=1e-5))
    check(f"channels shape [B,11,{N_PRED_CHANNELS}]", tuple(channels.shape) == (2, 11, N_PRED_CHANNELS))

    llm = TinyLLMStub(vocab_size=64, hidden_size=128)
    bridge = LanguageBridge(llm, latent_dim=meta["d_model"], n_pred_channels=N_PRED_CHANNELS,
                            n_soft_tokens=32)
    soft = bridge.soft_tokens(grid, channels)
    check("real grid -> soft tokens [B,32,d_llm]", tuple(soft.shape) == (2, 32, llm.config.hidden_size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="optional real WM checkpoint for part C")
    args = ap.parse_args()
    g = torch.Generator().manual_seed(1234)

    bridge, llm = part_a_forward(g)
    part_b_decoder(g, bridge, llm)
    if args.ckpt and Path(args.ckpt).exists():
        part_c_real_wm(args, g)
    elif args.ckpt:
        print(f"\nC. skipped — ckpt not found: {args.ckpt}")
    else:
        print("\nC. skipped — pass --ckpt outputs/wm_3map_dist_v3m/h8_mt/best.pt to test against the real WM")

    print()
    if _fail:
        print(f"\033[31mSMOKE FAILED\033[0m — {len(_fail)} check(s): {_fail}")
        sys.exit(1)
    print("\033[32mSMOKE PASSED\033[0m — scaffold wiring + NLA firewall verified")


if __name__ == "__main__":
    main()
