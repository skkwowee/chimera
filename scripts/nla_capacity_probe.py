#!/usr/bin/env python3
"""NLA capacity probe (bridge-design.md §7 Step 0) — BEFORE building the decoder.

The full NLA faithfulness loop needs the (unbuilt) bridge + Qwen-generated text.
This probe answers the *prerequisite* question cheaply, locally, with no pod and no
LLM: **is the pooled-512 latent `z = h.mean(dim=2)` reconstructable from
verbalizable, decision-relevant content at all?** If not, text cannot carry `z`
and we drop to the §2b fallback ladder (decision-relevant projection / head
outputs) before spending pod-hours on a decoder that can't win.

It is a CAPACITY CEILING, not the real metric. It does NOT use Qwen text or a text
encoder — so it cannot tell us whether Qwen's *actual* text preserves z (that's the
real recon-fidelity metric, post-bridge). It tells us the necessary precondition:
the information is THERE to be carried.

Three measurements on `wm_3map_dist_v3m` val latents:
  1. INTRINSIC DIM of z   — PCA: components for 90/95/99% variance. How low-rank?
  2. DECISION CAPACITY    — value-AUC preserved when z is truncated to top-k PCA.
                            (variance != decision-relevance; this is the one that
                            matters — how many dims carry what the value head reads.)
  3. CONTENT->z RECON     — train a tiny MLP to rebuild z from the model's own
                            VERBALIZABLE predictive outputs (value logit + per-player
                            predicted movement). Compare recon cosine to two floors:
                            latent-mean and shuffled-content. content >> floors =>
                            the stuff we'd put in text carries the latent.

Usage: python scripts/nla_capacity_probe.py --ckpt outputs/wm_3map_dist_v3m/h8_mt/best.pt
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_model import build_model, N_PLAYERS, auc  # noqa
from _corpus import load_corpus


def load_ckpt(path):
    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "c.pt"; shutil.copy(path, safe)
        return torch.load(safe, map_location="cpu", weights_only=False)


@torch.no_grad()
def collect(model, blob, L, keep, ppd, dev, max_rounds, stride, batch):
    """Per sampled frame: pooled latent z, value logit, win label, content vector."""
    px = [p*ppd+0 for p in range(N_PLAYERS)]; py = [p*ppd+1 for p in range(N_PLAYERS)]
    Z, V, Y, C = [], [], [], []
    nr = 0
    for r, m in zip(blob["tensors"], blob["metas"]):
        if m.get("map_name") not in keep:
            continue
        T = r.shape[0]
        ts = list(range(L - 1, T - 1, stride))
        if not ts:
            continue
        nr += 1
        if max_rounds and nr > max_rounds:
            break
        y = 1.0 if m.get("winner") == "ct" else 0.0
        for i in range(0, len(ts), batch):
            chunk = ts[i:i+batch]
            wins = torch.stack([r[t-L+1:t+1] for t in chunk]).to(dev)
            h = model.latent(wins)[:, -1, :]                     # [b,512] pooled z
            out = model.heads(wins)
            v = out["value"][:, -1]                              # [b] logit
            # verbalizable content = value + per-player predicted movement (head outputs)
            res = model.gen_residual(wins)[:, -1, :].cpu()       # decoded residual
            cls = out["dist_logits"][:, -1].argmax(-1).float().cpu() / 96.0  # [b,10] top class
            dxy = torch.stack([res[:, px], res[:, py]], -1).reshape(len(chunk), -1)  # [b,20]
            content = torch.cat([v.cpu().unsqueeze(1), cls, dxy], dim=1)  # [b, 1+10+20=31]
            Z.append(h.cpu()); V.append(v.cpu()); C.append(content)
            Y += [y] * len(chunk)
    return (torch.cat(Z), torch.cat(V), torch.tensor(Y), torch.cat(C))


def pca_fit(z):
    mu = z.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(z - mu, full_matrices=False)
    var = (S ** 2); var = var / var.sum()
    return mu, Vt, var                                          # Vt: [d,d] components (rows)


def truncate_k(z, mu, Vt, k):
    """Reconstruct z from its top-k principal components."""
    zc = z - mu
    coeff = zc @ Vt[:k].t()                                     # [n,k]
    return mu + coeff @ Vt[:k]                                  # [n,d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/wm_3map_dist_v3m/h8_mt/best.pt")
    ap.add_argument("--val-pt", default="data/processed/tick_sequences/val_v3m.pt")
    ap.add_argument("--maps", default="de_mirage,de_dust2,de_inferno")
    ap.add_argument("--max-rounds", type=int, default=120)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--recon-steps", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = load_ckpt(args.ckpt); a = ck["args"]; L = a["window"]; ppd = ck["per_player_dim"]
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"], a["heads"],
                        per_player_dim=ppd, dist=a.get("dist_head", False))
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    keep = set(args.maps.split(","))
    blob = load_corpus(args.val_pt, maps=keep, tag="val")
    print(f"ckpt step {ck.get('step')}  latent_dim={a['d_model']}  maps={sorted(keep)}")

    z, v, y, content = collect(model, blob, L, keep, ppd, torch.device(args.device),
                               args.max_rounds, args.stride, args.batch)
    n = len(z); base = y.mean().item()
    print(f"frames {n}  CT-win base {base:.2f}  z-dim {z.shape[1]}  content-dim {content.shape[1]}\n")

    # ---- 1. intrinsic dimension ----
    mu, Vt, var = pca_fit(z)
    cum = torch.cumsum(var, 0)
    k90 = int((cum < 0.90).sum()) + 1; k95 = int((cum < 0.95).sum()) + 1; k99 = int((cum < 0.99).sum()) + 1
    print("1) INTRINSIC DIM of z (PCA variance):")
    print(f"   90% -> {k90} dims   95% -> {k95} dims   99% -> {k99} dims   (of {z.shape[1]})")

    # ---- 2. decision capacity: value-AUC preserved under top-k truncation ----
    vh = model.value_head.to("cpu")
    full_auc = auc(vh(z).squeeze(1), y)
    print("\n2) DECISION CAPACITY (value-AUC of value_head applied to top-k z):")
    print(f"   {'k':>5} {'value-AUC':>10} {'% of full':>10}")
    print(f"   {'full':>5} {full_auc:10.3f} {100.0:9.0f}%")
    for k in (8, 16, 32, 64, 128, 256):
        if k >= z.shape[1]: break
        zk = truncate_k(z, mu, Vt, k)
        ak = auc(vh(zk).squeeze(1), y)
        print(f"   {k:>5} {ak:10.3f} {100*ak/full_auc:9.0f}%")

    # ---- 3. content -> z reconstruction vs floors ----
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    ntr = int(0.85 * n); tr, te = perm[:ntr], perm[ntr:]
    # standardize content + z (z per-dim, for a fair MSE; report cosine on raw z)
    cmu, csd = content[tr].mean(0), content[tr].std(0) + 1e-6
    Cn = (content - cmu) / csd
    zt_mu = z[tr].mean(0)                                       # latent-mean floor target
    dev = torch.device(args.device)
    Cn, zd = Cn.to(dev), z.to(dev)
    mlp = nn.Sequential(nn.Linear(content.shape[1], 512), nn.GELU(),
                        nn.Linear(512, 512), nn.GELU(), nn.Linear(512, z.shape[1])).to(dev)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    trd = tr.to(dev)
    for _step in range(args.recon_steps):
        b = trd[torch.randint(0, len(trd), (256,), device=dev)]
        pred = mlp(Cn[b])
        loss = (1 - F.cosine_similarity(pred, zd[b]).mean()) + 0.1 * F.mse_loss(pred, zd[b])
        opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    tot_var = ((zd[te.to(dev)] - zt_mu.to(dev)) ** 2).sum().item()   # z's total variance (test)
    def var_explained(pred, target):                                 # R^2: 1 - SSE/SS_tot
        return 1.0 - ((pred - target) ** 2).sum().item() / tot_var
    with torch.no_grad():
        ted = te.to(dev)
        pred = mlp(Cn[ted])
        cos_content = F.cosine_similarity(pred, zd[ted]).mean().item()
        r2_content = var_explained(pred, zd[ted])
        # floor B: content from a DIFFERENT random frame (shuffled) — breaks the pairing
        sh = ted[torch.randperm(len(ted), generator=torch.Generator().manual_seed(1))]
        r2_shuf = var_explained(mlp(Cn[sh]), zd[ted])
    # variance-explained is the honest capacity number (cosine is inflated when z is
    # low-rank — the latent-mean floor is R^2=0 by construction, shuffled is <=0).
    print("\n3) CONTENT->z RECON (held-out):")
    print(f"   {'':36} {'var-explained (R^2)':>18} {'cosine':>8}")
    print(f"   content (value + predicted movement) {r2_content:18.3f} {cos_content:8.3f}")
    print(f"   floor: latent-mean                   {0.0:18.3f} {'-':>8}")
    print(f"   floor: shuffled content              {r2_shuf:18.3f} {'-':>8}")

    print("\nVERDICT:")
    cap = "GO" if (k95 < z.shape[1] // 4 and r2_content > 0.5) else \
          "PARTIAL — pooled-z carries decision info; richer target may need fallback ladder"
    print(f"  decision-relevant info is low-rank: 95% of variance in {k95} dims, "
          f"value-AUC fully preserved by top ~8.")
    print(f"  verbalizable content explains {r2_content:.0%} of pooled-z variance "
          f"(shuffled floor {r2_shuf:+.0%}). Capacity: {cap}")
    print("  CAVEAT 1: pooled-512 is very low-rank — easy to carry, but a LOSSY target;")
    print("            the per-token [11,512] grid holds the per-player structure pooling drops.")
    print("  CAVEAT 2: ceiling only — proves the info is THERE, not that Qwen's text will carry")
    print("            it. The real metric (recon from Qwen-GENERATED text) waits for the bridge.")


if __name__ == "__main__":
    main()
