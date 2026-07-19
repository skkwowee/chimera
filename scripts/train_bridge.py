#!/usr/bin/env python3
"""Phase-2 bridge SFT trainer (bridge-design.md §2/§7 milestone 1).

Trains the language bridge on the cached `(grid, channels, prompt, target)` pairs
from gen_bridge_sft.py: soft tokens (featurizer -> resampler) are PREFIXED to the
prompt+target embeddings and the LLM is teacher-forced on the TARGET span only
(prompt + soft positions are masked from the loss).

Trainable: featurizer + resampler (+ LoRA on the LLM in the pod path). The world
model is already frozen upstream (latents are precomputed in the cache); the LLM
base is frozen (4-bit) with LoRA adapters on the pod.

Two backends behind one interface so the whole loop is CPU-smokeable with ZERO pod
hours and the real model swaps in unchanged:
  --llm stub  (default): byte tokenizer + TinyLLMStub. Local, no deps.
  --llm qwen  : AutoTokenizer + 4-bit AutoModelForCausalLM + PEFT LoRA (pod only).

Also implements the §3 ablate-the-latent hook: same forward with soft tokens zeroed,
to measure latent-on vs latent-off loss.

Usage:
  python scripts/train_bridge.py --smoke                       # local CPU wiring test
  python scripts/train_bridge.py --llm qwen --model <path> ... # on the pod
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bridge import LanguageBridge, TinyLLMStub

SEP = "\n"


# ------------------------------------------------------------------- tokenizer(s)
class ByteTokenizer:
    """Trivial byte-level tokenizer for the local smoke (no HF deps). Reserved ids
    256=BOS, 257=EOS, 258=PAD above the 256 byte values."""
    BOS, EOS, PAD = 256, 257, 258
    vocab_size = 259

    def encode(self, s):
        return list(s.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", "ignore")


# ----------------------------------------------------------------------- data
class SFTPairs(Dataset):
    def __init__(self, cache):
        self.grid = cache["grid"]; self.channels = cache["channels"]
        self.prompt = cache["prompt"]; self.target = cache["target"]
        self.won = cache["won"]; self.value_logit = cache["value_logit"]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        return (self.grid[i].float(), self.channels[i].float(),
                self.prompt[i], self.target[i], self.won[i], self.value_logit[i])


def make_collate(tok, max_len=192):
    """Tokenize prompt+target; label-mask the prompt span so loss is on the target
    (answer) only. Returns padded tensors + stacked latent inputs."""
    bos, eos, pad = tok.BOS, tok.EOS, tok.PAD

    def collate(batch):
        grids, chans, ids_list, lab_list = [], [], [], []
        for grid, ch, prompt, target, *_ in batch:
            p_ids = [bos] + tok.encode(prompt + SEP)
            t_ids = tok.encode(target) + [eos]
            ids = (p_ids + t_ids)[:max_len]
            labels = ([-100] * len(p_ids) + t_ids)[:max_len]  # mask prompt + BOS
            ids_list.append(ids); lab_list.append(labels)
            grids.append(grid); chans.append(ch)
        T = max(len(x) for x in ids_list)
        input_ids = torch.full((len(batch), T), pad, dtype=torch.long)
        labels = torch.full((len(batch), T), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), T), dtype=torch.long)
        for j, (ids, lab) in enumerate(zip(ids_list, lab_list)):
            input_ids[j, :len(ids)] = torch.tensor(ids)
            labels[j, :len(lab)] = torch.tensor(lab)
            attn[j, :len(ids)] = 1
        return {"grid": torch.stack(grids), "channels": torch.stack(chans),
                "input_ids": input_ids, "labels": labels, "attn": attn}
    return collate


# ----------------------------------------------------------------------- step
def bridge_step(bridge, llm, batch, dev, ablate=False):
    """Soft-prefix forward; teacher-forced CE on the (label-masked) target span."""
    grid = batch["grid"].to(dev); channels = batch["channels"].to(dev)
    input_ids = batch["input_ids"].to(dev); labels = batch["labels"].to(dev)
    attn = batch["attn"].to(dev)
    soft = bridge.soft_tokens(grid, channels, ablate=ablate)        # [B,M,d_llm]
    M = soft.shape[1]
    text_emb = llm.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([soft, text_emb], dim=1)
    full_attn = torch.cat([torch.ones(soft.shape[:2], dtype=torch.long, device=dev), attn], dim=1)
    full_labels = torch.cat([torch.full((soft.shape[0], M), -100, device=dev), labels], dim=1)
    out = llm(inputs_embeds=inputs_embeds, attention_mask=full_attn)
    logits = out.logits[:, :-1]
    tgt = full_labels[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)


# -------------------------------------------------------------------- backends
def build_stub_backend(cache):
    tok = ByteTokenizer()
    llm = TinyLLMStub(vocab_size=tok.vocab_size, hidden_size=128, layers=2)
    for p in llm.parameters():
        p.requires_grad_(False)                    # frozen base (mirrors the pod)
    return tok, llm


def build_qwen_backend(model_path, lora_r=16, lora_alpha=32):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.BOS = tok.bos_token_id or tok.eos_token_id
    tok.EOS = tok.eos_token_id; tok.PAD = tok.pad_token_id
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    llm = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb, device_map="auto")
    lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    llm = get_peft_model(llm, lora)
    return tok, llm


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="data/processed/bridge_sft/train_single.pt")
    ap.add_argument("--llm", choices=["stub", "qwen"], default="stub")
    ap.add_argument("--model", default=None, help="HF path/id for --llm qwen")
    ap.add_argument("--soft-tokens", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs/bridge/single_v1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.llm = "stub"; args.batch = 8; args.steps = 150; args.device = "cpu"
        print("[smoke] local CPU wiring test on a tiny subset")

    cache = torch.load(args.sft, map_location="cpu", weights_only=False)
    sch = cache["schema"]
    if args.smoke:                                  # tiny overfit subset
        for k in ("grid", "channels", "value_logit", "won"):
            cache[k] = cache[k][:32]
        cache["prompt"] = cache["prompt"][:32]; cache["target"] = cache["target"][:32]

    if args.llm == "stub":
        tok, llm = build_stub_backend(cache)
    else:
        assert args.model, "--llm qwen needs --model"
        tok, llm = build_qwen_backend(args.model)
    dev = torch.device(args.device)
    llm.to(dev) if args.llm == "stub" else None     # qwen uses device_map

    bridge = LanguageBridge(llm, latent_dim=sch["latent_dim"], n_pred_channels=sch["n_pred_channels"],
                            n_soft_tokens=args.soft_tokens).to(dev)
    ds = SFTPairs(cache)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=make_collate(tok),
                    drop_last=True)
    trainable = [p for p in bridge.parameters() if p.requires_grad] + \
                [p for p in llm.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"backend={args.llm}  d_llm={llm.config.hidden_size}  trainable params={n_train/1e6:.2f}M")
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    it = iter(dl); losses = []
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        loss = bridge_step(bridge, llm, batch, dev)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    # ablate-the-latent sanity (§3): latent-on vs latent-off loss on a fresh batch
    bridge.eval()
    with torch.no_grad():
        b = next(iter(DataLoader(ds, batch_size=min(16, len(ds)), shuffle=False, collate_fn=make_collate(tok))))
        on = bridge_step(bridge, llm, b, dev, ablate=False).item()
        off = bridge_step(bridge, llm, b, dev, ablate=True).item()
    print(f"\nablate-the-latent: loss latent-on {on:.4f}  latent-off {off:.4f}  delta {off - on:+.4f}")

    if args.smoke:
        ok = True
        if not torch.isfinite(torch.tensor(losses[-1])):
            print("  FAIL: non-finite loss"); ok = False
        if losses[-1] > losses[0]:
            print(f"  FAIL: loss did not decrease ({losses[0]:.3f} -> {losses[-1]:.3f})"); ok = False
        gradok = any(p.grad is not None for p in bridge.featurizer.parameters())
        if not gradok:
            print("  FAIL: no gradient reached the bridge"); ok = False
        print("\033[32mSMOKE PASSED\033[0m" if ok else "\033[31mSMOKE FAILED\033[0m")
        sys.exit(0 if ok else 1)

    outp = Path(args.out); outp.mkdir(parents=True, exist_ok=True)
    torch.save({"featurizer": bridge.featurizer.state_dict(),
                "resampler": bridge.resampler.state_dict(),
                "args": vars(args), "schema": sch}, outp / "bridge.pt")
    if args.llm == "qwen":
        llm.save_pretrained(outp / "lora")
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
