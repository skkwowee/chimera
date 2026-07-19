#!/usr/bin/env python3
"""Ship the bridge SFT cache to HuggingFace (bridge-design.md §4; pod-runbook.md).

The Phase-2 SFT pairs are precomputed locally on the 4090, then trained on the pod.
Staging them through HF (a) decouples the pod from the AP-JP-1 network volume — so
training can run in ANY region with GPU stock, the fix for the JP stockout pain —
and (b) makes the SFT dataset reproducible.

Uploads to the existing dataset repo (skkwowee/chimera-cs2) under bridge_sft/. On
the pod, pull with:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("skkwowee/chimera-cs2", "bridge_sft/train_single.pt",
                        repo_type="dataset")

Usage: python scripts/push_sft_hf.py --sft data/processed/bridge_sft/train_single.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "skkwowee/chimera-cs2"  # matches scripts/data.py / merge convention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="data/processed/bridge_sft/train_single.pt")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--prefix", default="bridge_sft", help="path-in-repo folder")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, create_repo
    sft = Path(args.sft)
    assert sft.exists(), f"missing {sft} — run scripts/gen_bridge_sft.py first"
    samples = sft.with_suffix(".samples.jsonl")

    uploads = [(sft, f"{args.prefix}/{sft.name}")]
    if samples.exists():
        uploads.append((samples, f"{args.prefix}/{samples.name}"))
    sz = sum(p.stat().st_size for p, _ in uploads) / 1e6
    print(f"repo={args.repo} (dataset)  files={[r for _, r in uploads]}  total={sz:.1f} MB")
    if args.dry_run:
        print("[dry-run] not uploading"); return

    create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api = HfApi()
    for path, rel in uploads:
        print(f"  uploading {path.name} -> {rel} ...")
        api.upload_file(path_or_fileobj=str(path), path_in_repo=rel,
                        repo_id=args.repo, repo_type="dataset")
    print(f"\ndone. pull on the pod with hf_hub_download('{args.repo}', "
          f"'{args.prefix}/{sft.name}', repo_type='dataset')")


if __name__ == "__main__":
    main()
