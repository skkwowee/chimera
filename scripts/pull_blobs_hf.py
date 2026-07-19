#!/usr/bin/env python3
"""Pull + sha256-verify the v2.1 corpus blobs from HF (infra-plan.md §1 item 5).

Pod-side counterpart to push_blobs_hf.py: downloads `corpus_manifest.json` FIRST,
then each blob, and verifies every file's streamed sha256 against the manifest
record. EXITS NONZERO on any mismatch or missing record — a pod must refuse to
train on a stale or truncated blob. --dry-run skips the network entirely and
verifies files already present in --dest-dir (same check path).

Keep sha256_file/expected_sha in sync with push_blobs_hf.py (defines the
manifest "blobs"/{sha256,bytes} contract written by patch_corpus.py).

Usage (pod): python scripts/pull_blobs_hf.py --dest-dir /workspace/tick_sequences
Subset:      ... --files train_v2m_p1.pt val_v2m_p1.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ID = "skkwowee/chimera-cs2"
SRC = "tick_sequences_v2.1"
MANIFEST = "corpus_manifest.json"
BLOBS = ["train_v2m_p1.pt", "val_v2m_p1.pt", "train_v3m_p1.pt", "val_v3m_p1.pt"]
DEFAULT_FILES = BLOBS + ["split_manifest_v2.json"]
CHUNK = 8 << 20  # 8 MB — blobs run up to ~10 GB, never load whole


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def expected_sha(man: dict, name: str) -> tuple[str | None, int | None]:
    """Recorded (sha256, bytes) for `name`, (None, None) if the manifest has no entry."""
    e = (man.get("blobs") or {}).get(name)
    if isinstance(e, str):
        return e, None
    if isinstance(e, dict):
        return e.get("sha256"), e.get("bytes")
    if name == "split_manifest_v2.json":  # split record fallback (spec §4 "split_manifest hash")
        sr = man.get("split") or man.get("split_record") or {}
        for k in ("split_manifest_sha256", "split_manifest_hash", "sha256"):
            v = sr.get(k)
            if isinstance(v, str) and len(v) == 64:
                return v, None
    return None, None


def main():
    ap = argparse.ArgumentParser(description="download blobs from HF and verify against corpus_manifest.json")
    ap.add_argument("--dest-dir", required=True)
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                    help=f"basenames under {SRC}/ (the manifest is always pulled first)")
    ap.add_argument("--src", default=SRC, help="folder in the dataset repo")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--allow-no-manifest", action="store_true",
                    help="tolerate a file with no manifest entry (marked UNVERIFIED; never use for blobs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no network: verify files already present in --dest-dir")
    args = ap.parse_args()

    dest = Path(args.dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    src = args.src.rstrip("/")

    def fetch(name: str) -> Path:
        tgt = dest / name
        if args.dry_run:
            if not tgt.exists():
                sys.exit(f"[dry-run] {tgt} not present locally — nothing to verify")
            return tgt
        from huggingface_hub import hf_hub_download
        got = Path(hf_hub_download(args.repo, f"{src}/{name}", repo_type="dataset",
                                   local_dir=str(dest)))
        if got != tgt:
            got.replace(tgt)  # flatten the <src>/ subfolder local_dir mirrors
        return tgt

    man_path = fetch(MANIFEST)
    man = json.loads(man_path.read_text())
    rows = [(MANIFEST, sha256_file(man_path), "manifest (source of truth)")]

    failures = []
    for name in [n for n in args.files if n != MANIFEST]:
        p = fetch(name)
        sha = sha256_file(p)
        exp, exp_bytes = expected_sha(man, name)
        if exp is None:
            if args.allow_no_manifest:
                status = "UNVERIFIED (no manifest entry; --allow-no-manifest)"
            else:
                status = "NO MANIFEST ENTRY"
                failures.append(f"{name}: no sha256 recorded in {MANIFEST}")
        elif exp != sha:
            status = "MISMATCH"
            failures.append(f"{name}: sha256 MISMATCH got={sha} manifest={exp}")
        elif exp_bytes is not None and exp_bytes != p.stat().st_size:
            status = "SIZE MISMATCH"
            failures.append(f"{name}: size mismatch got={p.stat().st_size} manifest={exp_bytes}")
        else:
            status = "OK"
        print(f"  {name}: {sha[:16]}… {status}")
        rows.append((name, sha, status))

    if failures:
        bang = "!" * 76
        print(f"\n{bang}\n!! BLOB INTEGRITY FAILURE — stale/truncated download. "
              f"DO NOT TRAIN ON THESE FILES.", file=sys.stderr)
        for f in failures:
            print(f"!! {f}", file=sys.stderr)
        print(bang, file=sys.stderr)
        sys.exit(1)

    print(f"\nall verified -> {dest}")
    for name, sha, status in rows:
        print(f"  {name:<26} {sha}  {status}")


if __name__ == "__main__":
    main()
