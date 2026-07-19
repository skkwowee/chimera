#!/usr/bin/env python3
"""Push the patched v2.1 corpus blobs + manifests to HF (infra-plan.md §1 item 5).

Runs as runbook [1]'s final done-check line: one motion = offsite copy of the
single-copy asset + canonical pod input for all [6] runs. `corpus_manifest.json`
(written by patch_corpus.py per corpus-strategy.md §4) is the integrity source of
truth: every file's streamed sha256 must match the manifest record or the push is
REFUSED before any upload. Contract patch_corpus.py must write:

    "blobs": {"train_v2m_p1.pt": {"sha256": "<hex64>", "bytes": N}, ...}

split_manifest_v2.json may instead be covered by the split record
(`"split": {"split_manifest_sha256": "<hex64>"}`). --allow-no-manifest lets a
file with NO recorded entry through — meant for that json only, never blobs.
Blobs upload first, the manifest LAST, so an interrupted push never leaves a
manifest pointing at blobs that aren't there yet.

Keep sha256_file/expected_sha in sync with pull_blobs_hf.py (pods verify with it).

Usage: python scripts/push_blobs_hf.py --dry-run   # verify only, no network
       python scripts/push_blobs_hf.py             # verify then upload
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ID = "skkwowee/chimera-cs2"
DEST = "tick_sequences_v2.1"
BLOB_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "tick_sequences"
MANIFEST = "corpus_manifest.json"
BLOBS = ["train_v2m_p1.pt", "val_v2m_p1.pt", "train_v3m_p1.pt", "val_v3m_p1.pt"]
DEFAULT_FILES = [str(BLOB_DIR / n) for n in BLOBS + [MANIFEST, "split_manifest_v2.json"]]
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
    ap = argparse.ArgumentParser(description="verify against corpus_manifest.json, then upload to HF")
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument("--dest", default=DEST, help="folder in the dataset repo")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--allow-no-manifest", action="store_true",
                    help="permit a file with no manifest entry (split_manifest json only — never blobs)")
    ap.add_argument("--dry-run", action="store_true", help="sha256 + manifest verification only, no upload")
    args = ap.parse_args()

    files = [Path(f) for f in args.files]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit(f"REFUSED: missing local file(s): {', '.join(missing)}")

    man_path = next((f for f in files if f.name == MANIFEST), files[0].parent / MANIFEST)
    if not man_path.exists():
        sys.exit(f"REFUSED: {man_path} not found — corpus_manifest.json is the integrity "
                 "source of truth (written by patch_corpus.py); will not push unverifiable blobs")
    man = json.loads(man_path.read_text())

    rows, problems = [], []
    for f in files:
        sha = sha256_file(f)
        rows.append((f, sha))
        if f.name == MANIFEST:
            print(f"  {f.name}: {sha[:16]}… (the manifest itself — not self-checked)")
            continue
        exp, exp_bytes = expected_sha(man, f.name)
        if exp is None:
            if args.allow_no_manifest:
                print(f"  {f.name}: {sha[:16]}… UNTRACKED (no manifest entry; --allow-no-manifest)")
            else:
                problems.append(f"{f.name}: no sha256 recorded in {man_path.name}")
        elif exp != sha:
            problems.append(f"{f.name}: sha256 MISMATCH local={sha} manifest={exp}")
        elif exp_bytes is not None and exp_bytes != f.stat().st_size:
            problems.append(f"{f.name}: size mismatch local={f.stat().st_size} manifest={exp_bytes}")
        else:
            print(f"  {f.name}: {sha[:16]}… OK (matches manifest)")

    if problems:
        print("\nREFUSED — nothing uploaded:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    dest = args.dest.rstrip("/")
    if args.dry_run:
        print("\n[dry-run] verification passed; skipping upload")
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        for f, _ in sorted(rows, key=lambda r: r[0].name == MANIFEST):  # manifest LAST
            rel = f"{dest}/{f.name}"
            print(f"  uploading {f.name} ({f.stat().st_size / 1e9:.2f} GB) -> {rel} ...")
            api.upload_file(path_or_fileobj=str(f), path_in_repo=rel,
                            repo_id=args.repo, repo_type="dataset")

    print(f"\n{'file':<26} {'sha256':<64} url")
    for f, sha in rows:
        print(f"{f.name:<26} {sha} https://huggingface.co/datasets/{args.repo}/resolve/main/{dest}/{f.name}")


if __name__ == "__main__":
    main()
