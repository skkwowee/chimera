"""
Unified data manifest for tracking screenshot provenance and metadata.

JSONL format — one JSON object per line, append-only during collection.

Required fields:
    id (str): Matches file stem (e.g. "abc123_0042" -> abc123_0042.png + abc123_0042.json)
    source (str): "youtube", "youtube_transcript", "demo", "manual"
"""

from __future__ import annotations

import json
from pathlib import Path


def append_to_manifest(manifest_path: Path | str, entry: dict) -> None:
    """Append one entry to the manifest JSONL file.

    Args:
        manifest_path: Path to the manifest.jsonl file
        entry: Dict with at least 'id' and 'source' keys
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_manifest(manifest_path: Path | str) -> dict[str, dict]:
    """Load manifest into {id: metadata} dict.

    Args:
        manifest_path: Path to the manifest.jsonl file

    Returns:
        Dict mapping id to full metadata entry
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {}

    entries = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries[entry["id"]] = entry

    return entries


def filter_manifest(manifest: dict[str, dict], **kwargs) -> dict[str, dict]:
    """Filter manifest entries by field values.

    Supports filtering by any field. For string fields, checks equality.
    For list fields (teams, tags, keywords), checks intersection.

    Args:
        manifest: Dict from load_manifest()
        **kwargs: Field filters, e.g. source="youtube", tags=["awp"]

    Returns:
        Filtered dict with same structure
    """
    result = {}
    for entry_id, entry in manifest.items():
        match = True
        for key, value in kwargs.items():
            entry_value = entry.get(key)
            if entry_value is None:
                match = False
                break

            if isinstance(entry_value, list) and isinstance(value, list):
                # List intersection — entry must contain at least one filter value
                if not set(value) & set(entry_value):
                    match = False
                    break
            elif isinstance(entry_value, list) and isinstance(value, str):
                # Single value against list field
                if value not in entry_value:
                    match = False
                    break
            else:
                if entry_value != value:
                    match = False
                    break

        if match:
            result[entry_id] = entry

    return result
