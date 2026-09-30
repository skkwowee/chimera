"""Versioned JSON/NPZ contracts. Loading never enables NumPy pickle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = "chimera-forecast-v1"


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def save_bundle(path, metadata, arrays):
    validate_bundle(metadata, arrays)
    meta = dict(metadata)
    meta["array_hashes"] = {k: array_hash(v) for k, v in arrays.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, metadata=np.array(json.dumps(meta, sort_keys=True, allow_nan=False)), **arrays)


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def load_bundle(path):
    with np.load(path, allow_pickle=False) as bundle:
        meta = json.loads(str(bundle["metadata"]))
        arrays = {k: bundle[k] for k in bundle.files if k != "metadata"}
    validate_bundle(meta, arrays)
    if meta.get("array_hashes") != {k: array_hash(v) for k, v in arrays.items()}:
        raise ValueError("bundle array integrity mismatch")
    return meta, arrays


def validate_manifest(manifest):
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if manifest.get("version") != VERSION or manifest.get("manifest_hash") != digest(body):
        raise ValueError("manifest version/hash mismatch")
    anchors = manifest["anchors"]
    if not anchors or len({a["id"] for a in anchors}) != len(anchors):
        raise ValueError("empty or duplicate anchors")
    spec = manifest["spec"]
    if spec["window"] < 5 or spec["horizon"] < 1 or spec["steps"] < 1 or spec["nominal_hz"] != 8:
        raise ValueError("unsupported forecast specification")
    if spec["mask"] != "alive_at_anchor_and_target" or spec["units"] != "game_xy":
        raise ValueError("unsupported mask/units")
    if spec.get("timing") != "nominal_frame_offsets_not_verified_source_ticks":
        raise ValueError("unsupported timing contract")
    if any(not a["match_id"] or a["id"] != digest({k: v for k, v in a.items() if k != "id"}) for a in anchors):
        raise ValueError("invalid anchor identity")


def validate_bundle(meta, arrays):
    validate_manifest(meta["manifest"])
    if meta.get("version") != VERSION:
        raise ValueError("unsupported bundle version")
    spec, anchors = meta["manifest"]["spec"], meta["manifest"]["anchors"]
    shape = (len(anchors), spec["steps"], 10, 2)
    if arrays["truth"].shape != shape or arrays["mask"].shape != shape[:-1] or arrays["mask"].dtype != bool:
        raise ValueError("truth/mask shape mismatch")
    if arrays["origin"].shape != (len(anchors), 10, 2):
        raise ValueError("origin shape mismatch")
    methods = meta["methods"]
    if not methods or set(arrays) != {"truth", "mask", "origin", *("pred_" + k for k in methods)}:
        raise ValueError("method/array mismatch")
    k = meta["samples"]
    if not isinstance(k, int) or k < 2:
        raise ValueError("bundle requires fixed K >= 2")
    for name, method in methods.items():
        x = arrays["pred_" + name]
        if x.shape != (shape[0], k, *shape[1:]) or method["sampling"] not in {"iid", "deterministic"}:
            raise ValueError("forecast shape/sampling mismatch")
        if method["sampling"] == "deterministic" and not np.array_equal(x, np.repeat(x[:, :1], k, axis=1)):
            raise ValueError("deterministic members disagree")
    if any(not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("nonfinite bundle arrays")


def require_compatible(left_meta, left, right_meta, right):
    """No implicit intersection or reordering across independently generated files."""
    for key in ("manifest", "samples", "seed", "temperature"):
        if left_meta[key] != right_meta[key]:
            raise ValueError(f"incompatible bundles: {key}")
    # Scoring can compare old implementations, but provenance remains in each method.
    for key in ("truth", "mask", "origin"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"incompatible bundles: {key}")
