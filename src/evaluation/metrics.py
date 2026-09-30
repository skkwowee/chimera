"""Forecast metrics in game units. Lower is better except binary AUC.

Inputs: samples [N,K,H,P,2], truth [N,H,P,2], observed mask [N,H,P].
The mask is fixed by the truth bundle, never the model. Survivor masking makes
these conditional position diagnostics, not scores of complete world states.
"""

from __future__ import annotations

import numpy as np


def binary_auc(scores, labels) -> float:
    """Mann–Whitney AUC with average ranks for ties; one class returns NaN."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).reshape(-1)
    if len(s) != len(y) or not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("AUC requires aligned finite scores and binary labels")
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if not npos or not nneg:
        return float("nan")
    order = np.argsort(s, kind="stable")
    _, start, count = np.unique(s[order], return_index=True, return_counts=True)
    ranks = np.empty(len(s))
    ranks[order] = np.repeat(start + (count + 1) / 2, count)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def forecast_scores(samples, truth, mask, *, sampling: str) -> dict[str, np.ndarray]:
    """One value per anchor, NaN for empty support; never mix path members.

    Joint energy norm = flattened masked Euclidean norm / sqrt(observed H*P).
    iid: fair MC estimator, pair term / (2K(K-1)); deterministic: exact point
    score, all members must agree. Ensemble sizes are fixed by the bundle.
    """
    x, y, m = np.asarray(samples, dtype=float), np.asarray(truth, dtype=float), np.asarray(mask)
    if y.ndim != 4 or y.shape[-1] != 2 or x.ndim != 5 or x.shape[0] != y.shape[0]:
        raise ValueError("expected samples [N,K,H,P,2] and truth [N,H,P,2]")
    if x.shape[2:] != y.shape[1:] or m.shape != y.shape[:-1] or m.dtype != bool:
        raise ValueError("unaligned forecast/truth or non-boolean mask")
    if min(x.shape) < 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("empty or nonfinite forecast/truth")
    k = x.shape[1]
    if sampling not in {"iid", "deterministic"}:
        raise ValueError("sampling must be iid or deterministic")
    if sampling == "iid" and k < 2:
        raise ValueError("fair energy score needs at least two iid members")
    if sampling == "deterministic" and not np.array_equal(x, np.repeat(x[:, :1], k, axis=1)):
        raise ValueError("deterministic forecast members differ")
    names = ("ade_mean", "minade_joint", "minade_player", "fde_mean", "energy_joint")
    result = {name: np.full(len(y), np.nan) for name in names}
    for i in range(len(y)):
        valid = m[i]
        count = int(valid.sum())
        if not count:
            continue
        error = np.linalg.norm(x[i] - y[i], axis=-1)  # [K,H,P]
        ade = error[:, valid].mean(axis=1)
        result["ade_mean"][i] = ade.mean()
        result["minade_joint"][i] = ade.min()
        player_counts = valid.sum(axis=0)
        selected = player_counts > 0
        player_ade = (error * valid).sum(axis=1)[:, selected] / player_counts[selected]
        result["minade_player"][i] = player_ade.min(axis=0).mean()
        if valid[-1].any():
            result["fde_mean"][i] = error[:, -1, valid[-1]].mean()
        # Mask and normalization depend only on truth and are identical for all methods.
        vectors = x[i][:, valid, :].reshape(k, -1) / np.sqrt(count)
        target = y[i][valid].reshape(-1) / np.sqrt(count)
        first = np.linalg.norm(vectors - target, axis=1).mean()
        pair_sum = sum(np.linalg.norm(vectors[j + 1 :] - vectors[j], axis=1).sum() for j in range(k - 1))
        result["energy_joint"][i] = first - (pair_sum / (k * (k - 1)) if sampling == "iid" else 0)
    return result


def paired_cluster_interval(candidate, baseline, match_ids, *, seed=0, replicates=2000) -> dict:
    """Exploratory paired percentile CI, pooled-anchor estimand, match clusters.

    Resample matches and keep every anchor/multiplicity within each match.
    This is NOT the BCa/percentile-t inference required by the locked gates.
    """
    a, b, ids = np.asarray(candidate), np.asarray(baseline), np.asarray(match_ids)
    if a.ndim != 1 or a.shape != b.shape or a.shape != ids.shape or replicates < 100:
        raise ValueError("unaligned paired values or fewer than 100 bootstrap replicates")
    keep = np.isfinite(a) & np.isfinite(b)
    delta, ids = (a - b)[keep], ids[keep]
    unique = np.unique(ids)
    result = {
        "delta": float(delta.mean()) if len(delta) else None,
        "anchors": len(delta),
        "matches": len(unique),
        "ci95": None,
        "method": "exploratory_paired_match_percentile",
        "estimand": "pooled_anchor_mean_candidate_minus_baseline",
        "seed": seed,
        "replicates": replicates,
        "warning": "Not a locked-gate confidence interval; few clusters may undercover.",
    }
    if len(unique) < 2:
        return result
    sums = np.array([delta[ids == mid].sum() for mid in unique])
    counts = np.array([(ids == mid).sum() for mid in unique])
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates)
    for r in range(replicates):
        draw = rng.integers(len(unique), size=len(unique))
        boot[r] = sums[draw].sum() / counts[draw].sum()
    result["ci95"] = np.quantile(boot, [0.025, 0.975]).tolist()
    return result
