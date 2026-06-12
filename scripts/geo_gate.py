#!/usr/bin/env python3
"""Geometry-gated decode: constrain the dist head's displacement choices to
moves that don't cross map geometry — grammar-constrained decoding, but the
grammar is the .tri collision mesh.

The model proposes (class probabilities); the mesh disposes: the chosen class
is ray-tested from the player's current position, and if the path crosses
geometry we fall back to the next-most-probable feasible class (stationary,
class 0, is always feasible). Off-engine wall enforcement at ~µs/ray.

Caveats (v1, documented not hidden):
- Ray runs FLAT at foot+28u: above step height/stairs, below most walls.
  Steep ramps can false-block (ray hits the rising surface); the fallback
  then picks a shorter ring in the same direction, which is conservative
  but safe. Doors/props are part of the mesh, so closed doors block.
- Tests the straight chord to the class center + offset; a legal curved
  path around a corner can be vetoed. Again conservative.
"""
from __future__ import annotations
from pathlib import Path
import torch

from awpy.visibility import VisibilityChecker

RAY_H = 28.0          # game units above feet: clears steps, hits walls
PLAYER_R = 16.0       # keep this margin short of the hit point
TRI_DIR = Path.home() / ".awpy" / "tris"


class GeoGate:
    """One map's collision mesh + the gated decode."""

    def __init__(self, map_name: str):
        self.map_name = map_name
        self.vc = VisibilityChecker(path=TRI_DIR / f"{map_name}.tri")
        self.checks = 0     # alive-player decode decisions
        self.vetoes = 0     # decisions where the TOP class was wall-infeasible

    def feasible(self, x, y, z, dx_u, dy_u) -> bool:
        """Is the straight move (dx_u, dy_u) game units from (x,y,z) clear?
        The ray is extended PLAYER_R past the target so the player's radius
        doesn't end up inside the wall."""
        mag2 = dx_u * dx_u + dy_u * dy_u
        if mag2 < 4.0:                               # stationary-ish: always ok
            return True
        s = 1.0 + PLAYER_R / mag2 ** 0.5             # overshoot by player radius
        a = (x, y, z + RAY_H)
        b = (x + dx_u * s, y + dy_u * s, z + RAY_H)
        return self.vc.is_visible(a, b)

    def stats(self) -> str:
        r = 100 * self.vetoes / self.checks if self.checks else 0.0
        return (f"{self.map_name}: veto rate {r:.1f}% "
                f"({self.vetoes}/{self.checks} top-class choices wall-blocked)")

    @torch.no_grad()
    def gated_residual(self, model, x_win, sample=False, temperature=1.0):
        """Like model.gen_residual(x_win) for the LAST frame, but each player's
        displacement class is the most probable (or sampled) FEASIBLE one.
        x_win: [1, L, F]. Returns residual [F] for the newest frame."""
        from train_world_model import N_PLAYERS, DIST_C, RAW_PPD  # local import, no cycle

        res = model.gen_residual(x_win, sample=sample, temperature=temperature)[0, -1].clone()
        out = model.heads(x_win)
        if "dist_logits" not in out:
            return res                                # no dist head: nothing to gate
        logits = out["dist_logits"][0, -1].float()    # [P, C]
        off = out["dist_off"][0, -1].float()          # [P, C, 2]
        ppd = model.ppd
        cur = x_win[0, -1]
        if sample:
            order = torch.multinomial(torch.softmax(logits / temperature, -1),
                                      DIST_C, replacement=False)        # [P, C] sampled order
        else:
            order = logits.argsort(dim=-1, descending=True)             # [P, C]
        centers = model.centers.to(logits.device)                       # [C, 2] normalized
        for p in range(N_PLAYERS):
            if cur[p * ppd + 13] <= 0.5:              # dead: leave as-is
                continue
            px, py = float(cur[p*ppd]) * 3000, float(cur[p*ppd+1]) * 3000
            pz = float(cur[p*ppd+2]) * 500
            self.checks += 1
            for ci, c in enumerate(order[p].tolist()):
                d = (centers[c] + off[p, c]) * 3000   # game units
                dx_u, dy_u = float(d[0]), float(d[1])
                if self.feasible(px, py, pz, dx_u, dy_u):
                    if ci > 0:
                        self.vetoes += 1              # top choice was wall-blocked
                    res[p*ppd + 0] = d[0] / 3000
                    res[p*ppd + 1] = d[1] / 3000
                    break
            # (class 0 = stationary always passes, so the loop always resolves)
        return res
