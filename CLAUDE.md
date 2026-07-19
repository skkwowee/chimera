# Chimera

CS2 in-game tactical reasoning via a world-model pipeline: HLTV demos → tick state
→ **world model** (canonical retrain PENDING — old checkpoints are history) →
**language bridge** → **grounded GRPO**. Bet: learn the game by predicting it,
then attach language. Post-2026-07-18 reset: every stage is gated + pre-registered.

**Env:** `.venv/bin/python`. Local GPU: RTX 4090. WSL2 RAM: 15 GB until the next
`wsl --shutdown`, then 22 GB (+8 GB swap, `.wslconfig` written 2026-07-18); keep
bakes `--workers 1` regardless. Pods (RunPod) only for the 35B bridge/GRPO.

## Read first (in this order)
- `claude-progress.txt` **top section** — the ORDERED RUNBOOK [1]–[7] with
  done-checks; determines what to do next in any session.
- `docs/retrain-recipe.md` (+ `retrain-recipe-knobs4-7.md`) — Knobs 1–7 LOCKED,
  pre-registered. Do not unlock without evidence hitting a written switch trigger.
- `docs/first-principles-plan.md` — WHY each stage exists (5-whys causal chain,
  weak-links register, amendments A–G).
- `docs/adversarial-review.md` — confirmed defects + punch list (criticals fixed).
- `docs/datasheet.md` — corpus certification + defect registry (incl. broken
  bomb_site label: derive site from plant position).
- `docs/decisions-ledger.md` — rationale (killed/kept, eval traps).
- `docs/bridge-design.md` — Phase 2 + NLA gate (GROUNDED GRPO reward per §5).

## RunPod / GPU work
Before provisioning or driving any pod, read **`docs/pod-runbook.md`**. Key
guardrails: always `stop-pod` + verify `desiredStatus == EXITED` (cost); the
network volume `bp6ccofvnb` is **region-locked to AP-JP-1** (the GPU-stockout
cause — decouple via HuggingFace for region-free GPUs); match CUDA to torch before
training (the "14h-for-40-steps" lesson).
