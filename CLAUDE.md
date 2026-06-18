# Chimera

CS2 in-game tactical reasoning via a world-model pipeline: HLTV demos → tick state
→ **world model** (Phase 1, done) → **language bridge** (Phase 2, in progress) →
**GRPO** (Phase 3). Bet: learn the game by predicting it, then attach language.

**Env:** `.venv/bin/python`. Local GPU: RTX 4090 (WSL2, 15 GB RAM cap — keep bakes
`--workers 1`). Pods (RunPod) only for the 35B bridge/GRPO.

## Read first
- `docs/decisions-ledger.md` — rationale (what was killed/kept, eval traps).
- `docs/bridge-design.md` — Phase 2 architecture + NLA reconstruction objective.
- `docs/world-model-design.md` — Phase 1 model.
- `claude-progress.txt` — current run state.

## RunPod / GPU work
Before provisioning or driving any pod, read **`docs/pod-runbook.md`**. Key
guardrails: always `stop-pod` + verify `desiredStatus == EXITED` (cost); the
network volume `bp6ccofvnb` is **region-locked to AP-JP-1** (the GPU-stockout
cause — decouple via HuggingFace for region-free GPUs); match CUDA to torch before
training (the "14h-for-40-steps" lesson).
