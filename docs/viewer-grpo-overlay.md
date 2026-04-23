# GRPO overlay for the cs2-demo-viewer

This doc specifies the data format and UI hooks needed to overlay GRPO
training samples (with model advice and "neighbor" candidates) onto the
existing demo replay viewer at `cs2-demo-viewer`.

The goal: when watching a round, a sidebar shows what advice the model
generated for the spectated player at the current tick, and what other
ticks (in this round or other rounds) the system considered "tactically
neighboring." This lets a human verify whether our state-equivalence is
actually picking up the same kind of situation, which is the open
research question for the GRPO reward design.

## Data flow

```
extract_grpo_samples.py        →   smoke_test.jsonl
                                       (with `source` block from 2026-04-23 onward)

recover_source_metadata.py     →   smoke_test_source.jsonl
                                       (backfills source for legacy datasets)

train_grpo.py (GRPO training)  →   useful_jumps.jsonl
                                       (per-step audit: sample_idx, advices, rewards)

build_viewer_data.py           →   viewer_data.jsonl
                                       (per-sample record with advices + K neighbors
                                        under three equivalence metrics)

cs2-demo-viewer (Next.js)      →   reads viewer_data.jsonl,
                                       overlays on (demo, round, tick)
```

## viewer_data.jsonl record schema

One JSON object per line. Each object describes one GRPO training sample:

```jsonc
{
  "step_at_use": 47,                            // GRPO step where this was used
  "sample_idx": 1115,                           // Index into smoke_test.jsonl
  "source": {                                   // Where to find the replay frame
    "demo_stem": "furia_vitality_overpass",
    "round_num": 4,
    "tick": 16384,
    "player_name": "molodoy",
    "player_side": "t",
    "map_name": "de_overpass"
  },
  "query_state": {                              // The ground-truth game state
    "map_name": "de_overpass",
    "round_phase": "post-plant",
    "player_side": "T",
    "player_health": 73,
    "player_armor": 97,
    "weapon_primary": "AWP",
    "alive_teammates": 2,
    "alive_enemies": 2,
    "bomb_status": "planted"
    // ... full game_state
  },
  "pro_action": {                               // What the pro player actually did
    "behavior": {...},
    "categories": ["hold", "rotate"],
    "description": "held position; moved away from planted bomb"
  },
  "round_won": true,                            // Round outcome (binary)
  "advices": [                                  // G=4 model completions
    {
      "text_first200": "{\"game_state\":{\"round\":4,...",
      "reward": 0.415,                          // weighted total reward (incl. format gate)
      "passed_format": true
    },
    // ...3 more
  ],
  "neighbors": {                                // K nearest by each metric
    "tactical_19d": [                           // RECALL's 19-dim hand-engineered embedding
      {
        "sample_idx": 982,
        "distance": 0.041,                      // cosine distance (1 - cos_sim)
        "source": {                             // jump-to coordinates for this neighbor
          "demo_stem": "furia_vitality_inferno",
          "round_num": 12,
          "tick": 19872,
          "player_name": "ropz",
          "player_side": "ct",
          "map_name": "de_inferno"
        }
      }
      // ...K-1 more, sorted ascending by distance
    ],
    "sentence_emb": [...],                      // Sentence-transformer of game_state JSON
    "coarse_key":   [...]                       // Exact match on (side, alive_t, alive_e, bomb, phase)
  }
}
```

## Viewer UX expectations

### Sidebar component

When the viewer is on a tick that matches a `viewer_data.jsonl` record (by
`source.demo_stem + round_num + tick`), show a panel with:

1. **Model advices block** — for each of the G=4 completions, show:
   - The text (truncated at 200 chars; full text wasn't logged in the audit)
   - The reward score, color-coded (green > 0.3, yellow 0.1-0.3, red < 0.1)
   - A "format pass" checkmark
   - Mark the highest-scoring one as "best"

2. **Ground truth block** — show `pro_action.description` and the
   `round_won` outcome. This is what *actually* happened in this round.

3. **Neighbors block** — three tabs (one per metric: `tactical_19d`,
   `sentence_emb`, `coarse_key`). Each tab lists K=10 neighbors with:
   - Distance (small number = closer)
   - The neighbor's (demo_stem, round_num, player_name) as a clickable link
   - On click, the viewer should navigate to that demo+round and seek to
     the neighbor's tick

### Assessing equivalence

The user is essentially auditing our state-equivalence functions. Workflow:

1. Watch the current tick in the replay (see what's actually happening).
2. Look at the model's advice — does it make sense for this situation?
3. Click into a neighbor — does the neighbor really look like a similar
   tactical situation, or is it a "same-round different tick" hack
   (RECALL's failure mode)?
4. Toggle metrics. If `coarse_key` neighbors look more similar than
   `tactical_19d` neighbors, that's evidence our embedding is bad.

### Indicating cross-round vs same-round

Color-code each neighbor entry:
- **green border** if `neighbor.demo_stem == current.demo_stem AND
  neighbor.round_num != current.round_num` → cross-round neighbor (good
  generalization signal)
- **yellow border** if `neighbor.demo_stem != current.demo_stem`
  → cross-demo neighbor (strongest generalization signal)
- **red border** if `neighbor.demo_stem == current.demo_stem AND
  neighbor.round_num == current.round_num` → same-round neighbor
  (RECALL's failure mode; visually flag it)

Counting the red/yellow/green ratio over many viewed samples gives a
quick read on how well each metric is generalizing.

## Generation pipeline (server side)

After a training run completes:

```bash
# 1. Backfill source metadata for the dataset (one-time per dataset)
python scripts/recover_source_metadata.py \
    --dataset data/training/grpo/smoke_test.jsonl \
    --demos data/processed/demos \
    --output data/training/grpo/smoke_test_source.jsonl

# 2. Build the viewer data for a specific run
python scripts/build_viewer_data.py \
    --audit /workspace/outputs/grpo/f08v5/useful_jumps.jsonl \
    --source data/training/grpo/smoke_test_source.jsonl \
    --dataset data/training/grpo/smoke_test.jsonl \
    --output /workspace/outputs/grpo/f08v5/viewer_data.jsonl \
    --k 10

# 3. Push to the viewer's data directory (or HF dataset)
hf upload chimera-cs2 viewer_data.jsonl /workspace/outputs/grpo/f08v5/viewer_data.jsonl
```

The viewer can either fetch directly from HF or be pointed at a local
file via env var.

## Open follow-ups (not implemented)

- The audit only stores `completions_first200`. To inspect FULL advice
  text in the viewer, the trainer would need to log the full completion
  string. Trivial change but increases audit file size ~5-10x.
- `recover_source_metadata.py` will fail on samples where two demos on
  the same map have a player+round+HP collision. Acceptable today (we
  have 4 demos on 4 different maps); needs revisiting if we add more.
- The `tactical_19d` metric is included partly as a baseline to show
  what we *don't* want — its neighbors should look bad (mostly same-round
  ticks). If they don't look bad in the viewer, that's interesting too.
