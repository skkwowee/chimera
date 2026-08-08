# VERDICT: SAEs and the Chimera recipe

## 1. Is SAE "cap"?

Substantially yes for the claim as stated: for articulating what a model thinks or acting on known concepts, SAEs lose to trivial baselines — logistic regression/PCA at probing (Kantamneni et al., ICML 2025), prompting and diff-in-means at steering/detection (AxBench, ICML 2025) — and the standard SAE evals themselves pass on randomly initialized transformers (Heap et al. 2025) and randomized dictionaries (Korznikov et al. 2026), with dictionaries non-canonical (Leask et al., ICLR 2025), ~30% seed-stable (Paulo & Belrose, ICLR 2026), and absorption-distorted under hierarchy (Chanin et al., NeurIPS 2025). The critics did not prove SAEs useless: unsupervised discovery of *unknown* structure survives with real wins (Anthropic's auditing game, Marks et al. 2025; hypothesis generation, Movva et al., ICML 2025; board-game state F1 0.85–0.95, Karvonen et al., NeurIPS 2024), and Anthropic still ships SAE features as one audit instrument (Opus 4.6 System Card, Feb 2026). Field consensus by mid-2026: discovery flashlight, never certificate — GDM deprioritized SAEs on exactly these results (March 2025), and the Peng/Movva position paper (arXiv:2506.23845) codifies the split.

## 2. Chimera exposure (grep-verified role map)

| SAE-consuming site | Status |
|---|---|
| NLA faithfulness gate, keystone C1, recon ladder, GRPO gates | **Safe** — zero SAE dependence; behavioral gate is tier-2-backed-by-tier-1 evidence, the consensus-winning class |
| Amendment-D referee (ii) (fpp:159; runbook:183) | **Safe** — pre-registered as "SAE/linear-probe"; probe form already licensed |
| Weak-link #3 repair (b) (fpp:129) | **Safe** — written "SAE/probe audit" |
| Card 8 switch disjunct 2 (fpp:100, "SAE audit finds no head-inaccessible feature") | **Threatened** — SAE-only wording; a degenerate run on the ~6-effective-dim latent yields noise atoms that look "head-inaccessible," silently retaining soft tokens (judgment, from bridge-design §7 + Heap/Korznikov); no acceptance criterion exists |
| anthropic-lineage.md:32/:54 narrative | **Safe** — already scopes SAEs out; "fails to transfer" now citable (Heap, Korznikov, GDM), though board-game/Evo-2 results argue small sequence models are friendlier than the doc assumes |
| Paper claims | **Safe** — none rest on SAE features |

## 3. RECOMMENDATION: RESCOPE referee (ii) — do not strike, do not keep as-is

**Trigger status: already armed.** The pre-registered "SAE/linear-probe" wording (fpp:159, runbook:183) licenses running referee (ii) as a probe audit at $0 — no knob amendment, no forking-paths cost.

Cheapest change, consistent with locked-knobs discipline:

1. **Execute referee (ii) as the paired-probe audit**: byte-identical linear probes on [11,512] latent vs head outputs per Knob-7f target; "head-inaccessible feature exists" ⟺ latent probe beats head probe beyond CI. Add the head-Jacobian null-space probe riding A-R2.2's already-scheduled SVD artifact. (Probes dominate SAEs for exactly this question: Kantamneni, AxBench, MIB arXiv:2504.13151, Karvonen.)
2. **One written clarification** at fpp:100: the "SAE audit" disjunct resolves to this probe audit. This is interpretation of pre-registered text, not a recipe change (judgment).
3. **Any trained SAE → optional appendix only**, behind pre-registered controls: random-init 19M control (Heap), ≥2 seeds reporting seed-stable features only (Paulo & Belrose), Matryoshka/BatchTopK (Bussmann et al., ICML 2025), dead-fraction/FVU floor. Feature-absence never counts as capacity-absence (Leask, Chanin). Disclose the directional bias (degenerate SAE → spurious soft-token retention).
4. Add Heap/Korznikov/GDM citations to anthropic-lineage.md:32.

## 4. Sources

1. Kantamneni et al., "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing," ICML 2025 — https://arxiv.org/abs/2502.16681
2. Wu et al., "AxBench," ICML 2025 — https://arxiv.org/abs/2501.17148
3. Heap et al., "SAEs Can Interpret Randomly Initialized Transformers," 2025 — https://arxiv.org/abs/2501.17727
4. Korznikov et al., "Sanity Checks for SAEs," 2026 — https://arxiv.org/abs/2602.14111
5. Chanin et al., "A is for Absorption," NeurIPS 2025 — https://arxiv.org/abs/2409.14507
6. Leask et al., "SAEs Do Not Find Canonical Units," ICLR 2025 — https://arxiv.org/abs/2502.04878
7. Paulo & Belrose, seed instability, ICLR 2026 — https://arxiv.org/abs/2501.16615
8. GDM Mech Interp, "Negative Results… Deprioritising SAE Research," Mar 2025 — https://deepmindsafetyresearch.medium.com/negative-results-for-saes-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9
9. Peng, Movva et al., "Discover Unknown Concepts, Not Act on Knowns" — https://arxiv.org/abs/2506.23845
10. Karvonen et al., board-game dictionary learning, NeurIPS 2024 — https://arxiv.org/abs/2408.00113
11. Marks et al., "Auditing LMs for Hidden Objectives," 2025 — https://arxiv.org/abs/2503.10965
12. Anthropic, NLA, Transformer Circuits, May 2026 — https://transformer-circuits.pub/2026/nla/