import Link from "next/link";

function Nav() {
  return (
    <nav className="fixed top-0 z-50 w-full border-b border-border bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-6">
        <Link href="/" className="font-mono text-lg font-bold tracking-tight">
          Chimera
        </Link>
        <div className="flex gap-6 text-sm text-muted">
          <a href="#abstract" className="transition-colors hover:text-foreground">
            Abstract
          </a>
          <a href="#method" className="transition-colors hover:text-foreground">
            Method
          </a>
          <a href="#results" className="transition-colors hover:text-foreground">
            Results
          </a>
          <a href="#demo" className="transition-colors hover:text-foreground">
            Demo
          </a>
          <a href="/viewer" className="transition-colors hover:text-foreground">
            Viewer
          </a>
          <a
            href="https://github.com/skkwowee/chimera"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-colors hover:text-foreground"
          >
            GitHub
          </a>
        </div>
      </div>
    </nav>
  );
}

function Hero() {
  return (
    <section className="flex min-h-[80vh] flex-col items-center justify-center px-6 pt-24 text-center">
      <div className="mb-4 inline-flex items-center rounded-full border border-border px-4 py-1.5 text-xs font-medium text-muted">
        NeurIPS 2026 (in preparation)
      </div>
      <h1 className="mx-auto max-w-4xl text-4xl font-bold leading-tight tracking-tight sm:text-5xl md:text-6xl">
        Think Before{" "}
        <span className="text-accent">You See</span>
      </h1>
      <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-muted sm:text-xl">
        VLMs as game agents without reinforcement learning from scratch
      </p>
      <p className="mt-4 text-sm text-muted">
        David Zeng
      </p>
      <div className="mt-8 flex gap-4">
        <a
          href="#abstract"
          className="rounded-full bg-foreground px-6 py-3 text-sm font-medium text-background transition-opacity hover:opacity-80"
        >
          Read Paper
        </a>
        <a
          href="https://github.com/skkwowee/chimera"
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-full border border-border px-6 py-3 text-sm font-medium transition-colors hover:bg-card-hover"
        >
          View Code
        </a>
      </div>
    </section>
  );
}

function Abstract() {
  return (
    <section id="abstract" className="mx-auto max-w-3xl px-6 py-24">
      <h2 className="text-sm font-semibold uppercase tracking-widest text-accent">
        Abstract
      </h2>
      <div className="mt-6 space-y-4 text-base leading-relaxed text-muted">
        <p>
          Frontier vision-language models (VLMs) exhibit a significant domain gap when
          applied to competitive gaming — failing at basic tasks like weapon identification
          and misinterpreting strategic context that human players read instantly. We
          hypothesize that this gap stems from a fundamental ordering problem: models are
          asked to visually ground strategic concepts they have never learned.
        </p>
        <p>
          We propose a <strong className="text-foreground">two-phase post-training</strong> paradigm
          where the model learns to think before it sees. First, supervised fine-tuning (SFT) on structured
          game replay data teaches the model CS2 domain knowledge — economy decisions,
          positional reasoning, round outcomes — through text alone. Second, Group Relative
          Policy Optimization (GRPO) refines visual grounding using five decomposed reward
          signals: format compliance, HUD accuracy, inferential accuracy, perception-reasoning
          consistency, and strategic reasoning quality.
        </p>
        <p>
          We evaluate on CS2 screenshot analysis, comparing against zero-shot baselines and
          vision-only SFT. Our results show that strategy pretraining reduces the labeled
          screenshots needed by up to 5x while producing more consistent strategic reasoning.
        </p>
      </div>
    </section>
  );
}

function Method() {
  return (
    <section id="method" className="border-t border-border py-24">
      <div className="mx-auto max-w-5xl px-6">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-accent">
          Method
        </h2>
        <h3 className="mt-4 text-3xl font-bold tracking-tight">
          Two-Phase Training Pipeline
        </h3>
        <p className="mt-4 max-w-2xl text-muted">
          The key insight is separating <em>what to reason about</em> (strategy) from{" "}
          <em>what to look at</em> (vision). Phase 1 teaches game knowledge cheaply from
          structured data. Phase 2 grounds that knowledge in visual perception.
        </p>

        {/* Pipeline diagram */}
        <div className="mt-12 grid gap-6 md:grid-cols-3">
          <PipelineCard
            phase="Phase 1"
            title="SFT — Domain Knowledge"
            description="Supervised fine-tuning on Claude-labeled data teaches the model output format (valid JSON with game_state, analysis, advice) and CS2 domain knowledge."
            details={[
              "LoRA on language + vision layers",
              "Merged 16-bit output for GRPO handoff",
              "~100 labeled screenshots",
            ]}
            color="purple"
          />
          <PipelineCard
            phase="Phase 2"
            title="GRPO — Quality Refinement"
            description="Group Relative Policy Optimization refines quality using five decomposed reward signals with independent advantage computation."
            details={[
              "Fresh LoRA on SFT-merged base",
              "16 generations per prompt",
              "Per-signal advantages (not collapsed)",
            ]}
            color="violet"
          />
          <PipelineCard
            phase="Evaluation"
            title="5-Signal Reward Decomposition"
            description="Each reward signal measures a distinct capability, enabling targeted analysis of where the model improves."
            details={[
              "Format gate (binary)",
              "HUD accuracy / Soft accuracy",
              "Consistency + Reasoning quality",
            ]}
            color="fuchsia"
          />
        </div>

        {/* Reward signals */}
        <div className="mt-16">
          <h3 className="text-xl font-bold tracking-tight">
            Reward Signal Design
          </h3>
          <p className="mt-2 text-muted">
            Instead of collapsing rewards into a single scalar, we pass five separate reward
            functions to GRPO. This lets the algorithm compute independent advantages per
            signal, avoiding reward hacking where format compliance dominates.
          </p>
          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {[
              { name: "Format Gate", weight: "5%", desc: "Valid JSON structure" },
              { name: "Hard Accuracy", weight: "40%", desc: "HUD-readable fields" },
              { name: "Soft Accuracy", weight: "15%", desc: "Inferential fields" },
              { name: "Consistency", weight: "25%", desc: "Perception → reasoning" },
              { name: "Reasoning", weight: "15%", desc: "Strategic quality" },
            ].map((signal) => (
              <div
                key={signal.name}
                className="rounded-lg border border-border bg-card p-4"
              >
                <div className="text-2xl font-bold text-accent">{signal.weight}</div>
                <div className="mt-1 text-sm font-semibold">{signal.name}</div>
                <div className="mt-1 text-xs text-muted">{signal.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function PipelineCard({
  phase,
  title,
  description,
  details,
  color,
}: {
  phase: string;
  title: string;
  description: string;
  details: string[];
  color: string;
}) {
  return (
    <div className="group rounded-xl border border-border bg-card p-6 transition-colors hover:bg-card-hover">
      <div className="text-xs font-semibold uppercase tracking-widest text-accent">
        {phase}
      </div>
      <h4 className="mt-3 text-lg font-bold">{title}</h4>
      <p className="mt-2 text-sm leading-relaxed text-muted">{description}</p>
      <ul className="mt-4 space-y-1.5">
        {details.map((detail) => (
          <li key={detail} className="flex items-start gap-2 text-xs text-muted">
            <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
            {detail}
          </li>
        ))}
      </ul>
    </div>
  );
}

function Results() {
  return (
    <section id="results" className="border-t border-border py-24">
      <div className="mx-auto max-w-5xl px-6">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-accent">
          Results
        </h2>
        <h3 className="mt-4 text-3xl font-bold tracking-tight">
          Experiment Comparison
        </h3>
        <p className="mt-4 max-w-2xl text-muted">
          We compare four model configurations to isolate the effect of strategy
          pretraining on visual grounding efficiency.
        </p>

        {/* Results table */}
        <div className="mt-8 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs font-semibold uppercase tracking-wider text-muted">
                <th className="pb-3 pr-6">Model</th>
                <th className="pb-3 pr-6">Training</th>
                <th className="pb-3 pr-6 text-right">Format</th>
                <th className="pb-3 pr-6 text-right">Hard Acc</th>
                <th className="pb-3 pr-6 text-right">Soft Acc</th>
                <th className="pb-3 pr-6 text-right">Consistency</th>
                <th className="pb-3 text-right">Weighted</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {[
                {
                  model: "A — Zero-shot",
                  training: "None",
                  format: "—",
                  hard: "—",
                  soft: "—",
                  consistency: "—",
                  weighted: "—",
                  highlight: false,
                },
                {
                  model: "B — Vision SFT",
                  training: "100 screenshots",
                  format: "—",
                  hard: "—",
                  soft: "—",
                  consistency: "—",
                  weighted: "—",
                  highlight: false,
                },
                {
                  model: "C — Strategy + Vision",
                  training: "Demos + 100 screenshots",
                  format: "—",
                  hard: "—",
                  soft: "—",
                  consistency: "—",
                  weighted: "—",
                  highlight: true,
                },
                {
                  model: "D — Strategy + Few-shot",
                  training: "Demos + 20 screenshots",
                  format: "—",
                  hard: "—",
                  soft: "—",
                  consistency: "—",
                  weighted: "—",
                  highlight: false,
                },
              ].map((row) => (
                <tr
                  key={row.model}
                  className={row.highlight ? "bg-accent/5" : ""}
                >
                  <td className="py-3 pr-6 font-medium">{row.model}</td>
                  <td className="py-3 pr-6 text-muted">{row.training}</td>
                  <td className="py-3 pr-6 text-right font-mono">{row.format}</td>
                  <td className="py-3 pr-6 text-right font-mono">{row.hard}</td>
                  <td className="py-3 pr-6 text-right font-mono">{row.soft}</td>
                  <td className="py-3 pr-6 text-right font-mono">{row.consistency}</td>
                  <td className="py-3 text-right font-mono font-semibold">
                    {row.weighted}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-xs text-muted">
          Results pending — experiments in progress. See{" "}
          <code className="rounded bg-card px-1.5 py-0.5 text-xs">experiment-state.json</code>{" "}
          for current status.
        </p>
      </div>
    </section>
  );
}

function Demo() {
  return (
    <section id="demo" className="border-t border-border py-24">
      <div className="mx-auto max-w-5xl px-6">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-accent">
          Demo
        </h2>
        <h3 className="mt-4 text-3xl font-bold tracking-tight">
          Try It Yourself
        </h3>
        <p className="mt-4 max-w-2xl text-muted">
          Upload a CS2 screenshot to see the model&apos;s analysis. Compare
          zero-shot vs. fine-tuned outputs side by side.
        </p>

        <div className="mt-8 grid gap-6 md:grid-cols-2">
          {/* Upload area */}
          <div className="flex min-h-[300px] flex-col items-center justify-center rounded-xl border-2 border-dashed border-border bg-card p-8 text-center transition-colors hover:border-accent/50">
            <div className="text-4xl">🎮</div>
            <p className="mt-4 text-sm font-medium">
              Drop a CS2 screenshot here
            </p>
            <p className="mt-1 text-xs text-muted">
              PNG, JPG, or WebP — 1920x1080 recommended
            </p>
            <p className="mt-6 text-xs text-muted">
              (Demo coming soon — requires inference backend)
            </p>
          </div>

          {/* Output preview */}
          <div className="rounded-xl border border-border bg-card p-6">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-accent" />
              <span className="text-xs font-semibold uppercase tracking-wider text-muted">
                Model Output
              </span>
            </div>
            <pre className="mt-4 overflow-auto rounded-lg bg-background p-4 text-xs leading-relaxed text-muted">
{`{
  "game_state": {
    "map_name": "de_dust2",
    "round_phase": "playing",
    "player_side": "CT",
    "player_health": 100,
    "player_armor": 100,
    "weapon_primary": "M4A1-S",
    "alive_teammates": 4,
    "alive_enemies": 5,
    "bomb_status": "carried"
  },
  "analysis": {
    "situation_summary": "...",
    "economy_assessment": "full-buy",
    "round_importance": "medium"
  },
  "advice": {
    "primary_action": "...",
    "reasoning": "...",
    "callout": "..."
  }
}`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}

function Architecture() {
  return (
    <section className="border-t border-border py-24">
      <div className="mx-auto max-w-5xl px-6">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-accent">
          Architecture
        </h2>
        <h3 className="mt-4 text-3xl font-bold tracking-tight">
          Training Pipeline
        </h3>

        {/* ASCII pipeline diagram */}
        <div className="mt-8 overflow-x-auto rounded-xl border border-border bg-card p-8">
          <pre className="font-mono text-xs leading-relaxed text-muted sm:text-sm">
{`  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  CS2 Demos  │     │ Screenshots │     │   Claude    │
  │  (replays)  │     │  (YouTube)  │     │  (labeler)  │
  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
         │                   │                    │
         ▼                   ▼                    ▼
  ┌─────────────┐     ┌────────────────────────────────┐
  │  Structured │     │      Labeled Screenshots       │
  │  Game State │     │   image + ground truth JSON     │
  └──────┬──────┘     └───────────────┬────────────────┘
         │                            │
         ▼                            ▼
  ┌─────────────────────────────────────────────────────┐
  │              Phase 1: SFT (Supervised)              │
  │  Qwen3-VL-8B + LoRA (vision + language layers)     │
  │  Loss: cross-entropy on ground truth JSON           │
  └──────────────────────┬──────────────────────────────┘
                         │  merged 16-bit model
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │               Phase 2: GRPO (RL)                    │
  │  Fresh LoRA on SFT base (language layers only)      │
  │  5 reward signals × 16 generations per prompt       │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │              Final Model (GGUF / 16-bit)            │
  │  Qwen3-VL-8B with CS2 domain expertise              │
  └─────────────────────────────────────────────────────┘`}
          </pre>
        </div>
      </div>
    </section>
  );
}

function Citation() {
  return (
    <section className="border-t border-border py-24">
      <div className="mx-auto max-w-3xl px-6">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-accent">
          Citation
        </h2>
        <pre className="mt-6 overflow-auto rounded-xl border border-border bg-card p-6 text-xs leading-relaxed text-muted">
{`@article{zeng2026chimera,
  title={Think Before You See: VLMs as Game Agents
         Without Reinforcement Learning from Scratch},
  author={Zeng, David},
  journal={arXiv preprint arXiv:26XX.XXXXX},
  year={2026}
}`}
        </pre>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t border-border py-8">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 text-xs text-muted">
        <span>Chimera — CS2 Gaming Copilot</span>
        <a
          href="https://github.com/skkwowee/chimera"
          target="_blank"
          rel="noopener noreferrer"
          className="transition-colors hover:text-foreground"
        >
          GitHub
        </a>
      </div>
    </footer>
  );
}

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Abstract />
        <Method />
        <Architecture />
        <Results />
        <Demo />
        <Citation />
      </main>
      <Footer />
    </>
  );
}
