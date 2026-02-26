import Link from "next/link";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center px-6 text-center">
      <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
        CS2 Demo Viewer
      </h1>
      <p className="mt-4 max-w-xl text-lg text-muted">
        Interactive visualization for Counter-Strike 2 demo replays. Radar
        overlay with player positions, vision cones, kill lines, shot tracers,
        and timeline scrubbing.
      </p>
      <Link
        href="/viewer"
        className="mt-8 rounded-full bg-foreground px-8 py-3 text-sm font-medium text-background transition-opacity hover:opacity-80"
      >
        Open Viewer
      </Link>
      <p className="mt-12 text-xs text-muted">
        Add demo data to <code className="rounded bg-card px-1.5 py-0.5">public/viewer-data/</code> to get started.
        See <a href="https://github.com/skkwowee/cs2-demo-viewer#data-format" className="underline hover:text-foreground">data format docs</a>.
      </p>
    </main>
  );
}
