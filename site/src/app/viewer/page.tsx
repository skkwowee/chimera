"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import DemoSelector from "@/components/DemoSelector";
import MapCanvas, { type MapInfo, type PlayerFrame } from "@/components/MapCanvas";
import TimelineSlider, { type KillEvent } from "@/components/TimelineSlider";

interface RoundData {
  round_num: number;
  start_tick: number;
  end_tick: number;
  freeze_end: number;
  winner: string;
  reason: string;
  bomb_plant_tick: number | null;
  bomb_site: string | null;
  frames: { tick: number; players: PlayerFrame[] }[];
}

interface MetaData {
  stem: string;
  map_name: string;
  header: Record<string, string>;
  rounds: { round_num: number; winner: string; reason: string }[];
  kills: KillEvent[];
  bomb: unknown[];
}

export default function ViewerPage() {
  const [selectedDemo, setSelectedDemo] = useState<string | null>(null);
  const [selectedRound, setSelectedRound] = useState(1);
  const [meta, setMeta] = useState<MetaData | null>(null);
  const [roundData, setRoundData] = useState<RoundData | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [mapData, setMapData] = useState<Record<string, MapInfo> | null>(null);

  // Load map coordinate data once
  useEffect(() => {
    fetch("/maps/map-data.json")
      .then((r) => r.json())
      .then(setMapData)
      .catch(() => setMapData(null));
  }, []);

  // Load demo metadata when demo changes
  useEffect(() => {
    if (!selectedDemo) return;
    setMeta(null);
    fetch(`/viewer-data/${selectedDemo}/meta.json`)
      .then((r) => r.json())
      .then(setMeta)
      .catch(() => setMeta(null));
  }, [selectedDemo]);

  // Load round data when round changes
  useEffect(() => {
    if (!selectedDemo) return;
    setRoundData(null);
    setFrameIndex(0);
    const padded = String(selectedRound).padStart(2, "0");
    fetch(`/viewer-data/${selectedDemo}/round_${padded}.json`)
      .then((r) => r.json())
      .then(setRoundData)
      .catch(() => setRoundData(null));
  }, [selectedDemo, selectedRound]);

  const handleSelect = useCallback((demo: string, round: number) => {
    setSelectedDemo(demo);
    setSelectedRound(round);
  }, []);

  const handleFrameChange = useCallback((idx: number) => {
    setFrameIndex(idx);
  }, []);

  const currentFrame = roundData?.frames[frameIndex];
  const players = currentFrame?.players ?? [];
  const mapName = meta?.map_name ?? "de_mirage";

  // Filter kills for current round
  const roundKills: KillEvent[] = (meta?.kills ?? []).filter(
    (k: KillEvent & { round_num?: number }) =>
      k.round_num === selectedRound
  );

  // Round info
  const roundInfo = meta?.rounds?.find(
    (r: { round_num: number }) => r.round_num === selectedRound
  );

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border bg-background/80 px-6 py-3 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="font-mono text-lg font-bold tracking-tight"
            >
              Chimera
            </Link>
            <span className="text-xs text-muted">/</span>
            <span className="text-sm font-medium">Demo Viewer</span>
          </div>
          <DemoSelector
            onSelect={handleSelect}
            selectedDemo={selectedDemo}
            selectedRound={selectedRound}
          />
        </div>
      </header>

      {/* Main content */}
      <main className="mx-auto flex w-full max-w-7xl flex-1 gap-6 px-6 py-6">
        {/* Map canvas */}
        <div className="flex-1">
          <MapCanvas
            mapName={mapName}
            players={players}
            mapData={mapData}
          />
        </div>

        {/* Side panel */}
        <div className="w-72 shrink-0 space-y-4">
          {/* Round info */}
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
              Round {selectedRound}
            </h3>
            {roundInfo && (
              <div className="mt-2 space-y-1 text-sm">
                <div>
                  Winner:{" "}
                  <span
                    className={
                      roundInfo.winner === "ct"
                        ? "font-semibold text-blue-400"
                        : "font-semibold text-amber-400"
                    }
                  >
                    {roundInfo.winner.toUpperCase()}
                  </span>
                </div>
                <div className="text-muted">Reason: {roundInfo.reason}</div>
              </div>
            )}
            {roundData && (
              <div className="mt-2 text-xs text-muted">
                {roundData.bomb_site && (
                  <div>Bomb site: {roundData.bomb_site}</div>
                )}
                <div>
                  Ticks: {roundData.start_tick} - {roundData.end_tick}
                </div>
              </div>
            )}
            {currentFrame && (
              <div className="mt-1 text-xs font-mono text-muted">
                Tick: {currentFrame.tick}
              </div>
            )}
          </div>

          {/* Player list */}
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
              Players
            </h3>
            <div className="mt-2 space-y-0.5">
              {players
                .sort((a, b) => {
                  if (a.side !== b.side) return a.side === "CT" ? -1 : 1;
                  return a.name.localeCompare(b.name);
                })
                .map((p) => (
                  <div
                    key={p.name}
                    className={`flex items-center justify-between text-xs ${
                      p.alive ? "" : "text-muted line-through"
                    }`}
                  >
                    <span>
                      <span
                        className={
                          p.side === "CT" ? "text-blue-400" : "text-amber-400"
                        }
                      >
                        {p.side}
                      </span>{" "}
                      {p.name}
                    </span>
                    <span className="font-mono">{p.hp}</span>
                  </div>
                ))}
            </div>
          </div>

          {/* Kill feed for this round */}
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
              Kills
            </h3>
            <div className="mt-2 max-h-48 space-y-0.5 overflow-y-auto">
              {roundKills.length === 0 && (
                <div className="text-xs text-muted">No kills</div>
              )}
              {roundKills.map((k, i) => (
                <div key={i} className="text-xs">
                  <span className="font-medium">{k.attacker_name}</span>
                  <span className="text-muted">
                    {" "}
                    [{k.weapon}]{" "}
                  </span>
                  <span className="font-medium">{k.victim_name}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>

      {/* Timeline */}
      <div className="border-t border-border px-6 py-4">
        <div className="mx-auto max-w-7xl">
          <TimelineSlider
            frameIndex={frameIndex}
            frameCount={roundData?.frames.length ?? 0}
            onFrameChange={handleFrameChange}
            kills={roundKills}
            startTick={roundData?.start_tick ?? 0}
            endTick={roundData?.end_tick ?? 1}
          />
        </div>
      </div>
    </div>
  );
}
