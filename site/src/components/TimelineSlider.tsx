"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface KillEvent {
  tick: number;
  attacker_name?: string;
  victim_name?: string;
  weapon?: string;
}

interface Props {
  frameIndex: number;
  frameCount: number;
  onFrameChange: (index: number) => void;
  kills: KillEvent[];
  startTick: number;
  endTick: number;
}

export default function TimelineSlider({
  frameIndex,
  frameCount,
  onFrameChange,
  kills,
  startTick,
  endTick,
}: Props) {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const rafRef = useRef<number>(0);
  const lastTime = useRef(0);
  const accum = useRef(0);

  const tickRange = endTick - startTick || 1;

  const animate = useCallback(
    (now: number) => {
      if (!playing) return;
      const dt = lastTime.current ? now - lastTime.current : 0;
      lastTime.current = now;
      accum.current += dt;

      // Advance one frame per (50ms / speed)
      const interval = 50 / speed;
      while (accum.current >= interval) {
        accum.current -= interval;
        onFrameChange(
          frameIndex < frameCount - 1 ? frameIndex + 1 : 0
        );
      }
      rafRef.current = requestAnimationFrame(animate);
    },
    [playing, speed, frameIndex, frameCount, onFrameChange]
  );

  useEffect(() => {
    if (playing) {
      lastTime.current = 0;
      accum.current = 0;
      rafRef.current = requestAnimationFrame(animate);
    }
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, animate]);

  return (
    <div className="flex flex-col gap-2">
      {/* Kill markers + slider */}
      <div className="relative">
        {/* Kill marker overlay */}
        <div className="pointer-events-none absolute inset-x-0 top-0 h-6">
          {kills.map((k, i) => {
            const pct = ((k.tick - startTick) / tickRange) * 100;
            if (pct < 0 || pct > 100) return null;
            return (
              <div
                key={i}
                className="absolute top-0 h-full w-0.5 bg-red-500/70"
                style={{ left: `${pct}%` }}
                title={`${k.attacker_name} killed ${k.victim_name} (${k.weapon})`}
              />
            );
          })}
        </div>

        <input
          type="range"
          min={0}
          max={Math.max(frameCount - 1, 0)}
          value={frameIndex}
          onChange={(e) => onFrameChange(Number(e.target.value))}
          className="relative z-10 w-full cursor-pointer"
        />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4 text-sm">
        <button
          onClick={() => setPlaying(!playing)}
          className="rounded border border-border px-3 py-1 hover:bg-card-hover"
        >
          {playing ? "Pause" : "Play"}
        </button>

        <label className="flex items-center gap-1 text-muted">
          Speed
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="rounded border border-border bg-card px-2 py-0.5 text-sm"
          >
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={4}>4x</option>
          </select>
        </label>

        <span className="ml-auto font-mono text-xs text-muted">
          Frame {frameIndex + 1} / {frameCount}
        </span>
      </div>
    </div>
  );
}
