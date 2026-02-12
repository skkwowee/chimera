"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface PlayerFrame {
  name: string;
  side: string;
  x: number;
  y: number;
  z: number;
  yaw?: number;
  hp: number;
  alive: boolean;
}

export interface MapInfo {
  pos_x: number;
  pos_y: number;
  scale: number;
  lower_level_max_units: number;
}

export interface KillLine {
  attackerX: number;
  attackerY: number;
  victimX: number;
  victimY: number;
  attackerSide: string;
}

export interface DamageLine {
  attackerX: number;
  attackerY: number;
  victimX: number;
  victimY: number;
  attackerSide: string;
  damage: number;
  weapon: string;
}

export interface ShotTracer {
  x: number;
  y: number;
  yaw: number;
  side: string;
}

interface Props {
  mapName: string;
  players: PlayerFrame[];
  mapData: Record<string, MapInfo> | null;
  killLines?: KillLine[];
  damageLines?: DamageLine[];
  shotTracers?: ShotTracer[];
}

const CANVAS_SIZE = 1024;
const DOT_RADIUS = 8;
const T_COLOR = "#f59e0b"; // amber-500
const CT_COLOR = "#3b82f6"; // blue-500
const DEAD_ALPHA = 0.3;
const CONE_WORLD_RADIUS = 1500; // world units — typical CS2 sightline
const CONE_FOV = Math.PI / 2; // 90 deg horizontal FOV (4:3 @ 1280x960)
const TRACER_LENGTH = 400; // world units for shot tracer line
const OTHER_LEVEL_ALPHA = 0.15; // opacity for players on the other level
const RAY_COUNT = 90;
const RAY_STEP = 1; // 1px steps for reliable thin-wall detection
const ALPHA_THRESH = 128;
const LUM_WALL_THRESH = 45; // opaque pixels darker than this are wall outlines

/**
 * Extract a binary wall mask from a radar image.
 * Walls are detected as:
 * 1. Alpha boundaries — edges between opaque map geometry and transparent areas
 * 2. Dark opaque pixels — wall outlines between adjacent indoor rooms
 */
function extractWallMask(img: HTMLImageElement): Uint8Array {
  const offscreen = document.createElement("canvas");
  offscreen.width = CANVAS_SIZE;
  offscreen.height = CANVAS_SIZE;
  const ctx = offscreen.getContext("2d")!;
  ctx.drawImage(img, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
  const imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  const data = imageData.data;
  const mask = new Uint8Array(CANVAS_SIZE * CANVAS_SIZE);

  for (let y = 0; y < CANVAS_SIZE; y++) {
    for (let x = 0; x < CANVAS_SIZE; x++) {
      const i = y * CANVAS_SIZE + x;
      const off = i * 4;
      const alpha = data[off + 3];
      const isOpaque = alpha >= ALPHA_THRESH;

      // Check if this pixel sits on an alpha boundary (opaque ↔ transparent)
      let isAlphaBoundary = false;
      if (x > 0) {
        const nAlpha = data[(i - 1) * 4 + 3];
        if ((nAlpha >= ALPHA_THRESH) !== isOpaque) isAlphaBoundary = true;
      }
      if (!isAlphaBoundary && x < CANVAS_SIZE - 1) {
        const nAlpha = data[(i + 1) * 4 + 3];
        if ((nAlpha >= ALPHA_THRESH) !== isOpaque) isAlphaBoundary = true;
      }
      if (!isAlphaBoundary && y > 0) {
        const nAlpha = data[(i - CANVAS_SIZE) * 4 + 3];
        if ((nAlpha >= ALPHA_THRESH) !== isOpaque) isAlphaBoundary = true;
      }
      if (!isAlphaBoundary && y < CANVAS_SIZE - 1) {
        const nAlpha = data[(i + CANVAS_SIZE) * 4 + 3];
        if ((nAlpha >= ALPHA_THRESH) !== isOpaque) isAlphaBoundary = true;
      }

      if (isAlphaBoundary) {
        mask[i] = 1;
      } else if (isOpaque) {
        const lum = (data[off] + data[off + 1] + data[off + 2]) / 3;
        if (lum < LUM_WALL_THRESH) {
          mask[i] = 1;
        }
      }
    }
  }

  return mask;
}

/** Raycast a vision cone, returning boundary points clipped to walls. */
function raycastCone(
  cx: number,
  cy: number,
  angleCenter: number,
  fovRad: number,
  maxRadius: number,
  wallMask: Uint8Array | null,
): Array<[number, number]> {
  const points: Array<[number, number]> = [];
  if (!wallMask) {
    // Fallback: return unclipped arc points
    for (let i = 0; i <= RAY_COUNT; i++) {
      const a = angleCenter - fovRad / 2 + (fovRad * i) / RAY_COUNT;
      points.push([cx + Math.cos(a) * maxRadius, cy + Math.sin(a) * maxRadius]);
    }
    return points;
  }
  for (let i = 0; i <= RAY_COUNT; i++) {
    const a = angleCenter - fovRad / 2 + (fovRad * i) / RAY_COUNT;
    const cosA = Math.cos(a);
    const sinA = Math.sin(a);
    let dist = RAY_STEP;
    let hitDist = maxRadius;
    while (dist <= maxRadius) {
      const sx = Math.round(cx + cosA * dist);
      const sy = Math.round(cy + sinA * dist);
      if (sx < 0 || sx >= CANVAS_SIZE || sy < 0 || sy >= CANVAS_SIZE) {
        hitDist = dist;
        break;
      }
      if (wallMask[sy * CANVAS_SIZE + sx]) {
        hitDist = dist;
        break;
      }
      dist += RAY_STEP;
    }
    points.push([cx + cosA * hitDist, cy + sinA * hitDist]);
  }
  return points;
}

function drawOnCanvas(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement | null,
  imgLoaded: boolean,
  mapName: string,
  info: MapInfo,
  players: PlayerFrame[],
  worldToPixel: (wx: number, wy: number) => [number, number],
  yawToCanvasAngle: (yawDeg: number) => number,
  killLines: KillLine[] | undefined,
  damageLines: DamageLine[] | undefined,
  shotTracers: ShotTracer[] | undefined,
  levelFilter: "upper" | "lower" | "all",
  lowerThreshold: number,
  wallMask: Uint8Array | null,
) {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // Draw radar background
  if (img && imgLoaded) {
    ctx.drawImage(img, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
  } else {
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.fillStyle = "#666";
    ctx.font = "16px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`Loading ${mapName}...`, CANVAS_SIZE / 2, CANVAS_SIZE / 2);
  }

  const isPlayerOnLevel = (z: number) => {
    if (levelFilter === "all") return true;
    const isLower = z < lowerThreshold;
    return levelFilter === "lower" ? isLower : !isLower;
  };

  // Draw shot tracers
  if (shotTracers && shotTracers.length > 0) {
    const tracerPx = TRACER_LENGTH / info.scale;
    for (const st of shotTracers) {
      const [sx, sy] = worldToPixel(st.x, st.y);
      const angle = yawToCanvasAngle(st.yaw);
      const ex = sx + Math.cos(angle) * tracerPx;
      const ey = sy + Math.sin(angle) * tracerPx;
      ctx.save();
      ctx.globalAlpha = 0.25;
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 0.75;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
      ctx.restore();
    }
  }

  // Draw damage lines
  if (damageLines && damageLines.length > 0) {
    for (const dl of damageLines) {
      const [ax, ay] = worldToPixel(dl.attackerX, dl.attackerY);
      const [vx, vy] = worldToPixel(dl.victimX, dl.victimY);
      const color = dl.attackerSide === "CT" ? CT_COLOR : T_COLOR;
      ctx.save();
      ctx.globalAlpha = Math.min(0.6, 0.2 + dl.damage / 150);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(vx, vy);
      ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.globalAlpha = Math.min(0.7, 0.25 + dl.damage / 120);
      ctx.beginPath();
      ctx.arc(vx, vy, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  // Draw kill lines
  if (killLines && killLines.length > 0) {
    for (const kl of killLines) {
      const [ax, ay] = worldToPixel(kl.attackerX, kl.attackerY);
      const [vx, vy] = worldToPixel(kl.victimX, kl.victimY);
      ctx.save();
      ctx.globalAlpha = 0.8;
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(vx, vy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2.5;
      const s = 6;
      ctx.beginPath();
      ctx.moveTo(vx - s, vy - s);
      ctx.lineTo(vx + s, vy + s);
      ctx.moveTo(vx + s, vy - s);
      ctx.lineTo(vx - s, vy + s);
      ctx.stroke();
      ctx.restore();
    }
  }

  // Draw players
  for (const p of players) {
    const [px, py] = worldToPixel(p.x, p.y);

    if (px < -20 || px > CANVAS_SIZE + 20 || py < -20 || py > CANVAS_SIZE + 20) continue;

    const onThisLevel = isPlayerOnLevel(p.z);
    const levelAlpha = onThisLevel ? 1.0 : OTHER_LEVEL_ALPHA;

    const alpha = p.alive ? levelAlpha : DEAD_ALPHA * levelAlpha;
    const color = p.side === "CT" ? CT_COLOR : T_COLOR;

    ctx.globalAlpha = alpha;

    // Vision cone (only for players on this level), clipped to walls
    if (p.alive && p.yaw != null && onThisLevel) {
      const angle = yawToCanvasAngle(p.yaw);
      const coneRadius = CONE_WORLD_RADIUS / info.scale;

      ctx.save();
      ctx.globalAlpha = levelAlpha;

      const grad = ctx.createRadialGradient(px, py, 0, px, py, coneRadius);
      const r = parseInt(color.slice(1, 3), 16);
      const g = parseInt(color.slice(3, 5), 16);
      const b = parseInt(color.slice(5, 7), 16);
      const base = 0.12;
      const e1 = Math.E - 1;
      for (const t of [0, 0.03, 0.08, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0]) {
        const a = t === 0 ? base : base * Math.max(0, 1 - Math.log(1 + t * e1));
        grad.addColorStop(t, `rgba(${r},${g},${b},${a.toFixed(3)})`);
      }

      const conePoints = raycastCone(px, py, angle, CONE_FOV, coneRadius, wallMask);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(px, py);
      for (const [bx, by] of conePoints) {
        ctx.lineTo(bx, by);
      }
      ctx.closePath();
      ctx.fill();
      ctx.restore();
      ctx.globalAlpha = alpha;
    }

    // Player dot
    ctx.beginPath();
    ctx.arc(px, py, DOT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Dead X marker
    if (!p.alive) {
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2;
      const s = DOT_RADIUS * 0.7;
      ctx.beginPath();
      ctx.moveTo(px - s, py - s);
      ctx.lineTo(px + s, py + s);
      ctx.moveTo(px + s, py - s);
      ctx.lineTo(px - s, py + s);
      ctx.stroke();
    }

    // Level chevron (only in single-canvas "all" mode)
    if (p.alive && levelFilter === "all" && lowerThreshold > -999999) {
      const isLower = p.z < lowerThreshold;
      ctx.fillStyle = "#fff";
      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(isLower ? "\u25BC" : "\u25B2", px, py - DOT_RADIUS - 2);
    }

    // Player name label
    if (onThisLevel) {
      ctx.fillStyle = "#fff";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(p.name, px, py + DOT_RADIUS + 12);
    }

    ctx.globalAlpha = 1;
  }
}

export default function MapCanvas({ mapName, players, mapData, killLines, damageLines, shotTracers }: Props) {
  const upperCanvasRef = useRef<HTMLCanvasElement>(null);
  const lowerCanvasRef = useRef<HTMLCanvasElement>(null);
  const upperImgRef = useRef<HTMLImageElement | null>(null);
  const lowerImgRef = useRef<HTMLImageElement | null>(null);
  const upperWallMaskRef = useRef<Uint8Array | null>(null);
  const lowerWallMaskRef = useRef<Uint8Array | null>(null);
  const [upperImgLoaded, setUpperImgLoaded] = useState(false);
  const [lowerImgLoaded, setLowerImgLoaded] = useState(false);
  const [tooltip, setTooltip] = useState<{
    player: PlayerFrame;
    x: number;
    y: number;
  } | null>(null);

  const info = mapData?.[mapName];
  const hasLower = info != null && info.lower_level_max_units > -999999;

  // Load upper radar image + extract wall mask
  useEffect(() => {
    setUpperImgLoaded(false);
    upperWallMaskRef.current = null;
    const img = new Image();
    img.src = `/maps/${mapName}.png`;
    img.onload = () => {
      upperImgRef.current = img;
      upperWallMaskRef.current = extractWallMask(img);
      setUpperImgLoaded(true);
    };
    img.onerror = () => {
      upperImgRef.current = null;
      upperWallMaskRef.current = null;
      setUpperImgLoaded(false);
    };
  }, [mapName]);

  // Load lower radar image + extract wall mask (only for multi-level maps)
  useEffect(() => {
    if (!hasLower) {
      lowerImgRef.current = null;
      lowerWallMaskRef.current = null;
      setLowerImgLoaded(false);
      return;
    }
    setLowerImgLoaded(false);
    lowerWallMaskRef.current = null;
    const img = new Image();
    img.src = `/maps/${mapName}_lower.png`;
    img.onload = () => {
      lowerImgRef.current = img;
      lowerWallMaskRef.current = extractWallMask(img);
      setLowerImgLoaded(true);
    };
    img.onerror = () => {
      lowerImgRef.current = null;
      lowerWallMaskRef.current = null;
      setLowerImgLoaded(false);
    };
  }, [mapName, hasLower]);

  const worldToPixel = useCallback(
    (wx: number, wy: number): [number, number] => {
      if (!info) return [0, 0];
      const px = (wx - info.pos_x) / info.scale;
      const py = (info.pos_y - wy) / info.scale;
      return [px, py];
    },
    [info]
  );

  const yawToCanvasAngle = useCallback((yawDeg: number): number => {
    const yawRad = (yawDeg * Math.PI) / 180;
    return -yawRad;
  }, []);

  // Draw
  useEffect(() => {
    if (!info) return;
    const lowerThreshold = info.lower_level_max_units;

    // Upper canvas (or single canvas for maps without lower level)
    const upperCanvas = upperCanvasRef.current;
    if (upperCanvas) {
      const ctx = upperCanvas.getContext("2d");
      if (ctx) {
        drawOnCanvas(
          ctx, upperImgRef.current, upperImgLoaded, mapName, info,
          players, worldToPixel, yawToCanvasAngle,
          killLines, damageLines, shotTracers,
          hasLower ? "upper" : "all", lowerThreshold,
          upperWallMaskRef.current,
        );
      }
    }

    // Lower canvas (multi-level maps only)
    if (hasLower) {
      const lowerCanvas = lowerCanvasRef.current;
      if (lowerCanvas) {
        const ctx = lowerCanvas.getContext("2d");
        if (ctx) {
          drawOnCanvas(
            ctx, lowerImgRef.current, lowerImgLoaded, mapName, info,
            players, worldToPixel, yawToCanvasAngle,
            killLines, damageLines, shotTracers,
            "lower", lowerThreshold,
            lowerWallMaskRef.current,
          );
        }
      }
    }
  }, [players, mapName, info, upperImgLoaded, lowerImgLoaded, worldToPixel, hasLower, yawToCanvasAngle, killLines, damageLines, shotTracers]);

  // Mouse hover for tooltip
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = e.currentTarget;
      if (!canvas || !info) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = CANVAS_SIZE / rect.width;
      const scaleY = CANVAS_SIZE / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;

      let found: PlayerFrame | null = null;
      for (const p of players) {
        const [px, py] = worldToPixel(p.x, p.y);
        const dx = mx - px;
        const dy = my - py;
        if (dx * dx + dy * dy < (DOT_RADIUS + 4) ** 2) {
          found = p;
          break;
        }
      }

      if (found) {
        setTooltip({
          player: found,
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
        });
      } else {
        setTooltip(null);
      }
    },
    [players, info, worldToPixel]
  );

  const tooltipEl = tooltip && (
    <div
      className="pointer-events-none absolute z-50 rounded border border-border bg-card px-3 py-2 text-xs shadow-lg"
      style={{
        left: tooltip.x + 12,
        top: tooltip.y - 10,
      }}
    >
      <div className="font-semibold">
        <span
          className={
            tooltip.player.side === "CT" ? "text-blue-400" : "text-amber-400"
          }
        >
          [{tooltip.player.side}]
        </span>{" "}
        {tooltip.player.name}
      </div>
      <div className="text-muted">
        HP: {tooltip.player.hp} | Z: {tooltip.player.z.toFixed(0)}
      </div>
      <div className="text-muted">
        {tooltip.player.alive ? "Alive" : "Dead"}
        {tooltip.player.yaw != null && ` | Yaw: ${tooltip.player.yaw.toFixed(0)}\u00B0`}
      </div>
    </div>
  );

  if (hasLower) {
    return (
      <div className="flex gap-3">
        <div className="relative flex-1">
          <div className="mb-1 text-center text-xs font-semibold uppercase tracking-wider text-muted">
            Upper
          </div>
          <canvas
            ref={upperCanvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            className="h-full w-full rounded-lg border border-border"
            style={{ aspectRatio: "1/1" }}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setTooltip(null)}
          />
          {tooltipEl}
        </div>
        <div className="relative flex-1">
          <div className="mb-1 text-center text-xs font-semibold uppercase tracking-wider text-muted">
            Lower
          </div>
          <canvas
            ref={lowerCanvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            className="h-full w-full rounded-lg border border-border"
            style={{ aspectRatio: "1/1" }}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setTooltip(null)}
          />
          {tooltipEl}
        </div>
      </div>
    );
  }

  return (
    <div className="relative">
      <canvas
        ref={upperCanvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        className="h-full w-full rounded-lg border border-border"
        style={{ maxWidth: CANVAS_SIZE, aspectRatio: "1/1" }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      />
      {tooltipEl}
    </div>
  );
}
