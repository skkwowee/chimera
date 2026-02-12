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

export default function MapCanvas({ mapName, players, mapData, killLines, damageLines, shotTracers }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgLoaded, setImgLoaded] = useState(false);
  const [tooltip, setTooltip] = useState<{
    player: PlayerFrame;
    x: number;
    y: number;
  } | null>(null);

  const info = mapData?.[mapName];
  const hasLower =
    info && info.lower_level_max_units > -999999;

  // Load radar image
  useEffect(() => {
    setImgLoaded(false);
    const img = new Image();
    img.src = `/maps/${mapName}.png`;
    img.onload = () => {
      imgRef.current = img;
      setImgLoaded(true);
    };
    img.onerror = () => {
      imgRef.current = null;
      setImgLoaded(false);
    };
  }, [mapName]);

  // World → pixel coordinate transform
  const worldToPixel = useCallback(
    (wx: number, wy: number): [number, number] => {
      if (!info) return [0, 0];
      const px = (wx - info.pos_x) / info.scale;
      const py = (info.pos_y - wy) / info.scale;
      return [px, py];
    },
    [info]
  );

  // Convert CS2 yaw (degrees) to canvas angle (radians).
  // CS2 yaw: 0 = +X axis, 90 = +Y axis (counter-clockwise in world).
  // Canvas: 0 = right, positive = clockwise.
  // Since pixel_y is flipped (pos_y - world_Y), we negate the Y component.
  const yawToCanvasAngle = useCallback((yawDeg: number): number => {
    // CS2 yaw in radians (counter-clockwise from +X in world space)
    const yawRad = (yawDeg * Math.PI) / 180;
    // In canvas coords, Y is inverted, so angle becomes clockwise
    return -yawRad;
  }, []);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Draw radar background
    if (imgRef.current && imgLoaded) {
      ctx.drawImage(imgRef.current, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
    } else {
      ctx.fillStyle = "#1a1a2e";
      ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      ctx.fillStyle = "#666";
      ctx.font = "16px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`Loading ${mapName}...`, CANVAS_SIZE / 2, CANVAS_SIZE / 2);
    }

    if (!info) return;

    // Draw shot tracers (missed shots — short directional line from shooter)
    if (shotTracers && shotTracers.length > 0) {
      const tracerPx = TRACER_LENGTH / info.scale;
      for (const st of shotTracers) {
        const [sx, sy] = worldToPixel(st.x, st.y);
        const angle = yawToCanvasAngle(st.yaw);
        const ex = sx + Math.cos(angle) * tracerPx;
        const ey = sy + Math.sin(angle) * tracerPx;
        ctx.save();
        ctx.globalAlpha = 0.25;
        ctx.strokeStyle = "#fbbf24"; // amber-400
        ctx.lineWidth = 0.75;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.restore();
      }
    }

    // Draw damage lines (gunfire hits — thinner and more subtle than kill lines)
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
        // Small dot at hit point
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
        // X at victim
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

      // Skip if off-canvas
      if (px < -20 || px > CANVAS_SIZE + 20 || py < -20 || py > CANVAS_SIZE + 20) continue;

      // Dim players on wrong level for multi-level maps
      const isLower = hasLower && p.z < info.lower_level_max_units;
      const levelAlpha = isLower ? 0.4 : 1.0;

      const alpha = p.alive ? levelAlpha : DEAD_ALPHA * levelAlpha;
      const color = p.side === "CT" ? CT_COLOR : T_COLOR;

      ctx.globalAlpha = alpha;

      // Vision cone with log falloff (alive players with yaw data only)
      if (p.alive && p.yaw != null) {
        const angle = yawToCanvasAngle(p.yaw);
        const coneRadius = CONE_WORLD_RADIUS / info.scale; // world units → pixels

        ctx.save();
        ctx.globalAlpha = levelAlpha;

        // Radial gradient with log-curve falloff
        const grad = ctx.createRadialGradient(px, py, 0, px, py, coneRadius);
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        // Log falloff: a = base * max(0, 1 - ln(1 + t*(e-1)))
        const base = 0.12;
        const e1 = Math.E - 1;
        for (const t of [0, 0.03, 0.08, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0]) {
          const a = t === 0 ? base : base * Math.max(0, 1 - Math.log(1 + t * e1));
          grad.addColorStop(t, `rgba(${r},${g},${b},${a.toFixed(3)})`);
        }

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.arc(px, py, coneRadius, angle - CONE_FOV / 2, angle + CONE_FOV / 2);
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

      // Z elevation chevron
      if (p.alive && hasLower) {
        ctx.fillStyle = "#fff";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(isLower ? "\u25BC" : "\u25B2", px, py - DOT_RADIUS - 2);
      }

      // Player name label
      ctx.fillStyle = "#fff";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(p.name, px, py + DOT_RADIUS + 12);

      ctx.globalAlpha = 1;
    }
  }, [players, mapName, info, imgLoaded, worldToPixel, hasLower, yawToCanvasAngle, killLines, damageLines, shotTracers]);

  // Mouse hover for tooltip
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
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

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        className="h-full w-full rounded-lg border border-border"
        style={{ maxWidth: CANVAS_SIZE, aspectRatio: "1/1" }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      />
      {tooltip && (
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
      )}
    </div>
  );
}
