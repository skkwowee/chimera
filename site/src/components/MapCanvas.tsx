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

interface Props {
  mapName: string;
  players: PlayerFrame[];
  mapData: Record<string, MapInfo> | null;
  killLines?: KillLine[];
}

const CANVAS_SIZE = 1024;
const DOT_RADIUS = 8;
const T_COLOR = "#f59e0b"; // amber-500
const CT_COLOR = "#3b82f6"; // blue-500
const DEAD_ALPHA = 0.3;
const CONE_RADIUS = 60; // pixels
const CONE_FOV = Math.PI / 2; // 90 degrees
const CONE_ALPHA = 0.15;

export default function MapCanvas({ mapName, players, mapData, killLines }: Props) {
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

      // Vision cone (alive players with yaw data only)
      if (p.alive && p.yaw != null) {
        const angle = yawToCanvasAngle(p.yaw);
        ctx.save();
        ctx.globalAlpha = CONE_ALPHA * levelAlpha;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.arc(px, py, CONE_RADIUS, angle - CONE_FOV / 2, angle + CONE_FOV / 2);
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
  }, [players, mapName, info, imgLoaded, worldToPixel, hasLower, yawToCanvasAngle, killLines]);

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
