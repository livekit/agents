"use client";

import { useEffect, useRef } from "react";
import type { LocalAudioTrack, RemoteAudioTrack } from "livekit-client";
import { useMultibandTrackVolume, type TrackReferenceOrPlaceholder } from "@livekit/components-react";

/** What the orb shows: the agent at rest, at work, waiting on the person, or on a call. */
export type OrbState = "listening" | "thinking" | "asked" | "connecting" | "speaking";

/* How each state moves: how fast the surface flows and how lit the sphere is. */
const MOTION: Record<OrbState, { speed: number; light: number }> = {
  listening: { speed: 1, light: 1 },
  asked: { speed: 1, light: 1 },
  thinking: { speed: 3, light: 1 },
  connecting: { speed: 1.5, light: 0.4 },
  speaking: { speed: 1.6, light: 1 },
};

/* One glyph per row, in CSS pixels; a glyph is a mono cell, about 0.6 as wide as tall. */
const CELL = { small: 4, medium: 4.5, large: 7 };
const ASPECT = 0.6;

/* From dark to lit. */
const RAMP = " .:-=+*x#%&@";

/** A smooth, cheap 3D pattern for the surface: three crossed waves. */
function flow(x: number, y: number, z: number, t: number): number {
  return (Math.sin(x * 3.1 + t) * Math.sin(y * 3.7 - t * 0.8) * Math.sin(z * 2.9 + t * 1.3) + 0.5 * Math.sin((x + y) * 5.3 - t * 1.7)) * 0.5;
}

/**
 * The agent's presence: a sphere drawn in glyphs, in the orb's colour, its surface
 * flowing as it turns. Small beside the agent's latest message and in the
 * call bar, over the conversation as its mark. The state sets its pace; while
 * the agent speaks on a call, the track's volume swells it.
 */
export function Orb({ state, size, audioTrack }: { state: OrbState; size: "small" | "medium" | "large"; audioTrack?: LocalAudioTrack | RemoteAudioTrack | TrackReferenceOrPlaceholder }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bands = useMultibandTrackVolume(audioTrack, { bands: 5, loPass: 100, hiPass: 200 });
  const volume = state === "speaking" ? Math.min(1, (bands.reduce((a, b) => a + b, 0) / bands.length) * 2.5) : 0;
  const target = useRef({ ...MOTION[state], volume: 0 });
  useEffect(() => {
    target.current = { ...MOTION[state], volume };
  }, [state, volume]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const context = canvas.getContext("2d")!;
    const style = getComputedStyle(canvas);
    const now = { ...target.current, time: 0 };
    let last = performance.now();
    let frame = 0;

    const draw = (at: number) => {
      const dpr = window.devicePixelRatio || 1;
      const width = Math.round(canvas.clientWidth * dpr);
      const height = Math.round(canvas.clientHeight * dpr);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      /* Ease toward the state's motion, so a change of state flows instead of jumping. */
      const step = Math.min(0.1, (at - last) / 1000);
      last = at;
      const ease = 1 - Math.exp(-step * 6);
      for (const key of ["speed", "light", "volume"] as const) now[key] += (target.current[key] - now[key]) * ease;
      now.time += step * now.speed;

      const cell = CELL[size] * dpr;
      const cellWidth = cell * ASPECT;
      const columns = Math.floor(width / cellWidth);
      const rows = Math.floor(height / cell);
      const unit = Math.min(width, height);
      const radius = 0.42 + 0.06 * now.volume;
      const turn = now.time * 0.4;

      context.clearRect(0, 0, width, height);
      context.font = `${cell}px ${style.fontFamily}`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillStyle = style.color;

      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const px = (column + 0.5) * cellWidth;
          const py = (row + 0.5) * cell;
          const x = (px - width / 2) / unit / radius;
          const y = (py - height / 2) / unit / radius;
          const r = x * x + y * y;
          if (r > 1) continue;
          const z = Math.sqrt(1 - r);

          /* Light from the upper left, the surface pattern turning around the vertical axis. */
          const shade = Math.max(0, -0.45 * x - 0.5 * y + 0.75 * z);
          const sx = x * Math.cos(turn) + z * Math.sin(turn);
          const sz = z * Math.cos(turn) - x * Math.sin(turn);
          const level = Math.min(1, Math.max(0, (0.15 + 0.55 * shade + 0.6 * flow(sx, y, sz, now.time) + 0.3 * now.volume) * now.light));

          const glyph = RAMP[Math.round(level * (RAMP.length - 1))];
          if (glyph === " ") continue;
          context.globalAlpha = 0.3 + 0.7 * level;
          context.fillText(glyph, px, py);
        }
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);

    return () => cancelAnimationFrame(frame);
  }, [size]);

  return (
    <canvas
      ref={canvasRef}
      className="block shrink-0 font-mono text-orb data-[size=large]:size-mark-lg data-[size=medium]:size-mark data-[size=small]:size-orb"
      data-orb={state}
      data-size={size}
      data-speaking={state === "speaking" || undefined}
      aria-hidden="true"
    />
  );
}
