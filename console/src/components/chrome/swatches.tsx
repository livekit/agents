"use client";

import { useEffect, useState } from "react";

const SWATCHES = ["cyan", "green", "amber", "blue", "violet", "rose", "pink", "teal"] as const;
type Swatch = (typeof SWATCHES)[number];
const KEY = "console.accent";

/** The accent swatches: one dot per palette; the pick recolours the orb, the selection and the focus. */
export function Swatches() {
  const [accent, setAccent] = useState<Swatch>(() => {
    try {
      const saved = window.localStorage.getItem(KEY);
      return SWATCHES.includes(saved as Swatch) ? (saved as Swatch) : "blue";
    } catch {
      return "blue";
    }
  });

  useEffect(() => {
    document.documentElement.dataset.accent = accent;
    try {
      window.localStorage.setItem(KEY, accent);
    } catch {}
  }, [accent]);

  return (
    <div className="flex items-center gap-2" role="group" aria-label="Accent colour" data-swatches="true">
      {SWATCHES.map((swatch) => (
        <button
          key={swatch}
          type="button"
          className="swatch"
          style={{ "--swatch": `var(--swatch-${swatch})` } as React.CSSProperties}
          aria-pressed={swatch === accent}
          aria-label={swatch}
          title={swatch}
          data-swatch={swatch}
          onClick={() => setAccent(swatch)}
        />
      ))}
    </div>
  );
}
