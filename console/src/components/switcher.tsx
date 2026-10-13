"use client";

import { useEffect, useRef, useState } from "react";
import { Icon } from "./chrome/icon";
import { sameExample, type Example } from "@/lib/examples";

/**
 * The example this console talks to, as a chip; a click opens the list under
 * it, the popover of the sessions strip. One row per example, the chosen one
 * strong, one that is not running muted and said so. Picking one that is not
 * running is fine: the stage then shows what to start.
 */
export function ExamplePicker({
  examples,
  selected,
  onSelect,
}: {
  examples: Example[];
  selected: Example | null;
  onSelect: (example: Example) => void;
}) {
  const [shown, setToggled] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  /* The list closes on a click anywhere else. */
  useEffect(() => {
    if (!shown) return;
    const onDown = (event: PointerEvent) => {
      if (!ref.current?.contains(event.target as Node)) setToggled(false);
    };
    window.addEventListener("pointerdown", onDown);
    return () => window.removeEventListener("pointerdown", onDown);
  }, [shown]);

  if (examples.length === 0) return null;
  return (
    <div ref={ref} className="relative flex flex-none items-center" data-example-picker>
      <button type="button" className="chip" aria-expanded={shown} onClick={() => setToggled(!shown)}>
        {selected ? selected.name : "Examples"}
      </button>
      {shown ? (
        <div className="absolute top-full right-0 z-20 mt-2 grid w-[var(--w-info)] rounded-blender bg-raised py-1 shadow-[var(--shadow)]" data-switcher>
          {examples.map((example) => {
            const chosen = sameExample(example, selected);
            return (
              <div key={`${example.agent}/${example.endpoint}`} className="grid" data-example={example.agent} data-live={example.live}>
                <button
                  type="button"
                  className="flex min-h-row items-center gap-3 px-3 text-left text-ink hover:bg-header-quaternary data-[live=false]:text-label data-[selected=true]:text-strong"
                  title={example.description}
                  data-selected={chosen}
                  data-live={example.live}
                  aria-pressed={chosen}
                  onClick={() => {
                    setToggled(false);
                    onSelect(example);
                  }}
                >
                  <span className="min-w-0 flex-1 truncate">{example.name}</span>
                  {example.voice ? <Icon name="voice" size={12} className="flex-none" /> : null}
                  {example.live ? null : <span className="flex-none text-label">not running</span>}
                </button>

              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
