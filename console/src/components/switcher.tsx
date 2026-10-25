"use client";

import { useEffect, useRef, useState } from "react";
import { Icon } from "./chrome/icon";
import { sameExample, type Example } from "@/lib/examples";

/**
 * The header's name, which agent and endpoint this console talks to, and under
 * it the server the endpoint is on; a click drops every example under it by
 * name, the ones not running dimmed.
 * Picking one that is not running is fine: the stage then shows what to start.
 */
export function ExamplePicker({
  examples,
  selected,
  fallback,
  onSelect,
}: {
  examples: Example[];
  selected: Example | null;
  /** The name when there is no example to pick. */
  fallback: string;
  onSelect: (example: Example) => void;
}) {
  const [shown, setShown] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  /* The list closes on a click anywhere else, or Escape. */
  useEffect(() => {
    if (!shown) return;
    const onDown = (event: PointerEvent) => {
      if (!ref.current?.contains(event.target as Node)) setShown(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setShown(false);
    };
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [shown]);

  const name = selected ? `${selected.agent} / ${selected.endpoint}` : fallback || "Agent";
  if (examples.length === 0) return <>{name}</>;
  return (
    <span ref={ref} className="relative inline-grid max-w-full justify-items-start group-data-[size=large]:justify-items-center" data-example-picker>
      <button
        type="button"
        className="group/picker inline-flex max-w-full items-center gap-1 font-[inherit] text-[length:inherit] text-inherit"
        aria-haspopup="listbox"
        aria-expanded={shown}
        onClick={() => setShown(!shown)}
      >
        <span className="truncate">{name}</span>
        <Icon name={shown ? "disclosureUp" : "disclosureDown"} size={14} className="flex-none text-label group-hover/picker:text-ink" />
      </button>
      {selected ? (
        <span className="max-w-full truncate text-micro text-label" title={selected.url} data-endpoint-host>
          {new URL(selected.url).host}
        </span>
      ) : null}
      {shown ? (
        <div
          className="absolute top-full left-0 z-20 mt-2 grid w-max max-w-[var(--w-info)] grid-cols-[minmax(0,1fr)] rounded-blender border border-line-weak bg-raised py-1 text-left font-ui text-[length:var(--fs-ui)] shadow-[var(--shadow)] group-data-[size=large]:left-1/2 group-data-[size=large]:-translate-x-1/2"
          role="listbox"
          data-switcher
        >
          {examples.map((example) => {
            const chosen = sameExample(example, selected);
            return (
              <button
                key={`${example.agent}/${example.endpoint}`}
                type="button"
                role="option"
                aria-selected={chosen}
                className="mx-1 flex min-h-row min-w-0 items-center gap-2 rounded-blender px-2 text-left text-ink hover:bg-header-quaternary data-[live=false]:text-label"
                title={example.description}
                data-example={example.agent}
                data-live={example.live}
                onClick={() => {
                  setShown(false);
                  onSelect(example);
                }}
              >
                <Icon name="check" size={12} className={`flex-none ${chosen ? "" : "invisible"}`} />
                <span className="min-w-0 flex-1 truncate">{example.agent}</span>
                {example.live ? null : <span className="ml-4 flex-none text-micro">not running</span>}
              </button>
            );
          })}
        </div>
      ) : null}
    </span>
  );
}
