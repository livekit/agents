"use client";

import { useEffect, useRef, useState } from "react";
import { Icon } from "./icon";

/** The text onto the clipboard; the old way when the API is not there (a plain http origin). */
export async function copyText(text: string) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      /* denied: fall through to the selection route */
    }
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  document.execCommand("copy");
  area.remove();
}

/**
 * An id that copies itself: a mono run, the whole id on hover, one click
 * puts it on the clipboard and the run says "copied" for a moment. No
 * dialog, no second step. `short` is what it shows; the full id is what it
 * copies.
 */
export function CopyId({ id, short, className = "", stop = true }: { id: string; short?: string; className?: string; stop?: boolean }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (timer.current) window.clearTimeout(timer.current);
    },
    [],
  );
  return (
    <button
      type="button"
      className={`btn btnQuiet h-row-sm max-w-full px-1 font-mono text-micro data-[copied=true]:text-ink ${className}`}
      title={copied ? "copied" : `${id} (click to copy)`}
      data-copy={id}
      data-copied={copied || undefined}
      onClick={(event) => {
        if (stop) event.stopPropagation();
        void copyText(id).then(() => {
          setCopied(true);
          if (timer.current) window.clearTimeout(timer.current);
          timer.current = window.setTimeout(() => setCopied(false), 1200);
        });
      }}
    >
      <span className="truncate">{copied ? "copied" : (short ?? id)}</span>
      <Icon name={copied ? "check" : "copy"} size={10} className="flex-none" />
    </button>
  );
}
