"use client";

import { createContext, useContext, useState, type ReactNode } from "react";
import { pretty } from "@/lib/format";
import { Icon } from "./icon";

/** Inside the debug drawer, whose tab already names the view. */
export const InDrawer = createContext(false);

/**
 * A side panel: a header with the view's title, an optional count and its
 * actions, the body under it.
 */
export function Panel({
  title,
  meta,
  actions,
  children,
  id,
}: {
  title: string;
  meta?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  id: string;
}) {
  const drawer = useContext(InDrawer);
  return (
    <section className="grid h-full min-h-0 w-full min-w-0 grid-cols-[minmax(0,1fr)] grid-rows-[auto_minmax(0,1fr)] overflow-hidden bg-sunken" data-panel={id}>
      {drawer && !meta && !actions ? (
        <span />
      ) : (
        <header className="flex min-h-header min-w-0 items-center gap-2 px-4 data-[in-drawer=false]:h-[calc(var(--h-header)+var(--pad-3))]" data-in-drawer={drawer}>
          {drawer ? null : <h2 className="m-0 text-[length:var(--fs-title-sm)] font-medium text-strong">{title}</h2>}
          {meta ? <span className="truncate text-label tabular-nums">{meta}</span> : null}
          <span className="flex-1" />
          {actions}
        </header>
      )}
      <div className="min-h-0 overflow-auto">{children}</div>
    </section>
  );
}

/** What a view says when it has nothing: quiet, and what would make something appear. */
export function Empty({ children }: { children: ReactNode }) {
  return <p className="m-0 flex min-h-row items-center px-4 py-1 text-label">{children}</p>;
}

/** A label and its value on one 18px row, the value in the control well. */
export function Field({ label, children, tone }: { label: string; children: ReactNode; tone?: "attention" | "failed" }) {
  return (
    <div className="grid min-h-row grid-cols-[var(--w-label)_minmax(0,1fr)] items-center gap-3">
      <span className="truncate text-label">{label}</span>
      <span
        className="min-w-0 truncate font-mono text-micro text-ink data-[tone=attention]:text-attention data-[tone=failed]:text-failed-text"
        title={typeof children === "string" ? children : undefined}
        data-tone={tone}
      >
        {children}
      </span>
    </div>
  );
}

/**
 * A disclosure: a chevron (right closed, down open) and the label of what
 * it opens, drawn as a quiet control so it reads as one before it is
 * hovered. Controlled or not.
 */
export function Disclosure({
  label,
  open,
  onToggle,
  children,
  data,
}: {
  label: string;
  open?: boolean;
  onToggle?: () => void;
  children: ReactNode;
  data?: string;
}) {
  const [own, setOwn] = useState(false);
  const isOpen = open ?? own;
  const toggle = onToggle ?? (() => setOwn((value) => !value));
  return (
    <div className="grid min-w-0 gap-1" data-fold={data ?? label} data-open={isOpen}>
      <button
        type="button"
        className="btn btnQuiet h-row-sm justify-self-start px-1 text-micro"
        aria-expanded={isOpen}
        onClick={(event) => {
          event.stopPropagation();
          toggle();
        }}
      >
        <Icon name={isOpen ? "disclosureDown" : "disclosureRight"} size={12} />
        {label}
      </button>
      {isOpen ? children : null}
    </div>
  );
}

/** A document, folded behind a disclosure that says what it is. */
export function Json({ value, label }: { value: unknown; label: string }) {
  if (value == null) return null;
  return (
    <Disclosure label={label}>
      <Document value={value} />
    </Disclosure>
  );
}

/** A JSON document, pretty, in the sunken well. */
export function Document({ value }: { value: unknown }) {
  return (
    <pre className="m-0 max-h-[calc(var(--h-row)*14)] overflow-auto rounded-blender bg-sunken px-3 py-2 font-mono text-micro wrap-break-word whitespace-pre-wrap">
      {pretty(value)}
    </pre>
  );
}

/**
 * Blender's list: a 20px row per entry, filled on hover, the accent when
 * selected. A div grid, not a table, so rows keep their height.
 */
export function ListRow({
  children,
  selected,
  onSelect,
  className = "",
  data,
}: {
  children: ReactNode;
  selected?: boolean;
  onSelect?: () => void;
  className?: string;
  data?: Record<string, string | number | boolean | undefined>;
}) {
  const attrs: Record<string, string> = {};
  for (const [name, value] of Object.entries(data ?? {})) if (value !== undefined) attrs[`data-${name}`] = String(value);
  return (
    <button
      type="button"
      className={`grid min-h-row w-full items-center gap-3 border-0 bg-transparent px-4 py-1 text-left text-ink transition-colors duration-fast ease-blender hover:bg-header-quaternary data-[selected=true]:bg-raised data-[selected=true]:shadow-[inset_var(--pad-1)_0_0_var(--accent)] data-[selected=true]:hover:bg-raised ${className}`}
      data-selected={Boolean(selected)}
      aria-pressed={Boolean(selected)}
      onClick={onSelect}
      {...attrs}
    >
      {children}
    </button>
  );
}
