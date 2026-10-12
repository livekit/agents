"use client";

import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { InDrawer } from "./chrome/panel";

/** The debug drawer's views: the session's database in the plan's words, and what the page talks to. */
export type View = "lanes" | "tasks" | "journal" | "inbox" | "stream" | "info";

export const VIEWS: { id: View; name: string; tip: string }[] = [
  { id: "lanes", name: "Lanes", tip: "One per agent in the conversation, and who holds its lease" },
  { id: "tasks", name: "Tasks", tip: "Every turn, and the expert's work" },
  { id: "journal", name: "Journal", tip: "What happened, in order" },
  { id: "inbox", name: "Inbox", tip: "Messages that waited for a turn" },
  { id: "stream", name: "Stream", tip: "A reply, frame by frame" },
  { id: "info", name: "Info", tip: "What the page talks to" },
];

const VIEW_KEY = "console.drawer.view";
const HEIGHT_KEY = "console.drawer.height";
const DEFAULT_HEIGHT = 320;
const MIN_HEIGHT = 160;
/** Never more of the window than this: the conversation stays the page. */
const MAX_SHARE = 0.7;

const isView = (value: unknown): value is View => VIEWS.some((view) => view.id === value);

function clampHeight(height: number) {
  return Math.max(MIN_HEIGHT, Math.min(height, Math.floor(window.innerHeight * MAX_SHARE)));
}

/** Which view is open (null: closed) and how tall, persisted; closed on a first load. */
export function useDrawer() {
  const [view, setViewState] = useState<View | null>(() => {
    if (typeof window === "undefined") return null;
    const saved = window.localStorage.getItem(VIEW_KEY);
    /* The stream view needs a selected task, which is not persisted. */
    return isView(saved) && saved !== "stream" ? saved : null;
  });
  const [height, setHeightState] = useState(() => {
    if (typeof window === "undefined") return DEFAULT_HEIGHT;
    return clampHeight(Number(window.localStorage.getItem(HEIGHT_KEY)) || DEFAULT_HEIGHT);
  });
  /* The view Cmd+. reopens: the last one open. */
  const last = useRef<View>(view ?? "tasks");
  const setView = useCallback((next: View | null) => {
    if (next) last.current = next;
    setViewState(next);
    window.localStorage.setItem(VIEW_KEY, next ?? "closed");
  }, []);
  const setHeight = useCallback((next: number) => {
    const clamped = clampHeight(next);
    setHeightState(clamped);
    window.localStorage.setItem(HEIGHT_KEY, String(clamped));
  }, []);
  /* A click on the open view's tab closes the drawer. */
  const toggle = useCallback((next: View) => setView(view === next ? null : next), [setView, view]);

  /* Cmd+. (Ctrl+.): the drawer on its last view, or closed. */
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "." || !(event.metaKey || event.ctrlKey)) return;
      event.preventDefault();
      setView(view ? null : last.current);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setView, view]);

  return { view, height, setView, setHeight, toggle };
}

/**
 * The debug drawer under the conversation, as a browser's devtools: a strip
 * of tabs always there at the foot ("Debug", then one tab per view), and
 * when one is open its view above the strip at a height the edge on top
 * changes. Nothing here is the product; it shows how the session works.
 */
export function Drawer({
  view,
  height,
  disabled = {},
  onToggle,
  onHeight,
  onDragging,
  children,
}: {
  view: View | null;
  height: number;
  /** Views that cannot open yet, each with the one line that says why. */
  disabled?: Partial<Record<View, string>>;
  onToggle: (view: View) => void;
  onHeight: (height: number) => void;
  /** The edge is held: the room shields the panes (base.css). */
  onDragging: (dragging: boolean) => void;
  children: ReactNode;
}) {
  const [dragging, setDragging] = useState(false);
  const start = useRef<{ y: number; height: number } | null>(null);

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      event.currentTarget.setPointerCapture(event.pointerId);
      start.current = { y: event.clientY, height };
      setDragging(true);
      onDragging(true);
    },
    [height, onDragging],
  );
  const onPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (start.current) onHeight(start.current.height - (event.clientY - start.current.y));
    },
    [onHeight],
  );
  const onPointerUp = useCallback(() => {
    if (!start.current) return;
    start.current = null;
    setDragging(false);
    onDragging(false);
  }, [onDragging]);

  return (
    <section className="flex min-h-0 flex-none flex-col bg-sunken data-[drawer=closed]:border-t data-[drawer=closed]:border-line-weak" data-drawer={view ?? "closed"} aria-label="Debug">
      {view ? (
        <div
          className="drawerEdge"
          role="separator"
          aria-orientation="horizontal"
          aria-label="Resize the debug drawer"
          data-dragging={dragging}
          data-splitter="drawer"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        />
      ) : null}
      <nav role="tablist" aria-label="Debug views" className="flex h-[calc(var(--h-header)+var(--pad-3))] flex-none items-stretch gap-1 border-b border-line-weak px-4" data-drawer-tabs="true">
        {VIEWS.map((item) => {
          const why = disabled[item.id];
          return (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={item.id === view}
              disabled={Boolean(why)}
              title={why ?? item.tip}
              className="drawerTab"
              data-active={item.id === view}
              data-panel-tab={item.id}
              onClick={() => onToggle(item.id)}
            >
              {item.name}
            </button>
          );
        })}
        <span className="ml-auto flex items-center text-tag text-label" title="Open or close the debug drawer">
          ⌘.
        </span>
      </nav>
      {view ? (
        <div className="min-h-0 overflow-hidden pt-2" style={{ height }} data-pane="drawer">
          <InDrawer.Provider value={true}>{children}</InDrawer.Provider>
        </div>
      ) : null}
    </section>
  );
}
