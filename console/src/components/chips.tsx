"use client";

import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Icon } from "./chrome/icon";
import { sessionTitle } from "./panels/sessions";
import { ago } from "@/lib/format";
import type { SessionRow } from "@/lib/types";

/* How many conversations a page of the strip holds. */
const PAGE = 6;
/* The move: the leading blob gets there first, the trailing one follows (jukebox's stretch). */
const DURATION = 560;

const cubic = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);

type Edges = { l: number; r: number };

/**
 * The conversations as a centered row of tabs over the debug tabs, a page at
 * a time, the dots under it showing which page and turning it; "All" finds
 * any by title and "+" starts one, lit until its first message makes it a
 * tab. The current tab sits on a blob: choosing
 * another sends a second blob ahead to it, the two neck and part like
 * metaballs, and the one left behind catches up and merges.
 */
export function Chips({
  sessions: listed,
  selected,
  details,
  onSelect,
  onNew,
}: {
  sessions: SessionRow[];
  selected: string | null;
  details: Map<string, { lanes: number; firstMessage: string | null }>;
  onSelect: (id: string) => void;
  onNew: () => void;
}) {
  const goo = useId();
  const rowRef = useRef<HTMLDivElement>(null);
  const leadRef = useRef<SVGRectElement>(null);
  const trailRef = useRef<SVGRectElement>(null);
  const neckRef = useRef<SVGRectElement>(null);
  const blob = useRef({ from: { l: 0, r: 0 }, to: { l: 0, r: 0 }, at: 0, shown: false });

  /* Newest first by when each began, so sending a message or opening one moves nothing. */
  const sessions = useMemo(() => [...listed].sort((a, b) => b.createdAt - a.createdAt), [listed]);
  /* The open conversation before its first message: no tab yet, and "+" stays lit. */
  const pending = !sessions.some((session) => session.sessionId === selected);

  /* The page shown: the current conversation's, until a dot turns it. */
  const pages = Math.max(1, Math.ceil(sessions.length / PAGE));
  const current = sessions.findIndex((session) => session.sessionId === selected);
  const [turned, setTurned] = useState<{ page: number; selected: string | null } | null>(null);
  const page = Math.min(pages - 1, turned && turned.selected === selected ? turned.page : Math.max(0, Math.floor(current / PAGE)));
  const shown = sessions.slice(page * PAGE, page * PAGE + PAGE);

  const title = (session: SessionRow) => sessionTitle(session, details.get(session.sessionId)?.firstMessage);

  /* A new selection or page: the blob leaves from where it is drawn now. */
  useLayoutEffect(() => {
    const row = rowRef.current;
    const chip = row?.querySelector<HTMLElement>(`[data-session="${selected}"]`);
    if (!row || !chip) {
      blob.current.shown = false;
      draw(blob.current, leadRef.current, trailRef.current, neckRef.current, 0);
      return;
    }

    const to = { l: chip.offsetLeft, r: chip.offsetLeft + chip.offsetWidth };
    const now = performance.now();
    const still = !blob.current.shown || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const at = blob.current.shown ? edges(blob.current, now) : { lead: to, trail: to };
    blob.current = { from: still ? to : at.lead, to, at: now, shown: true };

    let frame = requestAnimationFrame(function step(time) {
      draw(blob.current, leadRef.current, trailRef.current, neckRef.current, time);
      if (time - blob.current.at < DURATION) frame = requestAnimationFrame(step);
    });
    return () => cancelAnimationFrame(frame);
  }, [selected, page, sessions]);

  return (
    <div className="flex flex-none flex-col items-center gap-2 px-4 pt-3 pb-3" data-panel="sessions">
      <div className="flex max-w-full min-w-0 items-center gap-1">
        <div ref={rowRef} className="relative flex min-w-0 items-center gap-1" role="tablist" aria-label="Conversations">
          <svg className="pointer-events-none absolute inset-0 size-full overflow-visible fill-header-quaternary" aria-hidden="true">
            <filter id={goo} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="5" />
              <feColorMatrix values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 20 -9" />
            </filter>
            <g filter={`url(#${goo})`}>
              <rect ref={trailRef} />
              <rect ref={neckRef} />
              <rect ref={leadRef} />
            </g>
          </svg>
          {shown.map((session) => (
            <button
              key={session.sessionId}
              type="button"
              role="tab"
              aria-selected={session.sessionId === selected}
              className="chip relative min-w-0"
              title={`${title(session)} · ${ago(session.lastActive)}`}
              data-session={session.sessionId}
              data-selected={session.sessionId === selected}
              data-titled={Boolean(session.attributes?.title) || undefined}
              data-delegated={(details.get(session.sessionId)?.lanes ?? 0) > 1 || undefined}
              onClick={() => onSelect(session.sessionId)}
            >
              <span className="max-w-[var(--w-chip)] truncate" data-session-title="true">
                {title(session)}
              </span>
            </button>
          ))}
        </div>
        {sessions.length > PAGE ? <All sessions={sessions} selected={selected} title={title} onSelect={onSelect} /> : null}
        <button type="button" className="chip chipIcon" title="New conversation (Cmd+K)" aria-label="New conversation" data-action="new-conversation" data-selected={pending} onClick={onNew}>
          <Icon name="plus" size={12} />
        </button>
      </div>

      {pages > 1 ? (
        <div className="flex items-center gap-1.5" role="group" aria-label="Pages">
          {Array.from({ length: pages }, (_, index) => (
            <button
              key={index}
              type="button"
              className="pageDot"
              aria-label={`Page ${index + 1}`}
              aria-current={index === page || undefined}
              data-current={index === page}
              onClick={() => setTurned({ page: index, selected })}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

/** Every conversation in a list filtered as you type; Enter opens the first match. */
function All({
  sessions,
  selected,
  title,
  onSelect,
}: {
  sessions: SessionRow[];
  selected: string | null;
  title: (session: SessionRow) => string;
  onSelect: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  const matches = sessions.filter((session) => title(session).toLowerCase().includes(query.trim().toLowerCase()));

  const close = () => {
    setOpen(false);
    setQuery("");
  };
  const choose = (id: string) => {
    onSelect(id);
    close();
  };

  /* The list closes on a click anywhere else. */
  useEffect(() => {
    if (!open) return;
    const onDown = (event: PointerEvent) => {
      if (!ref.current?.contains(event.target as Node)) close();
    };
    window.addEventListener("pointerdown", onDown);
    return () => window.removeEventListener("pointerdown", onDown);
  }, [open]);

  return (
    <div ref={ref} className="relative flex flex-none items-center">
      <button type="button" className="chip" aria-expanded={open} data-action="all-conversations" onClick={() => (open ? close() : setOpen(true))}>
        All {sessions.length}
      </button>
      {open ? (
        <div className="absolute right-0 bottom-full z-20 mb-2 grid w-[var(--w-info)] rounded-blender bg-raised shadow-[var(--shadow)]" data-all="true">
          <input
            autoFocus
            className="h-row border-b border-line-weak bg-transparent px-3 text-ink outline-none placeholder:text-label"
            placeholder="Find a conversation"
            aria-label="Find a conversation"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") close();
              if (event.key === "Enter" && matches[0]) choose(matches[0].sessionId);
            }}
          />
          <div className="grid max-h-[calc(var(--h-row)*12)] overflow-y-auto py-1">
            {matches.map((session) => (
              <button
                key={session.sessionId}
                type="button"
                className="flex min-h-row items-center gap-3 px-3 text-left text-ink hover:bg-header-quaternary data-[selected=true]:text-strong"
                data-selected={session.sessionId === selected}
                onClick={() => choose(session.sessionId)}
              >
                <span className="min-w-0 flex-1 truncate">{title(session)}</span>
                <span className="flex-none text-label tabular-nums">{ago(session.lastActive)}</span>
              </button>
            ))}
            {matches.length === 0 ? <span className="px-3 py-1 text-label">No match</span> : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

/** Where the two blobs are now: the leading one eased ahead, the trailing one behind. */
function edges(blob: { from: Edges; to: Edges; at: number }, now: number): { lead: Edges; trail: Edges } {
  const t = Math.min(1, (now - blob.at) / DURATION);
  const toward = (k: number) => ({ l: blob.from.l + (blob.to.l - blob.from.l) * k, r: blob.from.r + (blob.to.r - blob.from.r) * k });
  return { lead: toward(cubic(Math.min(t * 1.4, 1))), trail: toward(cubic(Math.max((t - 0.15) * 1.18, 0))) };
}

/**
 * Both blobs as pills the row's height, and the neck between their middles,
 * thinner the further apart they are; the filter melts the three into one.
 */
function draw(blob: { from: Edges; to: Edges; at: number; shown: boolean }, lead: SVGRectElement | null, trail: SVGRectElement | null, neck: SVGRectElement | null, now: number) {
  const height = lead?.ownerSVGElement?.clientHeight ?? 0;
  const at = edges(blob, now);
  for (const [rect, e] of [
    [lead, at.lead],
    [trail, at.trail],
  ] as const) {
    if (!rect) continue;
    rect.setAttribute("x", String(e.l));
    rect.setAttribute("width", String(blob.shown ? Math.max(0, e.r - e.l) : 0));
    rect.setAttribute("height", String(height));
    rect.setAttribute("rx", String(height / 2));
  }

  if (!neck) return;
  const a = (at.trail.l + at.trail.r) / 2;
  const b = (at.lead.l + at.lead.r) / 2;
  const apart = Math.abs(b - a) / Math.max(blob.to.r - blob.to.l, 1);
  const thick = blob.shown && apart > 0 ? height * Math.max(0.4, 0.75 - 0.15 * apart) : 0;
  neck.setAttribute("x", String(Math.min(a, b)));
  neck.setAttribute("width", String(Math.abs(b - a)));
  neck.setAttribute("y", String((height - thick) / 2));
  neck.setAttribute("height", String(thick));
  neck.setAttribute("rx", String(thick / 2));
}
