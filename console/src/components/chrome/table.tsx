"use client";

import { useCallback, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { Icon } from "./icon";

/**
 * One column: what it shows, what it is called, how wide it starts, and the
 * text it is filtered and sorted by (the cell's own text when the cell is a
 * string or number).
 */
export type Column<T> = {
  id: string;
  label: string;
  /** What the column means, for the header's tooltip. */
  tip?: string;
  /** The starting width in px; the last column takes the rest. */
  width: number;
  align?: "right";
  mono?: boolean;
  cell: (row: T) => ReactNode;
  text?: (row: T) => string;
  sort?: (a: T, b: T) => number;
};

type Sort = { id: string; dir: 1 | -1 } | null;

/** A row's place on the rail that draws a span of rows as one block: where it starts, runs, ends. */
export type RailPlace = "start" | "mid" | "end" | "single";
export type Rail = { place: RailPlace; tone?: "attention" } | null;

const MIN_COL = 32;

function textOf<T>(column: Column<T>, row: T): string {
  if (column.text) return column.text(row);
  const cell = column.cell(row);
  return typeof cell === "string" || typeof cell === "number" ? String(cell) : "";
}

/**
 * The side panel's data table, at Blender's density: a header row in the
 * quaternary grey that stays put, 20px rows filled on hover, mono for ids,
 * times and numbers, numbers on the right. A filter field above it matches
 * a substring in any cell; a click on a header sorts by it, again to
 * reverse, a third time to let go; the boundary between two headers drags
 * the width. The selected row takes the accent. A row with a fold has a
 * chevron in its first column, so the affordance is visible before the
 * pointer arrives; clicking the row opens the fold under it (one open at a
 * time) unless the row has a selection of its own, in which case the
 * chevron opens the fold.
 */
export function DataTable<T>({
  id,
  rows,
  columns,
  rowKey,
  rowData,
  selected,
  onSelect,
  fold,
  rail,
  group,
  empty,
  now,
}: {
  id: string;
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => string;
  /** `data-*` on the row, for tests and state styling. */
  rowData?: (row: T) => Record<string, string | number | boolean | undefined>;
  selected?: string | null;
  onSelect?: (row: T) => void;
  /** What a row folds open under it; null for a row with nothing to open. */
  fold?: (row: T) => ReactNode;
  /** A rail on the left that brackets rows into blocks (a run from its start to its end). */
  rail?: (row: T) => Rail;
  /** Rows partitioned under group headers, in the order the groups first appear. */
  group?: (row: T) => string;
  empty: string;
  /** The clock the cells read, so the table re-renders with it. */
  now?: number;
}) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<Sort>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [widths, setWidths] = useState<number[]>(() => columns.map((column) => column.width));
  const drag = useRef<{ index: number; x: number; width: number } | null>(null);
  /* A drag ends with a click on the header under it; that click is not a sort. */
  const justDragged = useRef(false);

  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    let out = needle ? rows.filter((row) => columns.some((column) => textOf(column, row).toLowerCase().includes(needle))) : rows.slice();
    if (sort) {
      const column = columns.find((c) => c.id === sort.id);
      if (column) {
        const compare = column.sort ?? ((a: T, b: T) => textOf(column, a).localeCompare(textOf(column, b), undefined, { numeric: true }));
        out = out.sort((a, b) => compare(a, b) * sort.dir);
      }
    }
    if (group) {
      /* Groups keep the order they first appear in; the sort holds inside each. */
      const order: string[] = [];
      const buckets = new Map<string, T[]>();
      for (const row of out) {
        const name = group(row);
        if (!buckets.has(name)) {
          order.push(name);
          buckets.set(name, []);
        }
        buckets.get(name)!.push(row);
      }
      out = order.flatMap((name) => buckets.get(name)!);
    }
    return out;
    /* `now` is read by the cells, not here; it is in the key so the clock reaches them. */
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [columns, group, query, rows, sort, now]);

  const onHeaderClick = (column: Column<T>) => {
    if (justDragged.current) {
      justDragged.current = false;
      return;
    }
    setSort((current) => (current?.id !== column.id ? { id: column.id, dir: 1 } : current.dir === 1 ? { id: column.id, dir: -1 } : null));
  };

  const onGripDown = useCallback(
    (index: number) => (event: ReactPointerEvent<HTMLSpanElement>) => {
      event.stopPropagation();
      event.currentTarget.setPointerCapture(event.pointerId);
      drag.current = { index, x: event.clientX, width: widths[index] };
    },
    [widths],
  );
  const onGripMove = useCallback((event: ReactPointerEvent<HTMLSpanElement>) => {
    const d = drag.current;
    if (!d) return;
    const width = Math.max(MIN_COL, d.width + event.clientX - d.x);
    setWidths((current) => current.map((w, i) => (i === d.index ? width : w)));
  }, []);
  const onGripUp = useCallback(() => {
    justDragged.current = drag.current !== null;
    drag.current = null;
  }, []);

  const hasFold = Boolean(fold);
  const hasRail = Boolean(rail);
  /* The rail and the chevron gutters are the compact row height and the row height: tokens, not numbers. */
  const template = `${hasRail ? "var(--pad-4) " : ""}${hasFold ? "var(--h-row) " : ""}${widths
    .map((width, index) => (index === widths.length - 1 ? `minmax(${width}px, 1fr)` : `${width}px`))
    .join(" ")}`;
  const minWidth = `calc(${widths.reduce((sum, width) => sum + width, 0)}px${hasFold ? " + var(--h-row)" : ""}${hasRail ? " + var(--pad-4)" : ""})`;

  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_minmax(0,1fr)]" data-table={id}>
      <div className="flex items-center px-3 pb-2">
        <input
          className="field h-row-sm w-full min-w-0"
          value={query}
          placeholder="Filter"
          aria-label={`Filter ${id}`}
          data-table-filter={id}
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>
      <div className="min-h-0 overflow-auto">
        <div role="table" className="grid content-start" style={{ gridTemplateColumns: template, minWidth }}>
          <div role="row" className="sticky top-0 z-10 col-span-full grid bg-header-quaternary" style={{ gridTemplateColumns: template }}>
            {hasRail ? <span /> : null}
            {hasFold ? <span /> : null}
            {columns.map((column, index) => {
              const sorted = sort?.id === column.id ? sort.dir : 0;
              return (
                <button
                  key={column.id}
                  type="button"
                  role="columnheader"
                  aria-sort={sorted === 1 ? "ascending" : sorted === -1 ? "descending" : "none"}
                  title={column.tip ?? column.label}
                  className="relative flex h-row-sm min-w-0 items-center gap-1 border-0 bg-transparent px-2 text-left window-label text-label hover:text-ink data-[align=right]:flex-row-reverse"
                  data-align={column.align}
                  data-column={column.id}
                  onClick={() => onHeaderClick(column)}
                >
                  <span className="truncate">{column.label}</span>
                  {sorted ? <Icon name={sorted === 1 ? "disclosureUp" : "disclosureDown"} size={10} className="flex-none" /> : null}
                  {index < columns.length - 1 ? (
                    <span
                      role="separator"
                      aria-orientation="vertical"
                      aria-label={`Resize ${column.label}`}
                      className="absolute top-0 right-0 h-full w-2 cursor-col-resize"
                      data-grip={column.id}
                      onPointerDown={onGripDown(index)}
                      onPointerMove={onGripMove}
                      onPointerUp={onGripUp}
                      onPointerCancel={onGripUp}
                    />
                  ) : null}
                </button>
              );
            })}
          </div>
          {shown.length === 0 ? <p className="col-span-full m-0 flex min-h-row items-center px-3 py-1 text-label">{query ? "Nothing matches." : empty}</p> : null}
          {shown.map((row, index) => {
            const key = rowKey(row);
            const folded = fold ? fold(row) : null;
            const place = rail ? rail(row) : null;
            const heading = group && (index === 0 || group(shown[index - 1]) !== group(row)) ? group(row) : null;
            const isOpen = open === key && folded != null;
            const isSelected = selected === key;
            const attrs: Record<string, string> = {};
            for (const [name, value] of Object.entries(rowData?.(row) ?? {})) if (value !== undefined) attrs[`data-${name}`] = String(value);
            const toggle = () => setOpen((current) => (current === key ? null : key));
            /* The row's own gesture: select when it can be selected, else open its fold. */
            const activate = () => (onSelect ? onSelect(row) : folded != null && toggle());
            return (
              <div key={key} className="contents">
                {heading !== null ? (
                  <div role="row" className="col-span-full flex h-row-sm items-center bg-sunken px-3 window-label text-label" data-group={heading}>
                    {heading}
                  </div>
                ) : null}
                <div
                  role="row"
                  tabIndex={0}
                  {...attrs}
                  data-row={key}
                  data-open={isOpen || undefined}
                  className="col-span-full grid h-row items-center text-ink transition-colors duration-fast ease-blender hover:bg-header-quaternary data-[selected=true]:bg-raised data-[selected=true]:shadow-[inset_var(--pad-1)_0_0_var(--accent)] data-[selected=true]:hover:bg-raised"
                  style={{ gridTemplateColumns: template }}
                  data-selected={isSelected}
                  onClick={activate}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      activate();
                    }
                  }}
                >
                  {hasRail ? (
                    <span className="relative h-full" data-rail={place?.place} data-rail-tone={place?.tone} aria-hidden="true">
                      {place ? (
                        <>
                          <span
                            className="absolute left-1/2 w-px bg-line data-[tone=attention]:bg-attention data-[place=start]:top-1/2 data-[place=start]:bottom-0 data-[place=mid]:top-0 data-[place=mid]:bottom-0 data-[place=end]:top-0 data-[place=end]:bottom-1/2 data-[place=single]:hidden"
                            data-place={place.place}
                            data-tone={place.tone}
                          />
                          {place.place !== "mid" ? (
                            <span
                              className="absolute top-1/2 left-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-label data-[tone=attention]:bg-attention"
                              data-tone={place.tone}
                            />
                          ) : null}
                        </>
                      ) : null}
                    </span>
                  ) : null}
                  {hasFold ? (
                    <button
                      type="button"
                      className="grid h-row w-full place-items-center border-0 bg-transparent text-label hover:text-ink disabled:opacity-30"
                      aria-label={isOpen ? "Close" : "Open"}
                      aria-expanded={isOpen}
                      disabled={folded == null}
                      data-fold-toggle={key}
                      onClick={(event) => {
                        event.stopPropagation();
                        toggle();
                      }}
                    >
                      <Icon name={isOpen ? "disclosureDown" : "disclosureRight"} size={12} />
                    </button>
                  ) : null}
                  {columns.map((column) => (
                    <span
                      key={column.id}
                      role="cell"
                      className="min-w-0 truncate px-2 data-[align=right]:text-right data-[mono=true]:font-mono data-[mono=true]:text-micro"
                      data-align={column.align}
                      data-mono={column.mono || undefined}
                      data-cell={column.id}
                    >
                      {column.cell(row)}
                    </span>
                  ))}
                </div>
                {isOpen ? (
                  <div className="col-span-full bg-header-quaternary px-3 py-2" data-fold-of={key}>
                    {folded}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
