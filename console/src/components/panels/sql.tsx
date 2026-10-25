"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { autocompletion, closeBrackets, closeBracketsKeymap, closeCompletion, completionKeymap } from "@codemirror/autocomplete";
import { bracketMatching, HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { SQLite, sql as sqlLanguage } from "@codemirror/lang-sql";
import { Compartment, EditorState, Prec } from "@codemirror/state";
import { EditorView, highlightActiveLine, highlightActiveLineGutter, keymap, lineNumbers, placeholder } from "@codemirror/view";
import { highlightCode, tags } from "@lezer/highlight";
import { Panel } from "../chrome/panel";
import { DataTable, type Column } from "../chrome/table";
import { getDb } from "@/lib/api";
import { Wsdb, type QueryResult, type Status } from "@/lib/wsdb";
import { pretty } from "@/lib/format";

type Field = { name: string; type: string; notNull: boolean; byDefault: string | null; key: number };
/** A table as `sqlite_master` and `pragma_table_info` describe it, and how many rows it holds. */
type Table = { name: string; columns: Field[]; sql: string; indexes: { name: string; sql: string }[]; count: number | null };
type Row = { key: string; cells: unknown[] };

const QUERY_KEY = "console.sql.query";
const LIMIT = 100;
/** The most rows a query brings back; past it the query is cancelled. */
const MAX_ROWS = 500;
const COLUMNS_SQL =
  'SELECT m.name, p.name, p.type, p."notnull", p.dflt_value, p.pk FROM sqlite_master m JOIN pragma_table_info(m.name) p ' +
  "WHERE m.type = 'table' AND m.name NOT LIKE 'sqlite_%' ORDER BY m.name, p.cid";
const MASTER_SQL = "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL";
const countSql = (names: string[]) => names.map((name) => `SELECT '${name}', COUNT(*) FROM "${name}"`).join(" UNION ALL ");

/** The tables, read in three statements: their columns, their definitions and indexes, their sizes. */
async function readSchema(db: Wsdb): Promise<Table[]> {
  const [columns, master] = await Promise.all([db.query(COLUMNS_SQL, 10_000), db.query(MASTER_SQL, 10_000)]);
  const found = new Map<string, Table>();
  for (const [table, name, type, notNull, byDefault, key] of columns.rows as [string, string, string, number, string | null, number][]) {
    if (!found.has(table)) found.set(table, { name: table, columns: [], sql: "", indexes: [], count: null });
    found.get(table)!.columns.push({ name, type: type.toLowerCase(), notNull: Boolean(notNull), byDefault, key });
  }
  for (const [type, name, table, sql] of master.rows as [string, string, string, string][]) {
    const entry = found.get(table);
    if (entry && type === "table") entry.sql = sql;
    else if (entry && type === "index") entry.indexes.push({ name, sql });
  }
  const tables = [...found.values()];
  if (tables.length) {
    const counts = await db.query(countSql(tables.map((t) => t.name)), tables.length);
    for (const [name, count] of counts.rows as [string, number][]) found.get(name)!.count = count;
  }
  return tables;
}

/** The table a statement reads from, when it names one after `FROM`. */
const fromTable = (sql: string) => /\bfrom\s+["`]?(\w+)/i.exec(sql)?.[1] ?? null;

/** What a value is, as SQLite calls its storage class. */
function storageClass(value: unknown) {
  if (typeof value === "number") return Number.isInteger(value) ? "integer" : "real";
  if (typeof value === "string") return "text";
  return isBlob(value) ? "blob" : null;
}

/** A table's rows as the editor writes them: JSONB columns through `json()`, so they read as text. */
function browse(table: Table) {
  const columns = table.columns.map((c) => (c.type === "jsonb" ? `json(${c.name}) AS ${c.name}` : c.name));
  return `SELECT ${columns.join(", ")}\nFROM ${table.name}\nLIMIT ${LIMIT}`;
}

const WRITES = new Set(["INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER", "ATTACH", "DETACH", "VACUUM", "REINDEX", "ANALYZE", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE"]);

/**
 * One statement that only reads: `SELECT`, `WITH`, `VALUES`, `EXPLAIN`, or a
 * `PRAGMA` that sets nothing, and no writing keyword outside its strings and
 * comments. The token can write, so this is the tab's only guard.
 */
export function readOnly(sql: string) {
  const code = sql
    .replace(/'[^']*'|"[^"]*"|`[^`]*`/g, " ")
    .replace(/--[^\n]*/g, " ")
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .trim()
    .replace(/;+\s*$/, "")
    .toUpperCase();
  if (code.includes(";")) return false;
  const words = code.split(/[^A-Z0-9_]+/).filter(Boolean);
  const reads = ["SELECT", "WITH", "VALUES", "EXPLAIN"].includes(words[0]) || (words[0] === "PRAGMA" && !code.includes("="));
  return reads && !words.some((word) => WRITES.has(word));
}

/** A cell parsed back to JSON when it is a JSON object or array as text. */
function parsed(value: unknown): unknown {
  if (typeof value !== "string" || !/^\s*[[{]/.test(value)) return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

const isBlob = (value: unknown): value is { blob: number } =>
  typeof value === "object" && value !== null && !Array.isArray(value) && Object.keys(value).length === 1 && "blob" in value;

function Cell({ value }: { value: unknown }) {
  if (value === null) return <span className="text-label italic">NULL</span>;
  if (isBlob(value)) return <span className="text-label" title="Binary; select json(column) to read JSONB">blob · {value.blob} B</span>;
  return <span className="truncate">{String(value)}</span>;
}

const theme = EditorView.theme({
  "&": { height: "100%", backgroundColor: "var(--room)", color: "var(--text-strong)", fontSize: "var(--fs-ui)" },
  "&.cm-focused": { outline: "none" },
  ".cm-scroller": { fontFamily: "var(--font-mono)", lineHeight: "var(--lh-ui)", fontVariantLigatures: "none" },
  ".cm-content": { padding: "var(--pad-3) 0", caretColor: "var(--text-strong)" },
  ".cm-line": { padding: "0 var(--pad-3)" },
  ".cm-gutters": { backgroundColor: "var(--room)", color: "var(--code-gutter)", border: "none", paddingLeft: "var(--pad-2)" },
  ".cm-activeLine": { backgroundColor: "var(--header-quaternary)" },
  ".cm-activeLineGutter": { backgroundColor: "var(--header-quaternary)", color: "var(--text)" },
  "&.cm-focused .cm-matchingBracket": { backgroundColor: "var(--control)", color: "var(--text-strong)" },
  ".cm-cursor": { borderLeftColor: "var(--text-strong)" },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": { backgroundColor: "var(--control)" },
  ".cm-placeholder": { color: "var(--text-label)" },
  ".cm-tooltip": { backgroundColor: "var(--raised)", border: "var(--line-width) solid var(--line-weak)", borderRadius: "var(--radius)", boxShadow: "var(--shadow)" },
  ".cm-tooltip-autocomplete > ul": { fontFamily: "var(--font-mono)", fontSize: "var(--fs-micro)" },
  ".cm-tooltip-autocomplete > ul > li[aria-selected]": { backgroundColor: "var(--accent)", color: "var(--accent-text)" },
  ".cm-completionDetail": { color: "var(--text-label)", fontStyle: "normal" },
});

const highlight = HighlightStyle.define([
  { tag: tags.keyword, color: "var(--code-keyword)" },
  { tag: [tags.string, tags.special(tags.string)], color: "var(--code-string)" },
  { tag: [tags.number, tags.bool, tags.null], color: "var(--code-number)" },
  { tag: [tags.typeName, tags.standard(tags.name)], color: "var(--code-type)" },
  { tag: tags.comment, color: "var(--code-comment)", fontStyle: "italic" },
  { tag: [tags.operator, tags.punctuation], color: "var(--text)" },
]);

/** SQL as the editor colours it, for the definitions: spans in the highlight's classes. */
function Highlighted({ code }: { code: string }) {
  const out: ReactNode[] = [];
  highlightCode(
    code,
    sqlLanguage({ dialect: SQLite }).language.parser.parse(code),
    highlight,
    (text, classes) => out.push(classes ? <span key={out.length} className={classes}>{text}</span> : text),
    () => out.push("\n"),
  );
  return <>{out}</>;
}

/** One table's columns (type, null, default, key), its indexes and the statement that made it. */
function Structure({ table }: { table: Table }) {
  return (
    <div className="grid min-h-0 content-start gap-4 overflow-auto px-4 pb-4" data-sql-structure={table.name}>
      <div className="grid w-max min-w-[var(--w-info)] grid-cols-[repeat(5,auto)] font-mono text-micro" role="table" aria-label={`${table.name}'s columns`}>
        <div role="row" className="col-span-full grid grid-cols-subgrid bg-header-quaternary text-label">
          {["column", "type", "null", "default", "key"].map((name) => (
            <span key={name} role="columnheader" className="flex h-row-sm items-center px-2">
              {name}
            </span>
          ))}
        </div>
        {table.columns.map((column) => (
          <div key={column.name} role="row" className="col-span-full grid h-row grid-cols-subgrid items-center hover:bg-header-quaternary" data-sql-field={column.name}>
            <span className="truncate px-2 text-strong">{column.name}</span>
            <span className="px-2 text-[color:var(--code-type)]">{column.type}</span>
            <span className="px-2 text-label">{column.notNull ? "not null" : "null"}</span>
            <span className="truncate px-2 text-ink">{column.byDefault ?? ""}</span>
            <span className="px-2 text-[color:var(--code-keyword)]">{column.key ? "primary" : ""}</span>
          </div>
        ))}
      </div>
      {table.indexes.length ? (
        <section className="grid gap-1">
          <h3 className="m-0 window-label text-label">Indexes</h3>
          {table.indexes.map((index) => (
            <pre key={index.name} className="m-0 font-mono text-micro whitespace-pre-wrap text-strong">
              <Highlighted code={index.sql} />
            </pre>
          ))}
        </section>
      ) : null}
      <section className="grid gap-1">
        <h3 className="m-0 window-label text-label">Definition</h3>
        <pre className="m-0 rounded-blender bg-room p-3 font-mono text-micro whitespace-pre-wrap text-strong" data-sql-definition="true">
          <Highlighted code={table.sql} />
        </pre>
      </section>
    </div>
  );
}

/** The socket's state: a dot, and what it means; the database id and commit on hover. */
function Connection({ status }: { status: Status }) {
  const text =
    status.state === "open"
      ? `Connected${status.latency === null ? "" : ` · ${status.latency} ms`}`
      : status.state === "connecting"
        ? "Connecting"
        : `Disconnected: ${status.error} Retrying in ${Math.ceil(status.retryIn / 1000)} s.`;
  return (
    <span className="flex min-w-0 items-center gap-2" title={status.state === "open" ? `wsdb, at commit ${status.tip}` : undefined} data-sql-connection={status.state}>
      <span className="size-2 flex-none rounded-full bg-label data-[state=open]:bg-accent data-[state=closed]:bg-failed-text data-[state=connecting]:bg-attention" data-state={status.state} />
      <span className="truncate">{text}</span>
    </span>
  );
}

/**
 * The conversation's database in SQL, over its own wsdb socket, whose state
 * the header shows: its tables down the side with their sizes, a click
 * browses one; a read-only statement in the editor, completed from the
 * schema, runs with Cmd+Enter; the rows come back in the data table, each
 * column with its type, a row folding open to its cells as JSON. Structure
 * shows the picked table's columns, indexes and definition.
 */
export function SqlPanel({ sessionId }: { sessionId: string | null }) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const schemaSlot = useRef(new Compartment());
  const [tables, setTables] = useState<Table[]>([]);
  /* The table picked on the side, and whether its rows or its structure show. */
  const [picked, setPicked] = useState<string | null>(null);
  const [mode, setMode] = useState<"rows" | "structure">("rows");
  const [result, setResult] = useState<(QueryResult & { run: number; statement: string }) | null>(null);
  const [status, setStatus] = useState<Status>({ state: "connecting" });
  const client = useRef<Wsdb | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const runs = useRef(0);

  const run = useCallback(
    async (text?: string) => {
      /* What is selected, else the whole editor. */
      const state = view.current?.state;
      const selected = state ? state.sliceDoc(state.selection.main.from, state.selection.main.to) : "";
      const statement = (text ?? (selected || state?.doc.toString()) ?? "").trim();
      if (!client.current || !statement) return;
      if (!readOnly(statement)) return setError("Only a single read statement runs here.");
      setRunning(true);
      try {
        setResult({ ...(await client.current.query(statement, MAX_ROWS)), run: ++runs.current, statement });
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setRunning(false);
      }
    },
    [],
  );
  const runRef = useRef(run);
  useEffect(() => {
    runRef.current = run;
  }, [run]);

  /* The editor, once; Cmd+Enter runs what it holds, which is kept for the next visit. */
  useEffect(() => {
    if (!host.current) return;
    let saved = "";
    try {
      saved = window.localStorage.getItem(QUERY_KEY) ?? "";
    } catch {}
    const editor = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: saved,
        extensions: [
          lineNumbers(),
          highlightActiveLine(),
          highlightActiveLineGutter(),
          bracketMatching(),
          closeBrackets(),
          history(),
          Prec.highest(keymap.of([{ key: "Mod-Enter", run: (editor) => (closeCompletion(editor), void runRef.current(), true) }])),
          keymap.of([...closeBracketsKeymap, ...defaultKeymap, ...historyKeymap, ...completionKeymap]),
          autocompletion(),
          schemaSlot.current.of(sqlLanguage({ dialect: SQLite, upperCaseKeywords: true })),
          syntaxHighlighting(highlight),
          placeholder("SELECT * FROM lanes"),
          EditorView.lineWrapping,
          EditorView.updateListener.of((update) => {
            if (!update.docChanged) return;
            try {
              window.localStorage.setItem(QUERY_KEY, update.state.doc.toString());
            } catch {}
          }),
          theme,
        ],
      }),
    });
    view.current = editor;
    return () => {
      editor.destroy();
      view.current = null;
    };
  }, []);

  /* The conversation's database over its own socket, and its schema, for the side and for
     completion; the room remounts the panel for another conversation. */
  useEffect(() => {
    if (!sessionId) return;
    const db = new Wsdb(() => getDb(sessionId), setStatus);
    client.current = db;
    readSchema(db)
      .then((next) => {
        setTables(next);
        view.current?.dispatch({
          effects: schemaSlot.current.reconfigure(
            sqlLanguage({ dialect: SQLite, upperCaseKeywords: true, schema: Object.fromEntries(next.map((t) => [t.name, t.columns.map((c) => c.name)])) }),
          ),
        });
      })
      .catch((e) => client.current === db && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      db.close();
      if (client.current === db) client.current = null;
    };
  }, [sessionId]);

  /* A table on the side: its rows in the editor and the results, its structure a tab away. */
  const pick = (table: Table) => {
    const text = browse(table);
    view.current?.dispatch({ changes: { from: 0, to: view.current.state.doc.length, insert: text } });
    setPicked(table.name);
    void run(text);
  };

  const rows = useMemo<Row[]>(() => (result ? result.rows.map((cells, index) => ({ key: String(index), cells })) : []), [result]);
  const columns = useMemo<Column<Row>[]>(() => {
    /* A column's type: the table's declared one when the statement reads a table that has it,
       else the storage class of its first value. */
    const source = tables.find((t) => t.name === fromTable(result?.statement ?? ""));
    return (result?.columns ?? []).map((name, index) => {
      const note = source?.columns.find((c) => c.name === name)?.type ?? storageClass(result?.rows.find((r) => r[index] !== null)?.[index]) ?? undefined;
      return {
        id: `${index}`,
        label: name,
        note,
        /* Wide enough for the name and its type, at the micro mono's advance. */
        width: Math.max(120, (name.length + (note?.length ?? 0) + 4) * 7),
        mono: true,
        cell: (row) => <Cell value={row.cells[index]} />,
        text: (row) => {
          const value = row.cells[index];
          return value === null ? "" : isBlob(value) ? "" : String(value);
        },
        sort: (a, b) => {
          const [x, y] = [a.cells[index], b.cells[index]];
          return typeof x === "number" && typeof y === "number" ? x - y : String(x ?? "").localeCompare(String(y ?? ""), undefined, { numeric: true });
        },
      };
    });
  }, [result, tables]);
  const table = tables.find((t) => t.name === picked) ?? null;

  const summary = !error && result ? `${result.rows.length}${result.truncated ? "+" : ""} ${result.rows.length === 1 && !result.truncated ? "row" : "rows"} · ${result.ms} ms` : null;

  return (
    <Panel
      id="sql"
      title="SQL"
      meta={sessionId ? <Connection status={status} /> : undefined}
      actions={
        <>
          {summary ? <span className="text-label tabular-nums" data-sql-summary="true">{summary}</span> : null}
          <button type="button" className="btn" data-primary="true" disabled={status.state !== "open" || running} title="Run the selection, or everything (Cmd+Enter)" data-action="run-sql" onClick={() => void run()}>
            Run <span className="opacity-70">⌘↵</span>
          </button>
        </>
      }
    >
      <div className="grid h-full min-h-0 grid-cols-[calc(var(--w-label)*2)_minmax(0,1fr)]" data-sql="true">
        <nav className="min-h-0 overflow-auto border-r border-line-weak py-1" aria-label="Tables" data-sql-tables="true">
          {tables.map((entry) => (
            <button
              key={entry.name}
              type="button"
              className="flex h-row w-full min-w-0 items-center gap-2 px-3 text-left font-mono text-ink hover:bg-header-quaternary hover:text-strong data-[picked=true]:bg-raised data-[picked=true]:text-strong"
              title={`Browse ${entry.name}`}
              data-sql-table={entry.name}
              data-picked={picked === entry.name}
              onClick={() => pick(entry)}
            >
              <span className="min-w-0 flex-1 truncate">{entry.name}</span>
              {entry.count === null ? null : <span className="flex-none text-micro text-label tabular-nums">{entry.count}</span>}
            </button>
          ))}
        </nav>
        <div className="grid min-h-0 grid-rows-[calc(var(--h-row)*7)_minmax(0,1fr)]">
          <div ref={host} className="min-h-0 border-b border-line-weak" data-sql-editor="true" />
          <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)]">
            <div className="flex items-center gap-1 px-3 py-2" role="tablist" aria-label="Results">
              {(["rows", "structure"] as const).map((name) => (
                <button
                  key={name}
                  type="button"
                  role="tab"
                  aria-selected={mode === name}
                  className="flex h-row-sm items-center rounded-blender px-2 text-label capitalize hover:text-ink data-[active=true]:bg-raised data-[active=true]:text-strong"
                  data-active={mode === name}
                  data-sql-mode={name}
                  onClick={() => setMode(name)}
                >
                  {name}
                </button>
              ))}
            </div>
            {mode === "structure" ? (
              table ? (
                <Structure table={table} />
              ) : (
                <p className="m-0 flex min-h-row items-center px-4 text-label">Pick a table to see its structure.</p>
              )
            ) : error ? (
              <p className="m-0 px-4 py-1 font-mono text-micro whitespace-pre-wrap text-failed-text" role="alert" data-sql-error="true">
                {error}
              </p>
            ) : result ? (
              <DataTable
                key={result.run}
                id="sql"
                rows={rows}
                columns={columns}
                rowKey={(row) => row.key}
                literalHeaders
                fold={(row) => (
                  <pre className="m-0 px-4 py-2 font-mono text-micro whitespace-pre-wrap text-ink">
                    {pretty(Object.fromEntries((result.columns ?? []).map((name, index) => [name, parsed(row.cells[index])])))}
                  </pre>
                )}
                empty="No rows."
              />
            ) : (
              <p className="m-0 flex min-h-row items-center px-4 text-label">{sessionId ? "Pick a table, or write a query and press Cmd+Enter." : "Open a conversation to query its database."}</p>
            )}
          </div>
        </div>
      </div>
    </Panel>
  );
}
