/**
 * A browser client of agent-db's data plane (`livekit.AgentDB.Wire`): one
 * WebSocket to one database, one protobuf message per binary frame. It says
 * `Hello` first, pings at the interval the answer gives, runs queries
 * (`Columns`, `ColumnBatch`*, `Done`, paced by `Credit`) and redials with a
 * fresh token when the socket drops. Only the messages it uses are encoded.
 */

/** Where a database answers and the token to dial it with (`/api/sessions/<id>/db`). */
export type Target = { url: string; databaseId: string; token: string };

export type Status =
  | { state: "connecting" }
  | { state: "open"; tip: number; latency: number | null }
  | { state: "closed"; error: string; retryIn: number };

export type QueryResult = { columns: string[]; rows: unknown[][]; truncated: boolean; tip: number; ms: number };

/* ---------------------------------------------------------------- encoding */

const utf8 = new TextEncoder();
const fromUtf8 = new TextDecoder();

function varint(out: number[], value: number) {
  let n = value;
  while (n >= 0x80) {
    out.push((n % 0x80) | 0x80);
    n = Math.floor(n / 0x80);
  }
  out.push(n);
}

function bytesField(out: number[], field: number, bytes: Uint8Array | number[]) {
  varint(out, (field << 3) | 2);
  varint(out, bytes.length);
  for (const b of bytes) out.push(b);
}

function intField(out: number[], field: number, value: number) {
  varint(out, field << 3);
  varint(out, value);
}

/** A `ClientMessage`: its request id and one message, already encoded, in `field`. */
function clientMessage(requestId: number, field: number, body: number[]) {
  const out: number[] = [];
  intField(out, 1, requestId);
  bytesField(out, field, body);
  return new Uint8Array(out);
}

const message = {
  hello: (token: string, databaseId: string) => {
    const out: number[] = [];
    bytesField(out, 1, utf8.encode(token));
    bytesField(out, 2, utf8.encode(databaseId));
    return out;
  },
  query: (sql: string) => {
    const out: number[] = [];
    bytesField(out, 1, utf8.encode(sql));
    return out;
  },
  credit: (batches: number) => {
    const out: number[] = [];
    intField(out, 1, batches);
    return out;
  },
  ping: (timestampMs: number) => {
    const out: number[] = [];
    intField(out, 1, timestampMs);
    return out;
  },
};
const FIELD = { hello: 2, query: 4, cancel: 9, credit: 10, ping: 11 };

/* ---------------------------------------------------------------- decoding */

class Reader {
  pos = 0;
  constructor(readonly buf: Uint8Array) {}

  get done() {
    return this.pos >= this.buf.length;
  }

  varint() {
    let value = 0;
    let scale = 1;
    for (;;) {
      const b = this.buf[this.pos++];
      value += (b & 0x7f) * scale;
      if (b < 0x80) return value;
      scale *= 0x80;
    }
  }

  bytes() {
    const length = this.varint();
    const out = this.buf.subarray(this.pos, this.pos + length);
    this.pos += length;
    return out;
  }

  /** Each field: its number and a reader of its value, by wire type. */
  *fields(): Generator<[number, number, Reader]> {
    while (!this.done) {
      const key = this.varint();
      yield [key >>> 3, key & 7, this];
    }
  }

  skip(wire: number) {
    if (wire === 0) this.varint();
    else if (wire === 1) this.pos += 8;
    else if (wire === 2) this.bytes();
    else if (wire === 5) this.pos += 4;
  }
}

/** A repeated scalar, packed or not, read with `one`. */
function repeated(reader: Reader, wire: number, one: (r: Reader) => number, into: number[]) {
  if (wire !== 2) return void into.push(one(reader));
  const inner = new Reader(reader.bytes());
  while (!inner.done) into.push(one(inner));
}

const zigzag = (n: number) => (n % 2 === 0 ? n / 2 : -(n + 1) / 2);
const double = (r: Reader) => {
  const value = new DataView(r.buf.buffer, r.buf.byteOffset + r.pos, 8).getFloat64(0, true);
  r.pos += 8;
  return value;
};

/** One column of a batch, back into its cells (`Wire.Column`): a blob is `{blob: size}`. */
function column(buf: Uint8Array): unknown[] {
  let types: Uint8Array = new Uint8Array();
  const ints: number[] = [];
  const doubles: number[] = [];
  const textEnds: number[] = [];
  const blobEnds: number[] = [];
  let text: Uint8Array = new Uint8Array();
  for (const [field, wire, r] of new Reader(buf).fields()) {
    if (field === 1) types = r.bytes();
    else if (field === 2) repeated(r, wire, (x) => zigzag(x.varint()), ints);
    else if (field === 3) repeated(r, wire, double, doubles);
    else if (field === 4) text = r.bytes();
    else if (field === 5) repeated(r, wire, (x) => x.varint(), textEnds);
    else if (field === 6) r.bytes();
    else if (field === 7) repeated(r, wire, (x) => x.varint(), blobEnds);
    else r.skip(wire);
  }
  const at = { int: 0, double: 0, text: 0, blob: 0 };
  return Array.from(types, (type) => {
    switch (type) {
      case 1:
        return ints[at.int++];
      case 2:
        return doubles[at.double++];
      case 3: {
        const i = at.text++;
        return fromUtf8.decode(text.subarray(i ? textEnds[i - 1] : 0, textEnds[i]));
      }
      case 4: {
        const i = at.blob++;
        return { blob: blobEnds[i] - (i ? blobEnds[i - 1] : 0) };
      }
      default:
        return null;
    }
  });
}

type ServerMessage =
  | { kind: "helloOk"; tip: number; pingIntervalMs: number }
  | { kind: "columns"; names: string[] }
  | { kind: "batch"; rows: unknown[][] }
  | { kind: "done"; tip: number }
  | { kind: "error"; code: string; message: string }
  | { kind: "pong"; lastPingMs: number }
  | { kind: "other" };

function decode(buf: Uint8Array): { requestId: number; message: ServerMessage } {
  let requestId = 0;
  let out: ServerMessage = { kind: "other" };
  for (const [field, wire, r] of new Reader(buf).fields()) {
    if (field === 1) {
      requestId = r.varint();
      continue;
    }
    if (wire !== 2) {
      r.skip(wire);
      continue;
    }
    const body = new Reader(r.bytes());
    const ints: Record<number, number> = {};
    const strings: Record<number, string[]> = {};
    const chunks: Uint8Array[] = [];
    for (const [f, w, b] of body.fields()) {
      if (w === 0) ints[f] = b.varint();
      else if (w === 2 && field === 4 && f === 1) chunks.push(b.bytes());
      else if (w === 2) (strings[f] ??= []).push(fromUtf8.decode(b.bytes()));
      else b.skip(w);
    }
    if (field === 2) out = { kind: "helloOk", tip: ints[1] ?? 0, pingIntervalMs: ints[2] ?? 0 };
    else if (field === 3) out = { kind: "columns", names: strings[1] ?? [] };
    else if (field === 4) {
      const cols = chunks.map(column);
      out = { kind: "batch", rows: Array.from({ length: ints[2] ?? cols[0]?.length ?? 0 }, (_, i) => cols.map((c) => c[i])) };
    } else if (field === 6) out = { kind: "done", tip: ints[1] ?? 0 };
    else if (field === 7) out = { kind: "error", code: strings[1]?.[0] ?? "", message: strings[2]?.[0] ?? "" };
    else if (field === 8) out = { kind: "pong", lastPingMs: ints[1] ?? 0 };
  }
  return { requestId, message: out };
}

/* ---------------------------------------------------------------- the client */

const RETRY_MS = [500, 1000, 2000, 5000];

type Pending = {
  limit: number;
  started: number;
  columns: string[];
  rows: unknown[][];
  resolve: (result: QueryResult) => void;
  reject: (error: Error) => void;
};

/**
 * One database's socket for as long as it is wanted: `target` is asked for
 * a fresh URL and token on every dial, `onStatus` hears each change.
 */
export class Wsdb {
  private socket: WebSocket | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private ping: ReturnType<typeof setInterval> | null = null;
  private retry: ReturnType<typeof setTimeout> | null = null;
  private attempt = 0;
  private closed = false;
  private tip = 0;
  private latency: number | null = null;
  private ready: Promise<void>;
  private markReady = () => {};

  constructor(
    private readonly target: () => Promise<Target>,
    private readonly onStatus: (status: Status) => void,
  ) {
    this.ready = new Promise((resolve) => (this.markReady = resolve));
    void this.dial();
  }

  /** One statement's rows, at most `limit`; past it the query is cancelled. */
  async query(sql: string, limit = 500): Promise<QueryResult> {
    await this.ready;
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) throw new Error("Not connected to the database.");
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { limit, started: performance.now(), columns: [], rows: [], resolve, reject });
      socket.send(clientMessage(id, FIELD.query, message.query(sql)));
    });
  }

  close() {
    this.closed = true;
    this.stop();
    this.socket?.close();
  }

  private status() {
    this.onStatus({ state: "open", tip: this.tip, latency: this.latency });
  }

  private stop() {
    if (this.ping) clearInterval(this.ping);
    if (this.retry) clearTimeout(this.retry);
    this.ping = this.retry = null;
  }

  private async dial() {
    this.onStatus({ state: "connecting" });
    let target: Target;
    try {
      target = await this.target();
    } catch (e) {
      return this.lost(e instanceof Error ? e.message : String(e));
    }
    if (this.closed) return;
    const url = new URL(target.url);
    url.searchParams.set("access_token", target.token);
    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    socket.onopen = () => socket.send(clientMessage(this.nextId++, FIELD.hello, message.hello(target.token, target.databaseId)));
    socket.onmessage = (event) => this.receive(new Uint8Array(event.data as ArrayBuffer));
    socket.onclose = (event) => {
      if (this.socket === socket) this.lost(event.reason || "The connection closed.");
    };
  }

  private lost(error: string) {
    this.stop();
    this.socket = null;
    for (const query of this.pending.values()) query.reject(new Error(error));
    this.pending.clear();
    if (this.closed) return;
    const retryIn = RETRY_MS[Math.min(this.attempt++, RETRY_MS.length - 1)];
    this.onStatus({ state: "closed", error, retryIn });
    this.ready = new Promise((resolve) => (this.markReady = resolve));
    this.retry = setTimeout(() => void this.dial(), retryIn);
  }

  private send(id: number, field: number, body: number[]) {
    this.socket?.send(clientMessage(id, field, body));
  }

  private receive(buf: Uint8Array) {
    const { requestId, message: m } = decode(buf);
    if (m.kind === "helloOk") {
      this.attempt = 0;
      this.tip = m.tip;
      this.status();
      this.markReady();
      const every = m.pingIntervalMs || 10_000;
      const ping = () => this.send(this.nextId++, FIELD.ping, message.ping(Date.now()));
      this.ping = setInterval(ping, every);
      ping();
      return;
    }
    if (m.kind === "pong") {
      this.latency = Date.now() - m.lastPingMs;
      return this.status();
    }
    const query = this.pending.get(requestId);
    if (m.kind === "error" && !query) {
      /* Not a query's: the Hello was refused, or the socket is done for. */
      const socket = this.socket;
      this.socket = null;
      socket?.close();
      return this.lost(m.message || m.code);
    }
    if (!query) return;
    if (m.kind === "columns") query.columns = m.names;
    else if (m.kind === "batch") {
      query.rows.push(...m.rows);
      if (query.rows.length > query.limit) {
        this.send(requestId, FIELD.cancel, []);
        this.finish(requestId, query, this.tip, true);
      } else this.send(requestId, FIELD.credit, message.credit(1));
    } else if (m.kind === "done") {
      this.tip = Math.max(this.tip, m.tip);
      this.status();
      this.finish(requestId, query, m.tip, false);
    } else if (m.kind === "error") {
      this.pending.delete(requestId);
      query.reject(new Error(m.message || m.code));
    }
  }

  private finish(id: number, query: Pending, tip: number, truncated: boolean) {
    this.pending.delete(id);
    query.resolve({
      columns: query.columns,
      rows: query.rows.slice(0, query.limit),
      truncated,
      tip,
      ms: Math.round(performance.now() - query.started),
    });
  }
}
