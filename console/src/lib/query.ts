import { querySnapshot } from "./api";
export type { QueryResult } from "./generated/QueryResult";
export type Status =
  | { state: "connecting" }
  | { state: "open"; tip: number; latency: number | null }
  | { state: "closed"; error: string; retryIn: number };

/** Read-only queries of an exported snapshot, authorized by the console backend. */
export class SnapshotDB {
  private controller = new AbortController();
  constructor(private readonly id: string, private readonly onStatus: (status: Status) => void) {}
  async query(sql: string, limit = 500) {
    const result = await querySnapshot(this.id, sql, limit, this.controller.signal);
    this.onStatus({ state: "open", tip: result.tip, latency: result.ms });
    return result;
  }
  close() { this.controller.abort(); }
}
