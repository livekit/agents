"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { ConnectionState, RoomEvent, TokenSourceConfigurable, type TokenSourceFetchOptions, type TokenSourceResponseObject } from "livekit-client";
import { useAgent, usePersistentUserChoices, useSession, useSessionMessages, type ReceivedMessage } from "@livekit/components-react";

/**
 * The room token for a call, from the console's own `/api/token`: the
 * current session id rides in the dispatch metadata, so the voice door
 * joins the conversation on screen. A new call is a new room, and a
 * conversation that has no session yet gets one from the route, which the
 * page adopts. One click mints one token: the answer is kept for a few
 * seconds, so a connect retried without the microphone reuses it instead
 * of minting again and making a second session.
 *
 * Armed by the speech button and by nothing else. `useSession` asks its
 * token source for a token on mount (to prepare the connection) and again
 * when the room drops; unarmed, the answer is a refusal, so nothing is
 * minted, no session is made and no agent dispatched before the click.
 */
class CallTokenSource extends TokenSourceConfigurable {
  private last: { body: string; at: number; answer: TokenSourceResponseObject } | null = null;

  constructor(
    private readonly armed: RefObject<boolean>,
    private readonly onSession: RefObject<(sessionId: string) => void>,
  ) {
    super();
  }

  async fetch(options: TokenSourceFetchOptions): Promise<TokenSourceResponseObject> {
    if (!this.armed.current) throw new Error("no call yet: the speech button starts one");
    const request = JSON.stringify({
      participant_identity: options.participantIdentity,
      participant_name: options.participantName,
      room_config: { agents: [{ agent_name: options.agentName, metadata: options.agentMetadata }] },
    });
    if (this.last && this.last.body === request && Date.now() - this.last.at < 20_000) return this.last.answer;
    const response = await fetch("/api/token", { method: "POST", headers: { "Content-Type": "application/json" }, body: request });
    const body = (await response.json().catch(() => ({}))) as {
      server_url?: string;
      participant_token?: string;
      session_id?: string;
      error?: string;
    };
    if (!response.ok || !body.server_url || !body.participant_token) throw new Error(body.error ?? `token: ${response.status}`);
    if (body.session_id) this.onSession.current(body.session_id);
    const answer = { serverUrl: body.server_url, participantToken: body.participant_token };
    this.last = { body: request, at: Date.now(), answer };
    return answer;
  }
}

/** The failure is the microphone's (none, refused, busy), not the connection's. */
const isMicrophoneError = (cause: unknown) =>
  /permission|device|media|microphone|NotAllowed|NotFound|NotReadable/i.test(String((cause as Error).name) + String((cause as Error).message));

/** A voice turn as the thread shows it, beside the text turns. */
export type VoiceTurn = { id: string; at: number; role: "user" | "agent"; text: string };

/** The room's transcriptions as turns: the person's as the realtime model heard them, the agent's as it spoke them. */
function toTurns(messages: ReceivedMessage[], since: number): VoiceTurn[] {
  const out: VoiceTurn[] = [];
  for (const message of messages) {
    const text = message.message?.trim();
    if (!text || message.timestamp < since) continue;
    const role = message.type === "userTranscript" || message.from?.isLocal ? "user" : "agent";
    out.push({ id: message.id, at: message.timestamp, role, text });
  }
  return out;
}

/** Two lists of turns as one, by id, the later text winning, in time order. */
function mergeTurns(kept: VoiceTurn[], live: VoiceTurn[]): VoiceTurn[] {
  const byId = new Map(kept.map((turn) => [turn.id, turn]));
  for (const turn of live) byId.set(turn.id, { ...turn, at: byId.get(turn.id)?.at ?? turn.at });
  return [...byId.values()].sort((a, b) => a.at - b.at);
}

/**
 * The voice half of the conversation: a LiveKit session (agents-ui's
 * session hooks) whose agent is dispatched into the session on screen.
 * `start` connects with the microphone on, the one picked last time when
 * there is one (kept the way agents-ui keeps it, in the browser); `end`
 * hangs up; `setMicrophone` switches the microphone of the call and
 * remembers it. The turns are
 * the room's transcriptions, kept after the call ends so the thread keeps
 * them until the page reloads (when they come back from the history):
 * what a call said is snapshotted into `kept` when it ends or the next one
 * starts, and `reset` forgets everything for a new conversation.
 */
export function useCall({ sessionId, sub, agent, endpoint, onSession }: { sessionId: string | null; sub: string; agent: string; endpoint: string; onSession: (sessionId: string) => void }) {
  const adopt = useRef(onSession);
  useEffect(() => {
    adopt.current = onSession;
  }, [onSession]);
  const armed = useRef(false);
  const tokenSource = useMemo(() => new CallTokenSource(armed, adopt), []);
  const { userChoices, saveAudioInputDeviceId } = usePersistentUserChoices();
  const session = useSession(tokenSource, {
    agentName: agent,
    agentMetadata: JSON.stringify({ session_id: sessionId ?? undefined, sub, endpoint }),
    participantIdentity: sub,
    participantName: sub,
    agentConnectTimeoutMilliseconds: 45_000,
  });
  const agentState = useAgent(session);
  const { messages } = useSessionMessages(session);
  const latest = useRef(messages);
  useEffect(() => {
    latest.current = messages;
  }, [messages]);

  const [kept, setKept] = useState<VoiceTurn[]>([]);
  const [since, setSince] = useState(0);
  const [began, setBegan] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);

  const turns = useMemo(() => mergeTurns(kept, toTurns(messages, since)), [kept, messages, since]);
  const keep = useCallback(() => setKept((current) => mergeTurns(current, toTurns(latest.current, since))), [since]);

  const start = useCallback(async () => {
    setError(null);
    setConnecting(true);
    keep();
    armed.current = true;
    try {
      await session.room.startAudio().catch(() => undefined);
      /* The microphone picked last time, set on the room before the track is made. */
      if (userChoices.audioDeviceId) await session.room.switchActiveDevice("audioinput", userChoices.audioDeviceId, false).catch(() => undefined);
      try {
        await session.start({ tracks: { microphone: { enabled: true } } });
      } catch (cause) {
        /* No microphone (a browser without one, or permission refused): the
           call still connects and the person types. When the room joined
           before the microphone failed it stays joined; otherwise connect
           without the mic, on the same token. A failure that is not the
           microphone's is the call's. */
        if (!isMicrophoneError(cause)) throw cause;
        const joined = () => session.room.state === ConnectionState.Connected;
        if (!joined()) {
          try {
            await session.start({ tracks: { microphone: { enabled: false } } });
          } catch (again) {
            if (!joined()) throw again;
          }
        }
      }
    } catch (cause) {
      setError((cause as Error).message);
      throw cause;
    } finally {
      setConnecting(false);
    }
  }, [keep, session, userChoices.audioDeviceId]);

  const end = useCallback(async () => {
    keep();
    armed.current = false;
    await session.end();
  }, [keep, session]);

  /** Another microphone for this call and the next ones. */
  const setMicrophone = useCallback(
    async (deviceId: string) => {
      saveAudioInputDeviceId(deviceId);
      await session.room.switchActiveDevice("audioinput", deviceId);
    },
    [saveAudioInputDeviceId, session],
  );

  const connected = session.isConnected;
  /* The call began when the room connected, however the microphone fared; a
     room that dropped on its own is not re-minted behind the person's back,
     the next click arms again. */
  useEffect(() => {
    const room = session.room;
    const onConnected = () => setBegan(Date.now());
    const onDisconnected = () => {
      setBegan(null);
      armed.current = false;
    };
    room.on(RoomEvent.Connected, onConnected);
    room.on(RoomEvent.Disconnected, onDisconnected);
    return () => {
      room.off(RoomEvent.Connected, onConnected);
      room.off(RoomEvent.Disconnected, onDisconnected);
    };
  }, [session.room]);
  const startedAt = began;

  const reset = useCallback(() => {
    setKept([]);
    setSince(Date.now());
  }, []);

  return { session, agent: agentState, turns, reset, connected, connecting, startedAt, error, start, end, setMicrophone };
}

/** mm:ss since the call began. */
export function elapsed(startedAt: number | null, now: number) {
  if (!startedAt) return "0:00";
  const seconds = Math.max(0, Math.floor((now - startedAt) / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}
