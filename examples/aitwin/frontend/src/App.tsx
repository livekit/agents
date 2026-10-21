import {
  FormEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'
import {
  AiTwin,
  type AiTwinHandle,
  type AiTwinSpeechChunk,
  type TtsEngineId,
} from '@streamoji/aitwin'
import {
  Room,
  Track,
  createLocalAudioTrack,
  type LocalAudioTrack,
} from 'livekit-client'

type TextMode = 'chat' | 'say'

function requireViteEnv(
  name: 'VITE_TOPIC_CHAT' | 'VITE_TOPIC_SPEECH' | 'VITE_TOPIC_SAY',
): string {
  const value = (import.meta.env[name] as string | undefined)?.trim()
  if (!value) {
    throw new Error(`Missing ${name} — set it in examples/aitwin/frontend/.env`)
  }
  return value
}

const TOPIC_CHAT = requireViteEnv('VITE_TOPIC_CHAT')
const TOPIC_SPEECH = requireViteEnv('VITE_TOPIC_SPEECH')
const TOPIC_SAY = requireViteEnv('VITE_TOPIC_SAY')

type ConnectPayload = {
  url: string
  token: string
  roomName: string
  aitwinAuthToken: string
  aitwinTwinId: string
  aitwinApiBase: string
}

type SpeechBeginMsg = {
  type: 'begin'
  text?: string
  ttsEngineId?: string
  sampleRate?: number
}

type SpeechChunkMsg = {
  type: 'chunk'
  audio?: string
  sampleRate?: number
  visemes?: AiTwinSpeechChunk['visemes']
  words?: AiTwinSpeechChunk['words']
  isNewSegment?: boolean
  audioOffsetMs?: number
}

type SpeechEndMsg = { type: 'end' }

type SpeechMsg = SpeechBeginMsg | SpeechChunkMsg | SpeechEndMsg

function parseSpeechMsg(raw: string): SpeechMsg | null {
  try {
    const data = JSON.parse(raw) as SpeechMsg
    if (!data || typeof data !== 'object' || !('type' in data)) return null
    if (data.type === 'begin' || data.type === 'chunk' || data.type === 'end') {
      return data
    }
    return null
  } catch {
    return null
  }
}

export default function App() {
  const twinRef = useRef<AiTwinHandle | null>(null)
  const roomRef = useRef<Room | null>(null)
  const micTrackRef = useRef<LocalAudioTrack | null>(null)
  const speakingRef = useRef(false)
  const speechQueueRef = useRef<SpeechMsg[]>([])
  const twinReadyRef = useRef(false)
  const speechOpenRef = useRef(false)
  const lastAgentTextRef = useRef<string | null>(null)
  const drainAllRef = useRef<() => void>(() => {})

  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle')
  const [twinReady, setTwinReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [roomName, setRoomName] = useState<string | null>(null)
  const [lastHeard, setLastHeard] = useState<string | null>(null)
  const [lastAgentText, setLastAgentText] = useState<string | null>(null)
  const [queuedCount, setQueuedCount] = useState(0)
  const [draft, setDraft] = useState('')
  const [mode, setMode] = useState<TextMode>('chat')
  const [sending, setSending] = useState(false)
  const [micEnabled, setMicEnabled] = useState(false)
  const [auth, setAuth] = useState<{
    authToken: string
    twinId: string
    apiBase: string
  } | null>(null)

  const updateQueuedCount = useCallback(() => {
    setQueuedCount(speechQueueRef.current.length)
  }, [])

  // Only agent-published chunks (TOPIC_SPEECH). Never twin.speakText /avatar_ttsWithPoses.
  const drainAll = useCallback(async () => {
    if (speakingRef.current) return
    const twin = twinRef.current
    if (!twin || !twinReadyRef.current) return
    if (!speechQueueRef.current.length) return

    speakingRef.current = true
    try {
      while (speechQueueRef.current.length) {
        const msg = speechQueueRef.current.shift()
        updateQueuedCount()
        if (!msg) break

        if (msg.type === 'begin') {
          if (speechOpenRef.current) {
            twin.endSpeech()
            speechOpenRef.current = false
          }
          const text = (msg.text || '').trim()
          if (text) {
            lastAgentTextRef.current = text
            setLastAgentText(text)
            setLastHeard(text)
          }
          console.log('[AiTwin] beginSpeech', text.slice(0, 120))
          await twin.beginSpeech({
            ttsEngineId: msg.ttsEngineId as TtsEngineId | undefined,
            spokenText: text || undefined,
          })
          speechOpenRef.current = true
        } else if (msg.type === 'chunk') {
          if (!speechOpenRef.current) {
            await twin.beginSpeech({
              spokenText: lastAgentTextRef.current ?? undefined,
            })
            speechOpenRef.current = true
          }
          const chunk: AiTwinSpeechChunk = {
            audio: msg.audio || '',
            sampleRate: msg.sampleRate,
            visemes: msg.visemes || [],
            words: msg.words || [],
            isNewSegment: msg.isNewSegment ?? false,
            audioOffsetMs: msg.audioOffsetMs ?? 0,
          }
          console.log(
            '[AiTwin] pushSpeechChunk',
            `visemes=${chunk.visemes.length}`,
            `audio=${chunk.audio ? 'yes' : 'no'}`,
            `sr=${chunk.sampleRate ?? 'default'}`,
          )
          await twin.pushSpeechChunk(chunk)
        } else if (msg.type === 'end') {
          if (speechOpenRef.current) {
            twin.endSpeech()
            speechOpenRef.current = false
          }
        }
      }
    } catch (err) {
      console.error('[AiTwin] speech chunk pipeline failed', err)
      setError(err instanceof Error ? err.message : 'speech pipeline failed')
      if (speechOpenRef.current) {
        try {
          twin.endSpeech()
        } catch {
          /* ignore */
        }
        speechOpenRef.current = false
      }
    } finally {
      speakingRef.current = false
      if (speechQueueRef.current.length) {
        void drainAllRef.current()
      } else {
        updateQueuedCount()
      }
    }
  }, [updateQueuedCount])

  useEffect(() => {
    drainAllRef.current = () => {
      void drainAll()
    }
  }, [drainAll])

  const enqueueSpeechMsg = useCallback(
    (msg: SpeechMsg, source: string) => {
      console.log(`[enqueue-speech:${source}]`, msg.type)
      speechQueueRef.current.push(msg)
      updateQueuedCount()
      void drainAll()
    },
    [drainAll, updateQueuedCount],
  )

  useEffect(() => {
    twinReadyRef.current = twinReady
    if (twinReady) void drainAll()
  }, [twinReady, drainAll])

  const disconnect = useCallback(async () => {
    speechQueueRef.current = []
    speakingRef.current = false
    twinReadyRef.current = false
    if (speechOpenRef.current) {
      try {
        twinRef.current?.endSpeech()
      } catch {
        /* ignore */
      }
      speechOpenRef.current = false
    }

    const mic = micTrackRef.current
    micTrackRef.current = null
    if (mic) {
      mic.stop()
      try {
        await roomRef.current?.localParticipant.unpublishTrack(mic)
      } catch {
        /* ignore */
      }
    }

    const room = roomRef.current
    roomRef.current = null
    if (room) {
      await room.disconnect()
    }

    setMicEnabled(false)
    setStatus('idle')
    setRoomName(null)
    setDraft('')
    setLastHeard(null)
    setLastAgentText(null)
    setQueuedCount(0)
    setAuth(null)
    setTwinReady(false)
  }, [])

  useEffect(() => {
    return () => {
      void disconnect()
    }
  }, [disconnect])

  const attachTextHandlers = useCallback(
    (room: Room) => {
      room.registerTextStreamHandler(TOPIC_SPEECH, (reader, participantInfo) => {
        void (async () => {
          try {
            const raw = (await reader.readAll()).trim()
            if (!raw) return
            if (participantInfo.identity === room.localParticipant.identity) {
              console.log('[skip local aitwin.speech]', raw.slice(0, 80))
              return
            }
            const msg = parseSpeechMsg(raw)
            if (!msg) {
              console.warn('[aitwin.speech] invalid JSON', raw.slice(0, 120))
              return
            }
            enqueueSpeechMsg(msg, `${TOPIC_SPEECH}:${participantInfo.identity}`)
          } catch (err) {
            console.warn(`${TOPIC_SPEECH} stream failed`, err)
          }
        })()
      })
    },
    [enqueueSpeechMsg],
  )

  const connect = useCallback(async () => {
    setError(null)
    setStatus('connecting')

    try {
      await disconnect()

      const response = await fetch('/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ room: `aitwin-demo-${Date.now()}` }),
      })

      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.error || `Token request failed (${response.status})`)
      }

      const data = (await response.json()) as ConnectPayload & { aitwinError?: string }
      if (!data.aitwinAuthToken) {
        throw new Error(
          data.aitwinError ||
            'AiTwin auth token missing — set AITWIN_CLIENT_ID / AITWIN_CLIENT_SECRET on the token server',
        )
      }

      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
      })
      attachTextHandlers(room)

      await room.connect(data.url, data.token)
      roomRef.current = room

      setAuth({
        authToken: data.aitwinAuthToken,
        twinId: data.aitwinTwinId || 'victoria',
        apiBase: data.aitwinApiBase || 'https://ai.aitwin.me',
      })
      setRoomName(data.roomName)
      setStatus('connected')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Connection failed'
      console.error(err)
      setError(message)
      setStatus('error')
    }
  }, [attachTextHandlers, disconnect])

  const toggleMic = useCallback(async () => {
    const room = roomRef.current
    if (!room || status !== 'connected') return

    setError(null)
    try {
      if (micEnabled) {
        const mic = micTrackRef.current
        micTrackRef.current = null
        if (mic) {
          await room.localParticipant.unpublishTrack(mic)
          mic.stop()
        }
        setMicEnabled(false)
        return
      }

      const track = await createLocalAudioTrack({
        echoCancellation: true,
        noiseSuppression: true,
      })
      await room.localParticipant.publishTrack(track, {
        source: Track.Source.Microphone,
      })
      micTrackRef.current = track
      setMicEnabled(true)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Microphone failed'
      console.error(err)
      setError(message)
    }
  }, [micEnabled, status])

  const onSubmitText = useCallback(
    async (event: FormEvent) => {
      event.preventDefault()
      const text = draft.trim()
      const room = roomRef.current
      if (!text || !room || sending) return

      setSending(true)
      setError(null)
      try {
        if (mode === 'say') {
          // Exact words via agent TTS + alignVisemes — never browser speakText /avatar_ttsWithPoses.
          await room.localParticipant.sendText(text, { topic: TOPIC_SAY })
        } else {
          await room.localParticipant.sendText(text, { topic: TOPIC_CHAT })
        }
        setDraft('')
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to send text'
        console.error(err)
        setError(message)
      } finally {
        setSending(false)
      }
    },
    [draft, mode, sending],
  )

  return (
    <div className="page">
      <header className="header">
        <div>
          <p className="eyebrow">AiTwin × LiveKit Agents</p>
          <h1>STT → LLM → LiveKit TTS + G2P → AiTwin</h1>
          <p className="lede">
            LiveKit runs conversation and synthesizes speech. Visemes come from G2P
            (or HeadTTS). AiTwin plays the audio and drives lipsync — no agent audio track.
          </p>
        </div>
        <div className="actions">
          {status !== 'connected' ? (
            <button type="button" onClick={() => void connect()} disabled={status === 'connecting'}>
              {status === 'connecting' ? 'Connecting…' : 'Connect'}
            </button>
          ) : (
            <button type="button" className="secondary" onClick={() => void disconnect()}>
              Disconnect
            </button>
          )}
        </div>
      </header>

      <main className="stage">
        <div className="avatar-shell">
          <div className="avatar-container">
            {auth ? (
              <AiTwin
                key={auth.authToken}
                ref={twinRef}
                id={auth.twinId}
                authToken={auth.authToken}
                apiBase={auth.apiBase}
                onReady={() => {
                  console.log('[AiTwin] ready')
                  twinReadyRef.current = true
                  setTwinReady(true)
                }}
                onError={(message) => {
                  console.error('[AiTwin]', message)
                  setError(message)
                }}
                showErrorOverlay={false}
                style={{
                  display: 'block',
                  width: '100%',
                  height: '100%',
                  lineHeight: 0,
                }}
                canvasStyle={{
                  display: 'block',
                  width: '100%',
                  height: '100%',
                }}
              />
            ) : (
              <div className="avatar-placeholder">Connect to load the AiTwin avatar</div>
            )}
          </div>
        </div>

        <aside className="panel">
          <p>
            <strong>Status:</strong> {status}
            {twinReady ? ' · twin ready' : auth ? ' · loading twin…' : ''}
            {queuedCount > 0 ? ` · queued ${queuedCount}` : ''}
          </p>
          {roomName && (
            <p>
              <strong>Room:</strong> {roomName}
            </p>
          )}
          {lastAgentText && (
            <p className="heard">
              <strong>Agent text:</strong> {lastAgentText}
            </p>
          )}
          {lastHeard && (
            <p className="heard">
              <strong>Speaking:</strong> {lastHeard}
            </p>
          )}
          {error && <p className="error">{error}</p>}

          {status === 'connected' ? (
            <>
              <button
                type="button"
                className={micEnabled ? 'mic on' : 'mic'}
                onClick={() => void toggleMic()}
              >
                {micEnabled ? 'Mic on — tap to mute' : 'Enable mic (STT)'}
              </button>

              <form className="text-form" onSubmit={(e) => void onSubmitText(e)}>
                <div className="mode-row" role="group" aria-label="Text mode">
                  <button
                    type="button"
                    className={mode === 'chat' ? 'mode active' : 'mode'}
                    onClick={() => setMode('chat')}
                  >
                    Chat (LLM)
                  </button>
                  <button
                    type="button"
                    className={mode === 'say' ? 'mode active' : 'mode'}
                    onClick={() => setMode('say')}
                  >
                    Speak text
                  </button>
                </div>
                <textarea
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  placeholder={
                    mode === 'say'
                      ? 'Exact words → agent TTS + alignVisemes…'
                      : 'Ask the agent something…'
                  }
                  rows={3}
                />
                <button type="submit" disabled={sending || !draft.trim()}>
                  {sending ? 'Sending…' : mode === 'say' ? 'Speak' : 'Send chat'}
                </button>
                <p className="hint">
                  {mode === 'say'
                    ? 'Speak text → LiveKit TTS + alignVisemes (no LLM, no avatar_ttsWithPoses).'
                    : 'Chat → LiveKit LLM → TTS + alignVisemes → aitwin.speech → lipsync.'}
                </p>
              </form>
            </>
          ) : (
            <p className="hint">
              Connect, then use mic and/or chat. Agent publishes speech chunks, not an audio track.
            </p>
          )}

          <ol>
            <li>Connect (LiveKit room + AiTwin auth)</li>
            <li>
              <strong>Mic</strong> → STT → LLM, or <strong>Chat</strong> for typed turns
            </li>
            <li>Agent TTS + G2P → <code>aitwin.speech</code> → <code>pushSpeechChunk</code></li>
          </ol>
        </aside>
      </main>
    </div>
  )
}
