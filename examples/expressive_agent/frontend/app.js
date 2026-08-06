/* Expressive agent demo. The config (expressive on/off + voice) rides on the
 * agent dispatch metadata via the token endpoint, so changing either control
 * reconnects with a freshly dispatched agent. */

const { Room, RoomEvent, Track } = LivekitClient;

// Hue carries valence and saturation carries intensity, so a recessive mood
// never out-shouts a strong one.
const MOOD_COLORS = {
  angry: '#F5222D',
  excited: '#FF7A45',
  happy: '#FFC53D',
  playful: '#F759AB',
  surprised: '#B37FEB',
  anxious: '#D46B08',
  hopeful: '#52C41A',
  empathetic: '#36CFC9',
  curious: '#40A9FF',
  sad: '#2F54EB',
  calm: '#8C9BAB',
};
const NEUTRAL_COLOR = '#1FD5F9';

const EXPRESSION_ATTRIBUTE = 'lk.expression';
const AGENT_STATE_ATTRIBUTE = 'lk.agent.state';
const SEGMENT_ID_ATTRIBUTE = 'lk.segment_id';
const TRANSCRIPTION_TOPIC = 'lk.transcription';
const SETTLE_INTERVAL_MS = 150;
const SETTLE_TICKS = 20;
const MOOD_TTL_TURNS = 2;
const CAPTION_LINES = 4;

const els = {
  aura: document.getElementById('aura'),
  moodRow: document.getElementById('mood-row'),
  moodDot: document.getElementById('mood-dot'),
  moodLabel: document.getElementById('mood-label'),
  expressionLabel: document.getElementById('expression-label'),
  pipeline: document.getElementById('pipeline'),
  mic: document.getElementById('mic'),
  modeExpressive: document.getElementById('mode-expressive'),
  modeFlat: document.getElementById('mode-flat'),
  tts: document.getElementById('tts'),
  connect: document.getElementById('connect'),
  captions: document.getElementById('captions'),
  hint: document.getElementById('hint'),
};

const state = {
  expressive: true,
  tts: 'fishaudio',
  room: null,
  connecting: false,
  agentState: 'disconnected',
  mood: null,
  expression: null,
  turnsSinceMood: 0,
  analyser: null,
  audioEl: null,
  captions: new Map(), // lk.segment_id -> {who, text}
};

// ---- colors ----

function hexToRgb(hex) {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function rgbToHsl([r, g, b]) {
  (r /= 255), (g /= 255), (b /= 255);
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  const h =
    max === r ? ((g - b) / d + (g < b ? 6 : 0)) / 6 : max === g ? ((b - r) / d + 2) / 6 : ((r - g) / d + 4) / 6;
  return [h * 360, s, l];
}

function moodColor() {
  return state.mood && MOOD_COLORS[state.mood] ? MOOD_COLORS[state.mood] : NEUTRAL_COLOR;
}

// ---- aura visualizer ----

// A cluster of soft orbs orbiting a hot core, drawn additively. Orbit speed
// follows the agent state (thinking swirls, listening breathes) and the
// agent's live audio pushes the orbs outward and brightens the core.
class Aura {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.rgb = hexToRgb(NEUTRAL_COLOR); // displayed color, eased toward the mood
    this.energy = 0; // eased audio level
    this.spin = 0;
    this.blobs = Array.from({ length: 7 }, (_, i) => ({
      speed: (0.3 + 0.13 * i) * (i % 2 ? 1 : -1),
      phase: (i * Math.PI * 2) / 7,
      dist: 0.16 + 0.07 * (i % 3),
      size: 0.3 + 0.06 * (i % 4),
      hueShift: (i - 3) * 9,
      band: i,
    }));
    this.resize();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const size = this.canvas.clientWidth;
    this.canvas.width = size * dpr;
    this.canvas.height = size * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.size = size;
  }

  orb(x, y, radius, h, s, l, alpha) {
    const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, `hsla(${h}, ${s}%, ${l}%, ${alpha})`);
    gradient.addColorStop(0.55, `hsla(${h}, ${s}%, ${l * 0.8}%, ${alpha * 0.4})`);
    gradient.addColorStop(1, `hsla(${h}, ${s}%, ${l * 0.7}%, 0)`);
    this.ctx.fillStyle = gradient;
    this.ctx.beginPath();
    this.ctx.arc(x, y, radius, 0, Math.PI * 2);
    this.ctx.fill();
  }

  draw(now, level, bands, mode) {
    const { ctx, size } = this;
    const t = now / 1000;
    const center = size / 2;
    const connected = mode !== 'disconnected';

    // ease color and energy so mood changes wash in instead of snapping
    const target = hexToRgb(moodColor());
    this.rgb = this.rgb.map((c, i) => c + (target[i] - c) * 0.08);
    this.energy += (level - this.energy) * 0.25;

    const [h, s, l] = rgbToHsl(this.rgb);
    const sat = s * 100;
    const swirl = mode === 'thinking' ? 3.2 : mode === 'speaking' ? 1.4 : 0.55;
    this.spin += swirl * 0.016;

    const breathe = connected ? 1 + Math.sin(t * (mode === 'listening' ? 1.2 : 2.1)) * 0.025 : 0.8;
    const scale = size * 0.5 * breathe * (connected ? 1 : 0.75);
    const dim = connected ? 1 : 0.3;

    ctx.clearRect(0, 0, size, size);
    ctx.globalCompositeOperation = 'lighter';

    // orbiting orbs, pushed outward and brightened by their frequency band
    for (const blob of this.blobs) {
      const energy = bands[blob.band % bands.length] * 0.7 + this.energy * 0.3;
      const angle = blob.phase + this.spin * blob.speed;
      const dist = scale * (blob.dist + energy * 0.34);
      const wobble = Math.sin(t * 1.7 + blob.phase * 3) * scale * 0.03;
      const x = center + Math.cos(angle) * (dist + wobble);
      const y = center + Math.sin(angle) * (dist + wobble);
      const radius = scale * (blob.size + energy * 0.25);
      this.orb(x, y, radius, h + blob.hueShift, Math.min(100, sat), l * 100, (0.34 + energy * 0.3) * dim);
    }

    // hot core
    const coreL = Math.min(88, l * 100 + 18 + this.energy * 30);
    this.orb(center, center, scale * (0.4 + this.energy * 0.12), h, sat * 0.9, coreL, 0.75 * dim);

    // halo ring that flares with speech
    ctx.globalCompositeOperation = 'source-over';
    ctx.strokeStyle = `hsla(${h}, ${sat}%, ${Math.min(90, l * 100 + 20)}%, ${(0.1 + this.energy * 0.35) * dim})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(center, center, scale * (0.72 + this.energy * 0.2), 0, Math.PI * 2);
    ctx.stroke();
  }
}

const aura = new Aura(els.aura);
window.addEventListener('resize', () => aura.resize());

const BAND_COUNT = 7;
const bands = new Array(BAND_COUNT).fill(0);

function tick(now) {
  let level = 0;
  if (state.analyser && state.agentState === 'speaking') {
    const { analyser, data } = state.analyser;
    analyser.getByteFrequencyData(data);
    let sum = 0;
    for (const v of data) sum += v * v;
    level = Math.min(1, Math.sqrt(sum / data.length) / 80);
    const per = Math.floor(data.length / BAND_COUNT);
    for (let i = 0; i < BAND_COUNT; i++) {
      let bandSum = 0;
      for (let j = i * per; j < (i + 1) * per; j++) bandSum += data[j];
      bands[i] += (Math.min(1, bandSum / per / 160) - bands[i]) * 0.3;
    }
  } else {
    for (let i = 0; i < BAND_COUNT; i++) bands[i] *= 0.92;
  }
  aura.draw(now, level, bands, state.agentState);
  requestAnimationFrame(tick);
}

// ---- rendering ----

function render() {
  const connected = !!state.room;
  const color = moodColor();

  const showMood = connected && state.expressive;
  els.moodRow.classList.toggle('hidden', !showMood);
  els.moodDot.style.backgroundColor = color;
  els.moodLabel.style.color = color;
  els.moodLabel.textContent = state.mood ?? 'neutral';
  const extra = state.expression && state.expression !== state.mood ? state.expression : '';
  els.expressionLabel.textContent = extra;
  els.expressionLabel.title = extra;

  els.mic.classList.toggle('hidden', !connected);
  els.connect.textContent = state.connecting
    ? 'Connecting…'
    : connected
      ? 'End call'
      : 'Start talking';
  els.connect.disabled = state.connecting;
  els.connect.classList.toggle('danger', connected);

  els.modeExpressive.setAttribute('aria-pressed', String(state.expressive));
  els.modeFlat.setAttribute('aria-pressed', String(!state.expressive));
}

function setHint(text, isError = false) {
  els.hint.textContent = text;
  els.hint.classList.toggle('error', isError);
}

function renderCaptions() {
  els.captions.replaceChildren(
    ...[...state.captions.values()].slice(-CAPTION_LINES).map(({ who, text }) => {
      const line = document.createElement('div');
      line.className = `line ${who}`;
      const label = document.createElement('span');
      label.className = 'speaker';
      label.textContent = who === 'user' ? 'You' : 'Agent';
      line.append(label, document.createTextNode(text));
      return line;
    }),
  );
  els.captions.scrollTop = els.captions.scrollHeight;
}

// ---- mood from lk.expression ----

function applyExpression(raw) {
  try {
    const parsed = JSON.parse(raw);
    const expression = parsed.expression?.trim() || null;
    if (!expression && !parsed.mood) return false;
    state.mood = parsed.mood ?? null;
    state.expression = expression;
    state.turnsSinceMood = 0;
    render();
    return true;
  } catch {
    return false;
  }
}

function ageMood() {
  if (state.mood === null && state.expression === null) return;
  if (++state.turnsSinceMood >= MOOD_TTL_TURNS) {
    state.mood = null;
    state.expression = null;
    render();
  }
}

// ---- transcription streams ----

// A segment id is stable while a segment updates: the agent streams one delta
// stream per segment, and user STT opens a fresh stream per interim update.
// Keying by segment id makes updates replace their line instead of appending.
async function onTranscription(reader, participantInfo) {
  const room = state.room;
  if (!room) return;

  const who = participantInfo.identity === room.localParticipant.identity ? 'user' : 'agent';
  const id = reader.info.attributes?.[SEGMENT_ID_ATTRIBUTE] ?? reader.info.id;

  let text = '';
  for await (const chunk of reader) {
    if (state.room !== room) return; // stream from an ended session
    text += chunk;
    if (text.trim()) {
      state.captions.set(id, { who, text });
      renderCaptions();
    }
  }

  if (who !== 'agent') return;

  // The expression rides the stream's *closing* trailer, which livekit-client
  // merges into the info object we already hold, by mutation and after the
  // reader completes — so poll briefly for it to land.
  for (let ticks = 0; ticks < SETTLE_TICKS; ticks++) {
    const raw = reader.info.attributes?.[EXPRESSION_ATTRIBUTE];
    if (raw) {
      applyExpression(raw);
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, SETTLE_INTERVAL_MS));
  }
  ageMood();
}

// ---- agent audio ----

function watchAudioLevel(mediaStreamTrack) {
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(new MediaStream([mediaStreamTrack]));
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.6;
  source.connect(analyser);
  state.analyser = { ctx, analyser, data: new Uint8Array(analyser.frequencyBinCount) };
}

// ---- pipeline label from the agent's echoed attributes ----

function updatePipeline(participant) {
  const attrs = participant.attributes ?? {};
  if (attrs[AGENT_STATE_ATTRIBUTE]) {
    state.agentState = attrs[AGENT_STATE_ATTRIBUTE];
  }
  if (attrs.tts_label) {
    const mode = attrs.expressive === 'true' ? 'Expressive' : 'Flat';
    els.pipeline.textContent = `${mode} · ${attrs.tts_label}`;
  }
  render();
}

// ---- session lifecycle ----

async function connect() {
  if (state.connecting || state.room) return;
  state.connecting = true;
  setHint('Tell it some good news, or some bad news, and listen to how the delivery changes.');
  render();

  try {
    const res = await fetch('/api/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expressive: state.expressive, tts: state.tts }),
    });
    if (!res.ok) throw new Error(`token request failed (${res.status})`);
    const { server_url: serverUrl, participant_token: token } = await res.json();

    const room = new Room();
    // claim the slot before any await so no concurrent connect can double up
    state.room = room;
    room.registerTextStreamHandler(TRANSCRIPTION_TOPIC, onTranscription);

    room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
      if (track.kind !== Track.Kind.Audio) return;
      state.audioEl?.remove();
      state.audioEl = track.attach();
      document.body.appendChild(state.audioEl);
      watchAudioLevel(track.mediaStreamTrack);
      updatePipeline(participant);
    });
    room.on(RoomEvent.ParticipantAttributesChanged, (_changed, participant) => {
      if (participant.isLocal) return;
      updatePipeline(participant);
    });
    // a torn-down room emits Disconnected asynchronously; by then a new
    // session may own the slot, and it must not be torn down with it
    room.on(RoomEvent.Disconnected, () => {
      if (state.room === room) teardown();
    });

    await room.connect(serverUrl, token);
    if (state.room !== room) return; // ended while connecting

    state.agentState = 'listening';
    state.connecting = false;
    render();

    room.startAudio().catch(() => {});
    // don't block on the permission prompt; the session is already up
    room.localParticipant.setMicrophoneEnabled(true).catch(() => {
      els.mic.setAttribute('aria-pressed', 'true');
      setHint('Microphone unavailable — allow access, then tap the mic button.', true);
    });
  } catch (err) {
    teardown();
    setHint(err instanceof Error ? err.message : 'could not start the demo', true);
  } finally {
    state.connecting = false;
    render();
  }
}

function teardown() {
  const room = state.room;
  state.room = null;
  state.agentState = 'disconnected';
  state.mood = null;
  state.expression = null;
  state.turnsSinceMood = 0;
  state.captions.clear();
  state.audioEl?.remove();
  state.audioEl = null;
  state.analyser?.ctx.close().catch(() => {});
  state.analyser = null;
  els.pipeline.textContent = '';
  els.mic.setAttribute('aria-pressed', 'false');
  renderCaptions();
  render();
  room?.disconnect();
}

// both controls ride on dispatch metadata, so changing one restarts the agent
async function restart(next) {
  Object.assign(state, next);
  render();
  if (state.room) {
    teardown();
    await connect();
  }
}

// ---- wiring ----

els.connect.addEventListener('click', () => (state.room ? teardown() : connect()));
els.modeExpressive.addEventListener('click', () => restart({ expressive: true }));
els.modeFlat.addEventListener('click', () => restart({ expressive: false }));
els.tts.addEventListener('change', (event) => restart({ tts: event.target.value }));
els.mic.addEventListener('click', async () => {
  if (!state.room) return;
  const muted = els.mic.getAttribute('aria-pressed') === 'true';
  await state.room.localParticipant.setMicrophoneEnabled(muted);
  els.mic.setAttribute('aria-pressed', String(!muted));
});

render();
requestAnimationFrame(tick);
