// Barge-in: user starts speaking 1500ms into the agent's response.
// Tests adaptive interruption + cancellation propagation through the
// entire LLM → TTS → audio pipeline.

local lk = import 'livekit/test.libsonnet';

// ── Audio fixtures ────────────────────────────────────────────────────
local QUESTION = lk.audio.fixture('tell_me_about_yourself');
local BARGE    = lk.audio.fixture('actually_wait');

// ── Timing model ──────────────────────────────────────────────────────
local BARGE_AT          = 1500;  // user starts barging in (after agent.speaking)
local VAD_DETECT        = 50;    // FakeVAD start_of_speech detection latency
local CANCEL_TAIL_MAX   = 200;   // cumulative LLM+TTS+flush after barge-in
local MIN_INTERRUPT_DUR = 500;   // session.interruption.min_duration

// Derived: audio_played bounds — barge-in time ± pipeline jitter
local AUDIO_FLOOR = BARGE_AT - VAD_DETECT;        // 1450
local AUDIO_CEIL  = BARGE_AT + CANCEL_TAIL_MAX;   // 1700

lk.test('barge-in: user cuts agent off mid-speech, agent stops fast') {
  agent: 'examples.echo:EchoAgent',

  session: {
    interruption: {
      min_duration_ms:      MIN_INTERRUPT_DUR,
      backchannel_boundary: [0, 0],   // cooldowns off — deterministic
    },
  },

  fakes: {
    llm: lk.llm.responses([
      'I am about to give a very long and detailed explanation that takes time.',
      'Sure, what is your question?',
    ]),
    stt: lk.stt.script([
      { events: [
        { type: 'interim', text: 'tell me',                at_ms: 300 },
        { type: 'interim', text: 'tell me about yourself', at_ms: 700 },
        { type: 'final',   text: 'tell me about yourself', at_ms: 900 },
      ] },
      { events: [
        { type: 'interim', text: 'actually',       at_ms: 200 },
        { type: 'final',   text: 'actually, wait', at_ms: 600 },
      ] },
    ]),
  },

  scenario: [
    lk.user.speaks(QUESTION, at='t=0'),
    lk.user.speaks(BARGE,    at='agent.speaking + %dms' % BARGE_AT),
  ],

  expect: {
    'overlap.detected[0]': { is_interruption: true },

    'assistant_messages[0]': {
      interrupted:     true,
      audio_played_ms: lk.ms.between(AUDIO_FLOOR, AUDIO_CEIL),
    },

    'assistant_messages[1]': {
      interrupted: false,
      contains:    'what is your question',
    },
  },

  budgets: {
    llm_cancel_latency:  lk.ms.lt(50),
    tts_cancel_latency:  lk.ms.lt(50),
    audio_flush_latency: lk.ms.lt(50),
  },

  invariants: [
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.agent_state_machine_valid,
  ],
}
