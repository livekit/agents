// Barge-in: user starts speaking 1500ms into the agent's response.
// Tests adaptive interruption + cancellation propagation through the
// entire LLM → TTS → audio pipeline.

local lk = import 'livekit/test.libsonnet';

// ── Timing model ──────────────────────────────────────────────────────
local BARGE_AT          = 1500;  // user starts barging in (after agent.speaking)
local VAD_DETECT        = 50;    // FakeVAD start_of_speech detection latency
local CANCEL_TAIL_MAX   = 200;   // cumulative LLM+TTS+flush after barge-in
local MIN_INTERRUPT_DUR = 500;   // session.interruption.min_duration

// Derived: how much audio could legitimately have played
// Low bound: barge-in time minus the VAD latency before detection registers
// High bound: barge-in time plus the maximum cancel-tail allowed
local AUDIO_FLOOR = BARGE_AT - VAD_DETECT;        // 1450
local AUDIO_CEIL  = BARGE_AT + CANCEL_TAIL_MAX;   // 1700

lk.test('barge-in: user cuts agent off mid-speech, agent stops fast') {
  agent: 'examples.echo:EchoAgent',
  clock: 'virtual',

  session: {
    interruption: {
      mode:                 'adaptive',
      min_duration_ms:      MIN_INTERRUPT_DUR,
      backchannel_boundary: [0, 0],   // cooldowns off — deterministic
    },
  },

  fakes: {
    llm: lk.llm.responses([
      'I am about to give a very long and detailed explanation that takes time.',
      'Sure, what is your question?',
    ]),
    tts: { audio_duration_per_char_ms: 50, ttfb_ms: 300 },
  },

  scenario: lk.user.timeline([
    lk.user.says('tell me about yourself', duration_ms=900, at='t=0'),
    lk.user.says('actually, wait',         duration_ms=700,
                 at='agent.speaking + %dms' % BARGE_AT),
  ]),

  expect: {
    'overlap.detected[0]': { is_interruption: true },

    'messages[0]': {
      role:            'assistant',
      interrupted:     true,
      // Audio heard ≈ barge-in time, bounded by VAD latency and cancel tail
      audio_played_ms: lk.ms.between(AUDIO_FLOOR, AUDIO_CEIL),
    },

    'messages[1]': {
      role:        'assistant',
      interrupted: false,
      contains:    'what is your question',
    },
  },

  // Per-stage cancellation budgets — each stage of the pipeline
  budgets: {
    llm_cancel_latency:  lk.ms.lt(50),
    tts_cancel_latency:  lk.ms.lt(50),
    audio_flush_latency: lk.ms.lt(50),
  },

  invariants: [
    lk.invariant.no_errors,
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.agent_state_machine_valid,
  ],
}
