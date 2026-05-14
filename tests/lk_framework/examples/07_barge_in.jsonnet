// Barge-in: user starts speaking 1500ms into the agent's response. The
// adaptive interruption detector should classify this as a true
// interruption, the agent must stop within budget, and audio played should
// be bounded — only what was heard before the cut.

local lk = import 'livekit/test.libsonnet';

lk.test('barge-in: user cuts agent off mid-speech, agent stops fast') {
  agent: 'examples.echo:EchoAgent',
  clock: 'virtual',
  session: {
    interruption: {
      mode:                 'adaptive',
      min_duration:         0.5,
      backchannel_boundary: [0, 0],   // cooldowns off — deterministic
    },
  },

  fakes: {
    llm: lk.llm.responses([
      'I am about to give a very long and detailed explanation that takes time.',
      'Sure, what is your question?',
    ]),
    tts: { audio_duration_per_char: 50, ttfb_ms: 300 },
  },

  scenario: lk.user.timeline([
    lk.user.says('tell me about yourself', duration_ms=900, at='t=0'),

    // User barges in 1500ms after the agent starts speaking
    lk.user.says('actually, wait',         duration_ms=700,
                 at='agent.speaking + 1500ms'),
  ]),

  expect: {
    // Adaptive detector classified the overlap as a real interruption
    'overlap.detected[0]': { is_interruption: true },

    // First message is marked interrupted; audio played is bounded — only
    // what was heard before VAD+cancel propagation cut the playback.
    'messages[0]': {
      role:            'assistant',
      interrupted:     true,
      audio_played_ms: lk.ms.between(1200, 1800),
    },

    // Second message comes through normally
    'messages[1]': {
      role:        'assistant',
      interrupted: false,
      contains:    'what is your question',
    },
  },

  budgets: {
    // Cancellation propagates across the pipeline within tight budgets
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
