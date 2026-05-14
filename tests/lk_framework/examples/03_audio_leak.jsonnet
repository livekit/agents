// The load-bearing test: false EOU detection, user resumes the moment EOU
// fires. The agent's first response must be cancelled before any audio
// reaches the output sink.
//
// Proves: continuous-quantity assertions, pipeline-milestone anchors,
// the architecture handles cancellation correctness at sub-50ms precision.

local lk = import 'livekit/test.libsonnet';

lk.test('no audio leak on false EOU') {
  agent:   'examples.echo:EchoAgent',
  session: { endpointing: { min_delay: 0.25 } },
  clock:   'virtual',

  fakes: {
    llm: lk.llm.serial([
      { text: 'I should never be heard.', ttft_ms: 200, gen_ms: 800 },
      { text: 'Okay, go on.',             ttft_ms: 150, gen_ms: 400 },
    ]),
    tts: { audio_duration_per_char: 50, ttfb_ms: 300 },
    vad: { detection_latency_ms: 50 },
  },

  scenario: lk.user.timeline([
    lk.user.says('hello',       duration_ms=400, at='t=0'),
    lk.user.says('actually no', duration_ms=700, at='eou + 0ms'),
  ]),

  expect: {
    'messages[0]': {
      role:            'assistant',
      interrupted:     true,
      audio_played_ms: lk.ms.eq(0),       // load-bearing
    },
    'messages[1]': {
      role:            'assistant',
      interrupted:     false,
      audio_played_ms: lk.ms.gt(0),
    },
  },

  budgets: {
    llm_cancel_latency: lk.ms.lt(50),
    tts_cancel_latency: lk.ms.lt(50),
  },

  invariants: [
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.no_errors,
  ],
}
