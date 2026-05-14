// Parametrized matrix: sweep cancel delay across 6 timings, assert how
// much audio leaks at each point.
//
// Proves: composition, jsonnet's value over flat YAML. One source generates
// six tests; each case is one tunable parameter.

local lk = import 'livekit/test.libsonnet';

lk['test.parametrize']('audio leakage vs cancel delay after EOU', {
  cases: [
    { name: 'instant',       delay_ms: 0,    leak: lk.ms.eq(0) },
    { name: 'pre_llm',       delay_ms: 100,  leak: lk.ms.eq(0) },
    { name: 'mid_llm',       delay_ms: 300,  leak: lk.ms.eq(0) },
    { name: 'tts_first_byte', delay_ms: 500,  leak: lk.ms.lt(20) },
    { name: 'first_chunk',   delay_ms: 800,  leak: lk.ms.lt(100) },
    { name: 'mid_playback',  delay_ms: 1500, leak: lk.ms.between(400, 800) },
  ],
}, function(c) {
  agent:   'examples.echo:EchoAgent',
  session: { endpointing: { min_delay: 0.25 } },
  clock:   'virtual',

  fakes: {
    llm: lk.llm.serial([
      { text: 'aborted response', ttft_ms: 200, gen_ms: 800 },
      { text: 'follow up',        ttft_ms: 150, gen_ms: 400 },
    ]),
    tts: { audio_duration: 4.0, ttfb_ms: 300 },
  },

  scenario: lk.user.timeline([
    lk.user.says('hello',         at='t=0'),
    lk.user.says('wait actually', at='eou + %dms' % c.delay_ms),
  ]),

  expect: {
    'messages[0]': { interrupted: true,  audio_played_ms: c.leak },
    'messages[1]': { interrupted: false },
  },

  budgets: {
    llm_cancel_latency: lk.ms.lt(50),
    tts_cancel_latency: lk.ms.lt(50),
  },
})
