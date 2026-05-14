// End-of-turn timing: EOU fires exactly min_delay after the user stops
// speaking. The canonical test for audio_recognition.py:_run_eou_detection
// (extra_sleep = endpointing_delay, line 1101).

local lk = import 'livekit/test.libsonnet';

// ── Timing model (all values in milliseconds) ─────────────────────────
local MIN_DELAY    = 250;   // configured endpointing min_delay
local MAX_DELAY    = 3000;
local SPEECH_DUR   = 600;   // how long the user speaks
local TOLERANCE    = 5;     // virtual-clock floor for assertions

lk.test('end of turn fires at min_delay after speech stops') {
  agent: 'examples.echo:EchoAgent',
  clock: 'virtual',

  session: {
    turn_detection: 'vad',
    endpointing:    { mode: 'fixed', min_delay_ms: MIN_DELAY, max_delay_ms: MAX_DELAY },
  },

  fakes: { llm: lk.llm.responses(['hello back']) },

  scenario: lk.user.says('hello there', duration_ms=SPEECH_DUR),

  expect: {
    // EOU fires exactly MIN_DELAY after vad.end_of_speech, ± clock floor
    'eou.fired[0]': { delay_ms: lk.ms.approx(MIN_DELAY, tolerance=TOLERANCE) },

    'stt.final[0]': { text: 'hello there' },
    'messages[-1]': { role: 'assistant', interrupted: false },
  },

  budgets: {
    eou_delay: lk.ms.approx(MIN_DELAY, tolerance=TOLERANCE),
    e2e:       lk.ms.lt(2000),
  },

  invariants: [
    lk.invariant.no_errors,
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.user_state_machine_valid,
    lk.invariant.agent_state_machine_valid,
  ],
}
