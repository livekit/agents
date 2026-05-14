// End-of-turn timing: EOU fires exactly min_delay after the user stops
// speaking. The canonical test for audio_recognition.py:_run_eou_detection
// (extra_sleep = endpointing_delay, line 1101).

local lk = import 'livekit/test.libsonnet';

// ── Audio fixtures ────────────────────────────────────────────────────
local HELLO = lk.audio.fixture('hello_there');

// ── Timing model ──────────────────────────────────────────────────────
local MIN_DELAY = 250;   // configured endpointing min_delay
local MAX_DELAY = 3000;
local TOLERANCE = 5;     // virtual-clock floor for assertions

lk.test('end of turn fires at min_delay after speech stops') {
  agent: 'examples.echo:EchoAgent',

  session: {
    turn_detection: 'vad',
    endpointing:    { mode: 'fixed', min_delay_ms: MIN_DELAY, max_delay_ms: MAX_DELAY },
  },

  fakes: {
    llm: lk.llm.responses(['hello back']),
    stt: lk.stt.script([
      { events: [
        { type: 'interim', text: 'hello',       at_ms: 200 },
        { type: 'interim', text: 'hello there', at_ms: 500 },
        { type: 'final',   text: 'hello there', at_ms: 700 },
      ] },
    ]),
  },

  scenario: [
    lk.user.speaks(HELLO, at='t=0'),
  ],

  expect: {
    'eou.fired[0]':          { delay_ms: lk.ms.approx(MIN_DELAY, TOLERANCE) },
    'stt.final[0]':          { text: 'hello there' },
    'assistant_messages[0]': { interrupted: false },
  },

  budgets: { e2e: lk.ms.lt(2000) },

  invariants: [
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.user_state_machine_valid,
    lk.invariant.agent_state_machine_valid,
  ],
}
