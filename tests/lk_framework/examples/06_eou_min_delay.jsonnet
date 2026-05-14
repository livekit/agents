// End-of-turn timing: EOU fires exactly min_delay after the user stops
// speaking. Tests audio_recognition.py:_run_eou_detection (line 1101)
// AND adjacent invariants — exact counts, prompt threading, no retries.

local lk = import 'livekit/test.libsonnet';

local HELLO = lk.audio.fixture('hello_there');

local MIN_DELAY = 250;
local TOLERANCE = 5;

lk.test('end of turn fires exactly min_delay after speech stops') {
  agent: 'examples.echo:EchoAgent',

  session: {
    turn_detection: 'vad',
    endpointing:    { mode: 'fixed', min_delay_ms: MIN_DELAY, max_delay_ms: 3000 },
  },

  fakes: {
    llm: lk.llm.responses(['hello back']),
    stt: lk.stt.script([
      { events: [
        { type: 'interim', text: 'hello',       at_ms: 200 },
        { type: 'final',   text: 'hello there', at_ms: 700 },
      ] },
    ]),
  },

  scenario: [ lk.user.speaks(HELLO, at='t=0') ],

  expect: {
    // Timing rule: EOU fires min_delay after vad.end_of_speech.
    'eou.fired[0]': { delay_ms: lk.ms.approx(MIN_DELAY, TOLERANCE) },

    // Cross-check against the trace itself, not the SDK's reported metric.
    // Catches bugs where the metric is computed wrong but consistently.
    'delta(vad.end_of_speech[0], eou.fired[0])':
      lk.ms.approx(MIN_DELAY, TOLERANCE),

    // Exactly one of everything — catches retries, dupes, leaked turns.
    'eou.fired':           { count: 1 },
    'llm.requested':       { count: 1 },
    'assistant_messages':  { count: 1 },

    // Prompt building: LLM received the final transcript as a user message.
    // Tests the STT → chat_ctx → LLM threading, which is non-trivial code.
    'llm.requested[0]': {
      chat_history: [{ role: 'user', text: 'hello there' }],
    },
  },

  invariants: [
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.user_state_machine_valid,
    lk.invariant.agent_state_machine_valid,
  ],
}
