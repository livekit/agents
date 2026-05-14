// End-of-turn timing: EOU must fire exactly min_delay after the user stops
// speaking. This is the canonical timing test for the audio_recognition
// EOU detection path.

local lk = import 'livekit/test.libsonnet';

lk.test('end of turn fires at min_delay after speech stops') {
  agent: 'examples.echo:EchoAgent',
  clock: 'virtual',
  session: {
    turn_detection: 'vad',
    endpointing:    { mode: 'fixed', min_delay: 0.25, max_delay: 3.0 },
  },

  fakes: {
    llm: lk.llm.responses(['hello back']),
  },

  scenario: lk.user.says('hello there', duration_ms=600),

  expect: {
    // EOU fired exactly min_delay after vad.end_of_speech
    'eou.fired[0]': { delay_ms: lk.ms.approx(250, tolerance=5) },

    // STT captured the utterance
    'stt.final[0]': { text: 'hello there' },

    // Agent reply landed and was not interrupted
    'messages[-1]': { role: 'assistant', interrupted: false },
  },

  budgets: {
    eou_delay: lk.ms.approx(250, tolerance=5),
    e2e:       lk.ms.lt(2000),
  },

  invariants: [
    lk.invariant.no_errors,
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.user_state_machine_valid,
    lk.invariant.agent_state_machine_valid,
  ],
}
