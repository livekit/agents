// Property-style test: define no expectations, only invariants. Useful for
// fuzzing-style or stress runs where the scenario varies but the rules don't.
//
// Proves: flexibility — same framework, fundamentally different test style.

local lk = import 'livekit/test.libsonnet';

lk.test('invariants under three rapid turns') {
  agent: 'examples.echo:EchoAgent',

  fakes: {
    llm: lk.llm.responses(['ok one', 'ok two', 'ok three']),
  },

  scenario: lk.user.timeline([
    lk.user.says('one'),
    lk.user.says('two',   at='+1500ms'),
    lk.user.says('three', at='+3000ms'),
  ]),

  // No expect block. The contract is purely the invariants.
  invariants: [
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.user_state_machine_valid,
    lk.invariant.agent_state_machine_valid,
    lk.invariant.no_errors,
    lk.invariant.every_user_turn_has_response,
  ],
}
