// Backchannel: short utterance during agent speech that should NOT be
// classified as an interruption. The adaptive detector marks it
// is_interruption=false; after false_interruption_timeout the agent
// resumes; the original message completes normally.

local lk = import 'livekit/test.libsonnet';

lk.test('backchannel does not interrupt: agent resumes after false-interrupt timeout') {
  agent: 'examples.echo:EchoAgent',
  clock: 'virtual',
  session: {
    interruption: {
      mode:                       'adaptive',
      min_duration:               0.5,
      resume_false_interruption:  true,
      false_interruption_timeout: 1.5,
      backchannel_boundary:       [0, 0],
    },
  },

  fakes: {
    llm: lk.llm.responses([
      'Once upon a time there was a small village by a river. ' +
      'The villagers lived peacefully for many years.',
    ]),
    tts: { audio_duration_per_char: 50, ttfb_ms: 300 },
  },

  scenario: lk.user.timeline([
    lk.user.says('tell me a story', duration_ms=700, at='t=0'),

    // Short "mm-hmm" — below min_duration. Should be a backchannel.
    lk.user.says('mm-hmm',          duration_ms=300,
                 at='agent.speaking + 2000ms'),
  ]),

  expect: {
    // Overlap was detected but classified as NOT an interruption
    'overlap.detected[0]': { is_interruption: false },

    // False interruption event fires with resumed=true after timeout
    'agent.false_interruption[0]': {
      resumed:          true,
      after_overlap_ms: lk.ms.approx(1500, tolerance=100),
    },

    // The original assistant message completes normally — NOT interrupted
    'messages[0]': {
      role:        'assistant',
      interrupted: false,
      contains:    'peacefully for many years',
    },

    // The held STT transcript for the backchannel was suppressed; only one
    // final transcript made it through (the original user request).
    'stt.final': { count: lk.count.eq(1) },
  },

  budgets: {
    // Resume timing aligns with false_interruption_timeout
    false_interruption_resume: lk.ms.approx(1500, tolerance=100),
  },

  invariants: [
    lk.invariant.no_errors,
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.exactly_one_assistant_reply,
  ],
}
