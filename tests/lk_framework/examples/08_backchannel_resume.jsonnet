// Backchannel: short utterance during agent speech that should NOT be
// classified as an interruption. After false_interruption_timeout the
// agent resumes and the original message completes normally.

local lk = import 'livekit/test.libsonnet';

// ── Timing model ──────────────────────────────────────────────────────
local BACKCHANNEL_AT        = 2000;  // backchannel starts (after agent.speaking)
local BACKCHANNEL_DUR       = 300;   // duration — below MIN_INTERRUPT_DUR
local MIN_INTERRUPT_DUR     = 500;   // threshold for "real" interruption
local FALSE_INT_TIMEOUT     = 1500;  // wait after overlap end to resume
local TOLERANCE             = 100;   // tolerance for timeout assertions

// Derived: BACKCHANNEL_DUR < MIN_INTERRUPT_DUR → classified as backchannel.
// Resume fires FALSE_INT_TIMEOUT after the overlap ends.

lk.test('backchannel does not interrupt: agent resumes after timeout') {
  agent: 'examples.echo:EchoAgent',
  clock: 'virtual',

  session: {
    interruption: {
      mode:                           'adaptive',
      min_duration_ms:                MIN_INTERRUPT_DUR,
      resume_false_interruption:      true,
      false_interruption_timeout_ms:  FALSE_INT_TIMEOUT,
      backchannel_boundary:           [0, 0],
    },
  },

  fakes: {
    llm: lk.llm.responses([
      'Once upon a time there was a small village by a river. ' +
      'The villagers lived peacefully for many years.',
    ]),
    tts: { audio_duration_per_char_ms: 50, ttfb_ms: 300 },
  },

  scenario: lk.user.timeline([
    lk.user.says('tell me a story', duration_ms=700, at='t=0'),
    lk.user.says('mm-hmm',          duration_ms=BACKCHANNEL_DUR,
                 at='agent.speaking + %dms' % BACKCHANNEL_AT),
  ]),

  expect: {
    // Duration (300ms) < threshold (500ms) → not an interruption
    'overlap.detected[0]': { is_interruption: false },

    // Resume fires exactly FALSE_INT_TIMEOUT after overlap ends
    'agent.false_interruption[0]': {
      resumed:          true,
      after_overlap_ms: lk.ms.approx(FALSE_INT_TIMEOUT, tolerance=TOLERANCE),
    },

    // Original message completes normally — never marked interrupted
    'messages[0]': {
      role:        'assistant',
      interrupted: false,
      contains:    'peacefully for many years',
    },

    // Held-transcript machinery suppressed the backchannel transcript
    'stt.final': { count: lk.count.eq(1) },
  },

  budgets: {
    false_interruption_resume:
      lk.ms.approx(FALSE_INT_TIMEOUT, tolerance=TOLERANCE),
  },

  invariants: [
    lk.invariant.no_errors,
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.exactly_one_assistant_reply,
  ],
}
