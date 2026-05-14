// Backchannel: short utterance during agent speech that should NOT be
// classified as an interruption. After false_interruption_timeout the
// agent resumes and the original message completes normally.

local lk = import 'livekit/test.libsonnet';

// ── Audio fixtures ────────────────────────────────────────────────────
local STORY_REQ = lk.audio.fixture('tell_me_a_story');
local MM_HMM    = lk.audio.fixture('mm_hmm');

// ── Timing model ──────────────────────────────────────────────────────
local BACKCHANNEL_AT    = 2000;  // backchannel starts (after agent.speaking)
local FALSE_INT_TIMEOUT = 1500;  // wait after overlap end to resume
local MIN_INTERRUPT_DUR = 500;   // threshold for "real" interruption

lk.test('backchannel: agent resumes, no interruption') {
  agent: 'examples.echo:EchoAgent',

  session: {
    interruption: {
      min_duration_ms:               MIN_INTERRUPT_DUR,
      false_interruption_timeout_ms: FALSE_INT_TIMEOUT,
      backchannel_boundary:          [0, 0],   // override default (1.0, 3.5)
    },
  },

  fakes: {
    llm: lk.llm.responses(['Once upon a time there was a small village.']),
    // Second segment has no entry — FakeSTT mirrors real STT yielding
    // nothing on an unintelligible chunk.
    stt: lk.stt.script([
      { events: [
        { type: 'interim', text: 'tell',            at_ms: 200 },
        { type: 'interim', text: 'tell me a story', at_ms: 600 },
        { type: 'final',   text: 'tell me a story', at_ms: 800 },
      ] },
    ]),
  },

  scenario: [
    lk.user.speaks(STORY_REQ, at='t=0'),
    lk.user.speaks(MM_HMM,    at='agent.speaking + %dms' % BACKCHANNEL_AT),
  ],

  expect: {
    'vad.start_of_speech':         { count: 2 },
    'overlap.detected[0]':         { is_interruption: false },
    'agent.false_interruption[0]': { resumed: true,
                                     after_overlap_ms: lk.ms.approx(FALSE_INT_TIMEOUT, 100) },
    'assistant_messages[0]':       { interrupted: false, contains: 'small village' },
    'stt.final':                   { count: 1 },
  },

  invariants: [
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.exactly_one_assistant_reply,
  ],
}
