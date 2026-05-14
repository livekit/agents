// Backchannel: short utterance during agent speech that should NOT be
// classified as an interruption. After false_interruption_timeout the
// agent resumes; the original message completes normally; the
// held-transcript machinery suppresses the backchannel transcript so
// only one final reaches chat context.

local lk = import 'livekit/test.libsonnet';

local STORY_REQ = lk.audio.fixture('tell_me_a_story');
local MM_HMM    = lk.audio.fixture('mm_hmm');

local BACKCHANNEL_AT    = 2000;
local FALSE_INT_TIMEOUT = 1500;
local MIN_INTERRUPT_DUR = 500;

lk.test('backchannel: agent resumes, no interruption') {
  agent: 'examples.echo:EchoAgent',

  session: {
    interruption: {
      min_duration_ms:               MIN_INTERRUPT_DUR,
      false_interruption_timeout_ms: FALSE_INT_TIMEOUT,
      backchannel_boundary:          [0, 0],
    },
  },

  fakes: {
    llm: lk.llm.responses(['Once upon a time there was a small village.']),
    stt: lk.stt.script([
      { events: [
        { type: 'interim', text: 'tell',            at_ms: 200 },
        { type: 'final',   text: 'tell me a story', at_ms: 800 },
      ] },
      // Second segment intentionally absent.
    ]),
  },

  scenario: [
    lk.user.speaks(STORY_REQ, at='t=0'),
    lk.user.speaks(MM_HMM,    at='agent.speaking + %dms' % BACKCHANNEL_AT),
  ],

  expect: {
    // VAD detected both audio segments — real signal-level evidence the
    // backchannel reached the SDK, despite STT yielding nothing for it.
    'vad.start_of_speech': { count: 2 },

    // Classifier correctly rejected the second segment as a backchannel.
    'overlap.detected':    { count: 1 },
    'overlap.detected[0]': { is_interruption: false },

    // False-interrupt fired with resumed=true at the expected timing.
    'agent.false_interruption':    { count: 1 },
    'agent.false_interruption[0]': {
      resumed:          true,
      after_overlap_ms: lk.ms.approx(FALSE_INT_TIMEOUT, 100),
    },

    // Original message completes — not interrupted, and audio actually
    // finished playing (catches a class of bugs where the resume marks
    // the flag right but the audio playback was already aborted).
    'assistant_messages':    { count: 1 },
    'assistant_messages[0]': {
      interrupted:     false,
      audio_played_ms: lk.ms.gt(1000),
    },

    // Held-transcript machinery suppressed the backchannel: 2 VAD segments
    // but only 1 final transcript reached the chat context.
    'stt.final':     { count: 1 },

    // No second LLM call — agent didn't generate a response for the backchannel.
    'llm.requested': { count: 1 },
  },

  invariants: [
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.exactly_one_assistant_reply,
  ],
}
