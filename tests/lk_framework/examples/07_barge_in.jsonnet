// Barge-in: user starts speaking 1500ms into the agent's response.
// Tests classification, cancellation propagation, audio-leak bounds,
// AND that the chat context correctly threads the barge-in turn through
// to the next LLM call.

local lk = import 'livekit/test.libsonnet';

local QUESTION = lk.audio.fixture('tell_me_about_yourself');
local BARGE    = lk.audio.fixture('actually_wait');

local BARGE_AT          = 1500;
local VAD_DETECT        = 50;
local CANCEL_TAIL_MAX   = 200;
local MIN_INTERRUPT_DUR = 500;

local AUDIO_FLOOR = BARGE_AT - VAD_DETECT;        // 1450
local AUDIO_CEIL  = BARGE_AT + CANCEL_TAIL_MAX;   // 1700

lk.test('barge-in: user cuts agent off mid-speech') {
  agent: 'examples.echo:EchoAgent',

  session: {
    interruption: { min_duration_ms: MIN_INTERRUPT_DUR, backchannel_boundary: [0, 0] },
  },

  fakes: {
    llm: lk.llm.responses(['long explanation here.', 'follow up reply.']),
    stt: lk.stt.script([
      { events: [{ type: 'final', text: 'tell me about yourself', at_ms: 900 }] },
      { events: [{ type: 'final', text: 'actually, wait',         at_ms: 600 }] },
    ]),
  },

  scenario: [
    lk.user.speaks(QUESTION, at='t=0'),
    lk.user.speaks(BARGE,    at='agent.speaking + %dms' % BARGE_AT),
  ],

  expect: {
    // Classification: barge-in is a real interruption (not a backchannel).
    'overlap.detected':    { count: 1 },
    'overlap.detected[0]': { is_interruption: true },

    // First message marked interrupted; audio leak bounded by VAD detect
    // latency below and cancel-tail budget above.
    'assistant_messages[0]': {
      interrupted:     true,
      audio_played_ms: lk.ms.between(AUDIO_FLOOR, AUDIO_CEIL),
    },

    // Second LLM call must see the full conversation in chat context:
    // the interrupted assistant turn must appear, the barge-in must appear
    // as a new user message. This tests the prompt-building path that
    // most production bugs hide in.
    'llm.requested': { count: 2 },
    'llm.requested[1]': {
      chat_history: [
        { role: 'user',      text:    'tell me about yourself' },
        { role: 'assistant', text:    'long explanation here.', interrupted: true },
        { role: 'user',      text:    'actually, wait' },
      ],
    },

    // Second message actually played — confirms the cancellation didn't
    // also kill the *follow-up* response (a real bug class).
    'assistant_messages[1]': {
      interrupted:     false,
      audio_played_ms: lk.ms.gt(0),
    },

    // Negative assertion: no false interruption fired for what is a real one.
    'agent.false_interruption': { count: 0 },
  },

  budgets: {
    llm_cancel_latency:  lk.ms.lt(50),
    tts_cancel_latency:  lk.ms.lt(50),
    audio_flush_latency: lk.ms.lt(50),
  },

  invariants: [
    lk.invariant.no_audio_while_user_speaking,
    lk.invariant.agent_state_machine_valid,
  ],
}
