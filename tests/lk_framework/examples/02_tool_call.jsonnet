// Tool call with prompt verification + timing budgets.
// Proves: real assertions (not tautological), partial-order expectations.

local lk = import 'livekit/test.libsonnet';

lk.test('weather lookup via tool') {
  agent:   'examples.weather:WeatherAgent',
  session: { endpointing: { min_delay: 0.25 } },

  fakes: {
    llm: lk.llm.serial([
      { tool_call: { lookup_weather: { location: 'San Francisco' } } },
      { text: 'It is sunny in San Francisco, 70 degrees.' },
    ]),
    tools: { lookup_weather: lk.tool.returns('sunny, 70F') },
  },

  scenario: lk.user.says('what is the weather in San Francisco'),

  // Each entry is a trace query, not a sequential event match.
  expect: {
    // The LLM received the right context (tests prompt building).
    'llm_calls[0]': {
      user_message_contains: 'weather in San Francisco',
      tools_offered:         ['lookup_weather'],
    },
    // The tool got the right args (tests intent interpretation).
    'tool_calls.lookup_weather': { args: { location: 'San Francisco' } },
    // The final assistant message references the tool output.
    'messages[-1]': { role: 'assistant', contains: 'sunny' },
  },

  budgets: {
    eou_delay: lk.ms.approx(250, tolerance=10),
    llm_ttft:  lk.ms.lt(800),
    e2e:       lk.ms.lt(2500),
  },

  invariants: [lk.invariant.no_errors],
}
