// Smallest possible test: user says hello, agent replies.
// Proves: easy to use. 5 lines of meaningful content.

local lk = import 'livekit/test.libsonnet';

lk.test('greeting') {
  agent:    'examples.echo:EchoAgent',
  fakes:    { llm: lk.llm.responses(['hello back']) },
  scenario: lk.user.says('hello'),
  expect:   { 'messages[-1]': { role: 'assistant', contains: 'hello back' } },
}
