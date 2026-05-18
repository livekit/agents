# Inference

A minimal voice agent powered end-to-end by [LiveKit Inference](https://docs.livekit.io/agents/models/inference.md). The playground exposes STT, LLM, and TTS pickers that swap the corresponding component live via RPC, so you can hear how each provider feels in the same conversation without restarting the session.

## How the live swap works

The playground sends an RPC on every dropdown change:

```ts
room.localParticipant.performRpc({
  destinationIdentity: agent.identity,
  method: "stt" | "llm" | "tts",
  payload: JSON.stringify({ value: "deepgram/nova-3" }),
});
```

The agent registers one handler per control. STT and TTS call `update_options(model=...)`; the inference LLM doesn't expose `update_options` so the agent mutates `session.llm._opts.model`, which is read on the next `chat()` call.

## Run locally

```bash
pip install -r requirements.txt
python agent.py dev
```

The model list shown in the playground is sourced from `examples/playground.yaml`. To add or remove options, edit the `controls` block on the `inference` example there.
