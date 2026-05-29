# Inference

A minimal voice agent powered end-to-end by [LiveKit Inference](https://docs.livekit.io/agents/models/inference.md). The playground exposes STT, LLM, and TTS pickers that swap the corresponding component live via RPC, so you can hear how each provider feels in the same conversation without restarting the session.

## How the live swap works

The playground sends an RPC on every dropdown change:

```ts
room.localParticipant.performRpc({
  destinationIdentity: agent.identity,
  method: "set_stt_model" | "set_llm_model" | "set_tts_model",
  payload: JSON.stringify({ value: "deepgram/nova-3" }),
});
```

The agent registers one handler per control. All three (STT, LLM, TTS) call `update_options(model=...)` to swap the active model without restarting the session.

## Run locally

```bash
pip install -r requirements.txt
python agent.py dev
```

The model list shown in the playground is sourced from `examples/playground.yaml`. To add or remove options, edit the `controls` block on the `inference` example there.
