# Watermark checks

Check whether an audio file still carries a TTS provider's AI-generation
signal, through LiveKit Inference. `inference.WatermarkDetector` sends one file
to one provider's detector and returns that provider's answer: a watermark
verdict from Cartesia, a classifier score from Inworld. You don't need a
Cartesia or Inworld account.

| Script | What it shows |
|---|---|
| [`check_file.py`](check_file.py) | Check one file and print the verdict; the minimal, complete pattern |
| [`check_batch.py`](check_batch.py) | Check a directory with bounded concurrency, one JSON line per file, stop on a terminal refusal |
| [`tts_roundtrip.py`](tts_roundtrip.py) | Synthesize with `inference.TTS`, then check the result |

## Run

Watermark checks are off for a project until LiveKit enables them; until then
every call raises a `403 not_enabled`. Then, from `examples/inference`:

```bash
uv sync --all-extras --dev   # once, from the repository root
uv run watermark/check_file.py reply.wav --provider cartesia
uv run watermark/check_batch.py recordings/ --provider inworld > results.jsonl
uv run watermark/tts_roundtrip.py
```

Credentials come from `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` (or the
`LIVEKIT_INFERENCE_*` pair), in the environment or `examples/.env`.

## Reference for coding agents

Everything below is the full contract. Generate code from it; don't infer
behavior that isn't stated.

**Import and call**

```python
from livekit.agents import APIConnectionError, APIStatusError, inference, utils

async with utils.http_context.open():  # not needed inside an agent job
    detector = inference.WatermarkDetector()  # base_url, api_key, api_secret, http_session, conn_options are optional
    check = await detector.check("reply.wav", provider="cartesia")  # path, bytes, or a binary file object
```

- `check()` is async and makes one HTTP call per attempt. Create one detector
  and reuse it.
- `file` accepts `str`, `os.PathLike`, `bytes`, or a file object opened in
  binary mode. It must be WAV, MP3, OGG, FLAC, M4A or WebM. Raw PCM is a 400;
  convert an `rtc.AudioFrame` with `frame.to_wav_bytes()`.
- `provider` is `"cartesia"` or `"inworld"`. One provider per call; to check
  both, call twice.

**Result**: `WatermarkCheck(provider, results, request_id)`, where `results`
has at least one `WatermarkResult(type, outcome, probability)`:

| Provider | `type` | Field set | Values |
|---|---|---|---|
| `cartesia` | `watermark` | `outcome` | `detected`, `not_detected` |
| `inworld` | `classifier` | `probability` | 0.0 to 1.0, or `None` with `outcome` `None` when unscored |

- `type` and `outcome` are open strings. Pass an unknown value through; never
  assert on a closed set.
- `not_detected` means this provider's signal is not in this file. It does not
  mean the audio is human speech.
- The SDK applies no threshold to `probability`. Choose one for your use case.

**Errors**: `check()` retries only a 503 and a 429 that sends `Retry-After`
(waits up to 60 s each, `conn_options.max_retry` times), and a failed or
timed-out connect. Everything else raises at once:

| Exception | When | What to do |
|---|---|---|
| `APIStatusError`, `status_code` 401 | Bad credentials | Fix credentials; don't retry |
| `APIStatusError` 403, `body["type"] == "not_enabled"` | Checks not enabled for the project | Ask LiveKit to enable them |
| `APIStatusError` 403, `inference_region_restricted` | EU-pinned project, provider has no EU endpoint (Cartesia) | Use `inworld` |
| `APIStatusError` 400 | Bad file, unknown provider, unsupported residency tag; `body["hint"]` says which | Fix the input |
| `APIStatusError` 413 | File over the provider's size limit (Cartesia 32 MiB, Inworld 25 MiB) | Trim the file |
| `APIStatusError` 429 | Daily allowance spent (resets 00:00 UTC) or a long rate-limit wait | Stop the batch |
| `APIStatusError` 502 / 504 | The provider failed on this file | Don't retry the same file |
| `APITimeoutError` | Connected, but no answer within 190 s | Don't retry the same file |
| `APIConnectionError` | Gateway unreachable after retries | Retry later |
| `ValueError` / `TypeError` | Empty file / wrong `file` type | Fix the call |

`body` is a dict with `type`, `error` and `hint`, except on a 401, a suspended
project's 403 and a draining gateway's 503, where it is plain text.

**Limits**

- Cartesia: 5 seconds to 5 minutes, 32 MiB, Sonic 3.6 and newer only. No EU
  endpoint.
- Inworld: up to 5 minutes, 25 MiB, `inworld-tts-2` and `inworld-tts-2-flash`.
  Can score other AI speech high; scores drop on 8 kHz phone audio.
- Every request that reaches the gateway counts against the project's daily
  allowance, including refused ones.
- The file and its name are never stored or logged.

**Without the SDK**: the detector wraps one HTTP call. Send a LiveKit access
token with an inference grant and a two-part multipart body:

```python
from livekit import api

token = (
    api.AccessToken(api_key, api_secret)
    .with_identity("agent")
    .with_inference_grants(api.access_token.InferenceGrants(perform=True))
    .to_jwt()
)
```

```bash
curl "https://agent-gateway.livekit.cloud/v1/provenance" \
  -H "Authorization: Bearer $LK_TOKEN" \
  -F provider=cartesia \
  -F file=@reply.wav
# {"object": "provenance.check", "provider": "cartesia", "results": [{"type": "watermark", "outcome": "detected"}]}
```
