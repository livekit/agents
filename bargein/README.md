# Barge in Example

## Usage

- First, update the example to use your own model path (replacing `/Users/chenghao/Downloads/bd_best.onnx`)
- Config .env file with your credentials

```txt
LIVEKIT_URL=""
LIVEKIT_API_KEY=""
LIVEKIT_API_SECRET=""

CARTESIA_API_KEY=""
```

- Finally, you can start the console session with the barge-in model running:

```bash
uv run bargein/bargein_example.py console
```