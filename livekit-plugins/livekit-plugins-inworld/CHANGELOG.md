# livekit-plugins-inworld

## Unreleased

- Add `delivery_mode` parameter for `inworld-tts-2` output variation control.
  Accepts `"DELIVERY_MODE_UNSPECIFIED"`, `"STABLE"`, `"BALANCED"`, or `"CREATIVE"`.
  Wired through both the WebSocket streaming path (sent as `deliveryMode` on the
  `create` packet) and the HTTP streaming path (top-level `deliveryMode` on the
  request body). Note: Inworld ignores `temperature` on `inworld-tts-2`; use
  `delivery_mode` to steer output variation on that model instead.
