# LiveKit Plugins Minimal

This is a minimal example of a LiveKit plugin for Agents.

### Developer note

When copying this directory over to create a new `livekit-plugins` package, make sure it's nested within the `livekit-plugins` folder and that the `"name"` field in `package.json` follows the proper naming convention for CI:

```json
{
  "name": "livekit-plugins-<name>",
  "private": true
}
```
