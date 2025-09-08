# Text-to-Speech Examples

These examples demonstrate real-time text-to-speech generation using various TTS plugins with LiveKit.

## Environment Variables

### Plugin API Keys
Set the API key for your chosen plugin.

### LiveKit Connection
For connecting to LiveKit Cloud:
- `LIVEKIT_URL` - Your LiveKit server URL
- `LIVEKIT_API_KEY` - LiveKit API key
- `LIVEKIT_API_SECRET` - LiveKit API secret

## Running Examples

Execute the example to connect to a LiveKit room and stream TTS audio:

```bash
uv run examples/other/text-to-speech/{your_plugin}_tts.py start
```

The agent will join the room and stream synthesized speech to participants.

### Running Locally

Running the examples with `console` mode won't play audio since the examples use `rtc.LocalAudioTrack`, which requires the LiveKit room infrastructure for audio playback. The `LocalAudioTrack` is designed to publish audio streams to LiveKit rooms where they are processed and distributed to participants. Without a room connection, the audio frames are generated but not routed to any playback device.

To test TTS output locally without a LiveKit room, you would need to modify the example file to save the generated audio frames to a WAV file instead of publishing them to a track. The saved WAV file can then be played using any audio player on your system.
