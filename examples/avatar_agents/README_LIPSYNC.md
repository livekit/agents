# Character-Level Alignment for Lip Sync

This document explains how to use ElevenLabs TTS with LiveKit to get character-level timing data for accurate lip sync animation in avatar applications.

## Overview

The LiveKit ElevenLabs plugin now supports forwarding character-level alignment metadata, which provides precise timing information for each character in synthesized speech. This enables:

- Accurate lip sync animation for avatars
- Visual text highlighting synchronized with speech
- Precise timing for subtitle display
- Character-level speech analysis

## Server-Side Implementation

### Basic Setup

```python
from livekit.plugins import elevenlabs

tts = elevenlabs.TTS(
    enable_alignment_forwarding=True,  # Enable alignment metadata
    sync_alignment=True,  # Required for alignment data
)

# Listen for alignment events
def on_alignment_received(event_data):
    alignment = event_data["alignment"]
    # alignment.characters: list of characters
    # alignment.start_times_seconds: start time for each character
    # alignment.end_times_seconds: end time for each character

tts.on("alignment_received", on_alignment_received)
```

### Publishing to Clients

```python
import json

def on_alignment_received(event_data):
    alignment = event_data["alignment"]

    # Prepare payload
    payload = {
        "type": "elevenlabs_alignment",
        "segment_id": alignment.segment_id,
        "characters": alignment.characters,
        "start_times_seconds": alignment.start_times_seconds,
        "end_times_seconds": alignment.end_times_seconds,
    }

    # Publish to all clients via data channel
    await room.local_participant.publish_data(
        payload=json.dumps(payload).encode("utf-8"),
        topic="tts.alignment",
    )
```

### Complete Example

See `elevenlabs_lipsync_example.py` in this directory for a complete working example.

## Client-Side Implementation

### JavaScript/TypeScript

```typescript
import { Room, RoomEvent } from 'livekit-client';

room.on(RoomEvent.DataReceived, (
  payload: Uint8Array,
  participant,
  kind,
  topic
) => {
  if (topic === 'tts.alignment') {
    const alignmentData = JSON.parse(
      new TextDecoder().decode(payload)
    );

    console.log('Alignment received:', {
      segment_id: alignmentData.segment_id,
      char_count: alignmentData.characters.length,
      duration: alignmentData.end_times_seconds[
        alignmentData.end_times_seconds.length - 1
      ]
    });

    // Use for lip sync animation
    animateLipSync(alignmentData);
  }
});
```

### Lip Sync Animation Example

```typescript
interface AlignmentData {
  segment_id: string;
  characters: string[];
  start_times_seconds: number[];
  end_times_seconds: number[];
}

function animateLipSync(alignment: AlignmentData) {
  const audioStartTime = performance.now();

  alignment.characters.forEach((char, index) => {
    const startMs = alignment.start_times_seconds[index] * 1000;
    const endMs = alignment.end_times_seconds[index] * 1000;
    const duration = endMs - startMs;

    setTimeout(() => {
      // Map character to viseme (mouth shape)
      const viseme = mapCharacterToViseme(char);

      // Update avatar mouth shape
      updateAvatarMouth(viseme, duration);
    }, startMs);
  });
}

function mapCharacterToViseme(char: string): string {
  // Map characters to viseme IDs based on phonetics
  const vowels = 'aeiouAEIOU';
  const bilabials = 'bpmBPM';
  const labiodentals = 'fvFV';

  if (vowels.includes(char)) return 'vowel';
  if (bilabials.includes(char)) return 'bilabial';
  if (labiodentals.includes(char)) return 'labiodental';

  return 'neutral';
}
```

### React Example

```tsx
import { useEffect, useState } from 'react';
import { useDataChannel } from '@livekit/components-react';

function LipSyncAvatar() {
  const [currentViseme, setCurrentViseme] = useState('neutral');

  // Subscribe to alignment data
  useDataChannel('tts.alignment', (message) => {
    const alignment = JSON.parse(
      new TextDecoder().decode(message.payload)
    );

    animateLipSync(alignment, setCurrentViseme);
  });

  return (
    <div className="avatar">
      <AvatarMouth viseme={currentViseme} />
    </div>
  );
}
```

## Data Format

### Alignment Payload Structure

```json
{
  "type": "elevenlabs_alignment",
  "segment_id": "abc123",
  "characters": ["H", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"],
  "start_times_seconds": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
  "end_times_seconds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
  "char_count": 11
}
```

### Timing Notes

- Times are relative to the start of the audio segment
- Times are in seconds (floating point)
- Spaces and punctuation are included in the character array
- Each character has a start and end time

## Running the Example

```bash
cd examples/avatar_agents
python elevenlabs_lipsync_example.py
```

Make sure you have:
- `ELEVEN_API_KEY` environment variable set
- `LIVEKIT_URL` and `LIVEKIT_API_KEY` configured
- Client application subscribed to the `tts.alignment` data topic

## Advanced Usage

### Filtering Alignment Data

```python
def on_alignment_received(event_data):
    alignment = event_data["alignment"]

    # Filter out spaces and punctuation for cleaner visualization
    filtered = {
        "characters": [],
        "start_times": [],
        "end_times": []
    }

    for i, char in enumerate(alignment.characters):
        if char.strip() and char.isalnum():
            filtered["characters"].append(char)
            filtered["start_times"].append(alignment.start_times_seconds[i])
            filtered["end_times"].append(alignment.end_times_seconds[i])

    # Publish filtered data
    publish_alignment(filtered)
```

### Synchronization Tips

1. **Audio Buffering**: Account for audio buffer delays in your client
2. **Network Latency**: Consider using NTP or server timestamps for sync
3. **Frame Rate**: Align animations with your rendering frame rate
4. **Interpolation**: Smooth transitions between visemes

## Troubleshooting

### No Alignment Data Received

- Ensure `sync_alignment=True` in TTS initialization
- Verify `enable_alignment_forwarding=True` is set
- Check that the ElevenLabs model supports alignment
- Verify client is subscribed to `tts.alignment` topic

### Timing Issues

- Check for audio playback delays
- Verify performance.now() or similar high-resolution timer
- Account for network latency
- Consider pre-buffering on the client

### Performance

- Alignment data is sent per segment (not per word)
- Data size is proportional to text length
- Consider throttling for very long texts

## API Reference

### TTS Configuration

```python
elevenlabs.TTS(
    enable_alignment_forwarding: bool = True,  # Enable alignment forwarding
    sync_alignment: bool = True,               # Enable alignment in API
    preferred_alignment: Literal["normalized", "original"] = "normalized"
)
```

### Event Data Structure

```python
event_data = {
    "context_id": str,
    "segment_id": str,
    "alignment": CharacterAlignment(
        characters: list[str],
        start_times_seconds: list[float],
        end_times_seconds: list[float],
        segment_id: str
    )
}
```

### CharacterAlignment Class

The `CharacterAlignment` dataclass is available from `livekit.agents.tts`:

```python
from livekit.agents.tts import CharacterAlignment

@dataclass
class CharacterAlignment:
    characters: list[str]
    """List of characters in the synthesized text"""
    start_times_seconds: list[float]
    """Start time of each character in seconds (relative to audio start)"""
    end_times_seconds: list[float]
    """End time of each character in seconds (relative to audio start)"""
    segment_id: str = ""
    """Segment ID this alignment belongs to"""
```

## Backward Compatibility

This feature is fully backward compatible:

- Default behavior: `enable_alignment_forwarding=True` (events are emitted)
- To disable: Set `enable_alignment_forwarding=False` in TTS initialization
- Existing code without alignment handlers will continue to work normally
- No breaking changes to existing TTS functionality

## See Also

- [ElevenLabs API Documentation](https://elevenlabs.io/docs)
- [LiveKit Data Channels](https://docs.livekit.io/realtime/client/data-messages/)
- [LiveKit Agents Guide](https://docs.livekit.io/agents/)
- [LiveKit Python SDK](https://docs.livekit.io/reference/python/)

