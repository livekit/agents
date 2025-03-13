# Avatar Examples

Avatars provide a visual representation for agents. This collection showcases three different technical approaches to avatar implementation:

1. **audio_wave** - A simple local mock demo that visualizes audio input as waveforms
2. **simli** - Integration with the Simli API for avatar generation
3. **bithuman** - Direct integration of a local avatar model with the agent worker

## Integration Approaches

These examples demonstrate two primary methods for integrating avatars with agent systems:

### 1. Data Stream Sink Integration

Used in **audio_wave** and **simli** examples, this approach:
- Runs the avatar in a separate process or server that joins the same LiveKit room as a participant
- Receives agent audio output through DataStream
- Processes audio and publishes synchronized video back to the room
- Enables distributed architecture where avatar generation can run on dedicated hardware

The data stream sink approach allows for more flexible deployment options, as the avatar generation can be:
- Hosted on specialized hardware optimized for visual processing
- Scaled independently from the agent worker
- Deployed closer to the rendering endpoint to reduce latency

### 2. Direct Agent Worker Integration

Used in the **bithuman** and **simli/integrated_agent_worker.py** example, this approach:
- Integrates avatar inference directly within the agent worker process
- Uses a queue-based audio sink to buffer audio for avatar generation
- Embeds the avatar model/API calls in the same runtime environment as the agent
- Simplifies deployment by reducing the number of separate services

The direct agent worker integration works for scenarios where:
- The avatar generation is already an API call and the agent worker has low-latency network connectivity to the avatar server
- The avatar generation model is lightweight enough to run alongside the agent without significant resource contention
- Simplified deployment is preferred with fewer separate services to manage

Both integration methods leverage the AvatarRunner SDK to handle the functionality of:
- Passthrough audio input from the agent
- Publishing synchronized audio and video streams to the LiveKit room
- Handling agent interruption and audio playout finished notification


## Examples

### audio_wave

The audio_wave example demonstrates a basic local implementation that visualizes audio input as waveforms. This approach:
- Provides a lightweight visualization that runs locally
- Functions without requiring external services
- Processes all visualization locally with minimal latency

### simli

The simli example shows how to integrate with the [Simli API](https://docs.simli.com/introduction) for avatar generation. This approach:
- Utilizes a third-party service for avatar rendering
- Implements API request/response handling
- Requires network connectivity and API authentication

### bithuman

The bithuman example demonstrates direct integration of a [Bithuman local runtime](https://docs.bithuman.io/api-reference/runtime/introduction) with the agent worker. This approach:
- Runs the avatar generation model locally alongside the agent
- Operates independently of external services
- Implements direct model inference within the agent workflow


## Contributing

Feel free to contribute additional avatar examples or technical improvements to existing ones by submitting a pull request.
