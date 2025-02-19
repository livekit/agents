# LiveKit Avatar Example

This example demonstrates how to create an animated avatar that responds to audio
input using LiveKit's agent system. The sample project comprises several components
that work together:

- **agent_worker.py**: Sets up the agent task, connects to a LiveKit room, and
  handles the connection handshake with the avatar dispatcher.
- **avatar_runner.py**: Processes incoming audio to generate synchronized video frames.
  It features a customizable video generator using OpenCV (for waveform visualization).
- **dispatcher.py**: Implements a FastAPI server that launches avatar worker processes
  on demand.
- **wave_viz.py**: Contains a helper class to render audio waveforms, providing real-time visual feedback.

## Overview

1. **Agent and Avatar Interaction**  

   The agent sends connection info (room name, token, and server URL) to the dispatcher,
   which then launches an avatar runner process for that room. The runner processes audio,
   generates corresponding video, and publishes both streams back to the room.

2. **Customization Points**  
   Developers can customize this example by:
   
   - Modifying the audio-to-video processing logic in `avatar_runner.py` (see the `MyVideoGenerator` class).
   - Changing agent instructions or connecting different language models, TTS, or STT engines in `agent_worker.py`.
   - Extending the FastAPI endpoints in `dispatcher.py` or integrating alternative worker launching strategies.

## Prerequisites

- **Python Version**: Python 3.10 or newer (for modern type hinting syntax)
- **Dependencies**:  
  Install requirements with:
  
  ```bash
  pip install -r examples/avatar/requirements.txt
  ```
  Make sure that additional packages (e.g., `livekit`, `httpx`, `python-dotenv`) are available (these may be provided by your LiveKit integration).

## Getting Started

1. **Start the Avatar Dispatcher Server**  
   Run the dispatcher to listen for avatar launch requests:
   
   ```bash
   python examples/avatar/dispatcher.py --port 8089
   ```

2. **Start the Agent Worker**  
   Use the agent worker to initialize the connection and send the avatar handshake:
   
   ```bash
   python examples/avatar/agent_worker.py dev --avatar-url http://localhost:8089/launch
   ```

3. **Customize Your Avatar**  
   - **Modify Visualization**: Edit `wave_viz.py` to change how audio waveforms are visualized.
   - **Update Media Options**: In `avatar_runner.py`, adjust the media options (resolution, FPS, etc.) in the `MediaOptions` instance.
   - **Extend Functionality**: Incorporate additional processing or integrations in the agent or worker code to suit your project's needs.

## File Structure

- **agent_worker.py**: Manages avatar agent connection and handshake.
- **avatar_runner.py**: Handles media processing and video generation.
- **dispatcher.py**: Hosts the FastAPI server to launch and monitor worker processes.
- **wave_viz.py**: Provides audio waveform visualizations using OpenCV.
- **requirements.txt**: Lists the Python dependencies for the avatar sample.

Happy coding and feel free to extend this example for your own animated avatar projects!
