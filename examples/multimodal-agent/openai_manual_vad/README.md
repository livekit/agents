# Push to Talk Example

This example demonstrates how to manually control the VAD of the OpenAI realtime agent.

## How It Works

1. The example runs a web server that serves a simple push-to-talk interface
2. Call `agent.interrupt()` when you press the button
3. Call `agent.commit_audio_buffer()` when you release the button
4. The web interface communicates with the agent through WebSocket

## Running the Example

1. Start the application:
   ```bash
   PORT=8080 python push_to_talk.py
   ```

2. Open the web interface:
   - If running locally: http://localhost:8080
   - If running on a server: http://<your-server-ip>:8080
