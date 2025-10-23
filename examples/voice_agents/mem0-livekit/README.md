## [Mem0](https://mem0.ai/)

This guide demonstrates how to create a memory-enabled voice assistant using LiveKit, Deepgram, OpenAI, and Mem0, focusing on creating an intelligent, context-aware travel planning agent.

## Prerequisites

Before you begin, make sure you have:

1. Installed Livekit Agents SDK with voice dependencies of silero and deepgram:
```bash
pip install "livekit-agents[deepgram,openai,cartesia,silero,turn-detector]~=1.0" "livekit-plugins-noise-cancellation~=0.2" "python-dotenv"
```

2. Installed Mem0 SDK:
```bash
pip install mem0ai
```

3. Set up your API keys in a `.env` file:
```sh
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
DEEPGRAM_API_KEY=your_deepgram_api_key
MEM0_API_KEY=your_mem0_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Key Features of This Implementation

1. **Semantic Memory Retrieval**: Uses Mem0 to store and retrieve contextually relevant memories
2. **Voice Interaction**: Leverages LiveKit for voice communication with proper turn detection
3. **Intelligent Context Management**: Augments conversations with past interactions
4. **Travel Planning Specialization**: Focused on creating a helpful travel guide assistant
5. **Function Tools**: Modern tool definition for enhanced capabilities

## Running the Example

To run this example:

1. Install all required dependencies
2. Set up your `.env` file with the necessary API keys
3. Ensure your microphone and audio setup are configured
4. Run the script with Python 3.11 or newer and with the following command:
```sh
python mem0-livekit-voice-agent.py start
```
or to start your agent in console mode to run inside your terminal:

```sh
python mem0-livekit-voice-agent.py console
```
5. After the script starts, you can interact with the voice agent using [Livekit's Agent Platform](https://agents-playground.livekit.io/) and connect to the agent inorder to start conversations.

## Best Practices for Voice Agents with Memory

1. **Context Preservation**: Store enough context with each memory for effective retrieval
2. **Privacy Considerations**: Implement secure memory management
3. **Relevant Memory Filtering**: Use semantic search to retrieve only the most relevant memories
4. **Error Handling**: Implement robust error handling for memory operations

## Debugging Function Tools

- To run the script in debug mode simply start the assistant with `dev` mode:
```sh
python mem0-livekit-voice-agent.py dev
```

- When working with memory-enabled voice agents, use Python's `logging` module for effective debugging:

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_voice_agent")
```

- Check the logs for any issues with API keys, connectivity, or memory operations.
- Ensure your `.env` file is correctly configured and loaded.


## Help & Resources

- [LiveKit Documentation](https://docs.livekit.io/)
- [Mem0 Platform](https://app.mem0.ai/)
- Need assistance? Reach out through:

<Snippet file="get-help.mdx" />
