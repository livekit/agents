Filler Word Interrupt Handler Agent
Author: Guttula Viswa Venkata Yashwanth

Overview
This project implements an intelligent LiveKit voice agent that prevents unnecessary interruptions caused by common filler words (like “uh”, “umm”, “oh”, etc.) spoken by the user while the agent is talking. If the agent is interrupted with only filler words, it will immediately resume speaking. Real user commands (like "wait", "stop", or meaningful phrases) will correctly interrupt the agent.

Key Features
Robust filtering of filler words during agent speech (English and Hindi supported).

Ignores interruptions on filler words, causing the agent to seamlessly resume its output.

Tracks both agent speech generation and playback for maximum accuracy.

Confidence thresholding to control sensitivity.

Easy configuration and extension of filler word list.

Real-time statistics reporting of valid vs. ignored interruptions.

File Structure
File	Purpose
filler_interrupt_agent.py	The main LiveKit agent file with all event handling and agent logic
filler_filter.py	Implements the filler word filtering logic
README.md	This documentation
Setup Instructions
Clone your repository and enter your project directory.

Install dependencies:

bash
uv sync
Set your API keys in .env.local:

Get free API keys from Deepgram, Groq, and Cartesia.

Example contents:

text
LIVEKIT_URL=wss://your-livekit-url.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_secret
DEEPGRAM_API_KEY=your_deepgram_api_key
GROQ_API_KEY=your_groq_api_key
CARTESIA_API_KEY=your_cartesia_api_key
Export environment variables:

bash
export $(cat .env.local | xargs)
Run the agent in dev mode:

bash
uv run python filler_interrupt_agent.py dev
Usage
Connect to the agent via agents-playground.livekit.io or a compatible client.

Talk to the agent using your browser’s microphone.

Say filler words while the agent is speaking—agent should ignore these and keep talking without noticeable interruption.

Say a meaningful word or phrase (e.g., "stop", "what is the weather") to interrupt the agent and get a response.

Customization
Add/Remove filler words:
Edit the ignored_filler_words list in filler_interrupt_agent.py.

Adjust confidence threshold:
Edit the value of confidence_threshold when creating FillerInterruptionFilter.

Review real-time statistics in the agent’s terminal window.

Testing
Recommended scenarios:

Scenario	Expected Result
User says "umm" while agent speaking	Agent resumes, ignores utterance
User says "stop" while agent speaking	Agent stops and responds
User says "umm" while agent silent	Agent responds to the user
User says "uh can you help" while agent speaks	Agent stops (meaningful content)
Terminal Log Indicators:

🚫 IGNORED FILLER: Filler word detected and agent kept talking.

✅ VALID SPEECH: Real interruption detected and agent stopped.

Advanced Notes
Both agent speech generation and audio playback are tracked for best possible timing.

Interruptions are only ignored if all spoken words are fillers and the confidence is above the specified threshold.

If you encounter edge cases where the agent pauses incorrectly, expand the filler word list or lower the confidence threshold.

All event handling is robustly async-safe, using the correct LiveKit handler pattern.

License
This implementation is for the SalesCode.AI LiveKit Challenge (2025).

