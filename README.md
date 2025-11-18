<h1>🎙️ LiveKit Interrupt Handler – Real-Time Voice Control</h1>

<p>
  This branch introduces a <strong>real-time interrupt handling layer</strong> for a LiveKit-based voice agent.
  The system intelligently differentiates between <strong>filler speech</strong> (e.g., “umm”, “uh”, “hmm”) and
  <strong>intentional user commands</strong> (e.g., “stop”, “wait”, “no, not that one”) while maintaining a
  smooth conversational experience.
</p>

<hr>

<h2>1. What Changed</h2>

<h3>1.1 New / Updated Modules</h3>

<ul>
  <li>
    <code>custom/agent_runner.py</code><br />
    <ul>
      <li>Defines the main <code>entrypoint</code> for the LiveKit worker.</li>
      <li>Constructs a complete voice pipeline using:
        <ul>
          <li><strong>Deepgram</strong> – Speech-to-Text (STT)</li>
          <li><strong>Groq Llama 3.1</strong> – LLM for text responses</li>
          <li><strong>Cartesia</strong> – Text-to-Speech (TTS)</li>
          <li><strong>Silero</strong> – Voice Activity Detection (VAD)</li>
        </ul>
      </li>
      <li>Registers event handlers for:
        <ul>
          <li><code>user_input_transcribed</code> – user speech (STT results)</li>
          <li><code>speech_created</code> – agent speech (TTS output)</li>
        </ul>
      </li>
      <li>Integrates the custom interrupt logic into the normal LiveKit agent flow.</li>
    </ul>
  </li>

  <li>
    <code>custom/interrupt_handler.py</code><br />
    <ul>
      <li>Implements an <strong>InterruptHandler</strong> with three decision outcomes:
        <code>IGNORE</code>, <code>INTERRUPT</code>, <code>NORMAL</code>.
      </li>
      <li>Uses confidence thresholding + filler detection + command detection to decide:
        <ul>
          <li>Whether to ignore low-value audio (fillers).</li>
          <li>Whether to treat speech as a genuine interruption.</li>
          <li>Whether to route speech as normal user input.</li>
        </ul>
      </li>
    </ul>
  </li>

  <li>
    <code>custom/filler_manager.py</code><br />
    <ul>
      <li>Provides a configurable, reloadable list of filler words/phrases.</li>
      <li>Supports dynamic reading from JSON:
        <code>config/fillers_default.json</code>.
      </li>
      <li>Designed to be extended for multilingual filler sets.</li>
    </ul>
  </li>

  <li>
    Configuration files:
    <ul>
      <li><code>config/fillers_default.json</code> – defines filler words (e.g., “umm”, “uh”, “hmm”).</li>
      <li><code>config/commands_default.json</code> – defines interrupt commands (e.g., “stop”, “wait”, “no, not that”).</li>
    </ul>
  </li>
</ul>

<hr>

<h2>2. What Works (Verified Behaviour)</h2>

<h3>2.1 Filler Suppression While Agent Is Speaking</h3>
<p>
  When the agent is actively speaking (TTS is playing), the interrupt layer:
</p>
<ul>
  <li><strong>Ignores</strong> filler-like utterances such as:
    <code>“umm”</code>, <code>“uh”</code>, <code>“hmm”</code>, etc.</li>
  <li>Ensures the agent is <strong>not interrupted</strong> by background hesitations.</li>
  <li>Maintains a smooth, uninterrupted TTS experience unless a real command is detected.</li>
</ul>

<h3>2.2 Real-Time User Interrupts</h3>
<p>
  If the user speaks a clear interrupt phrase while the agent is speaking, such as:
</p>
<ul>
  <li><code>“stop”</code></li>
  <li><code>“wait a second”</code></li>
  <li><code>“no, not that one”</code></li>
  <li><code>“hold on”</code></li>
</ul>
<p>
  then:
</p>
<ul>
  <li>The current TTS output is <strong>stopped immediately</strong> using <code>session.interrupt()</code>.</li>
  <li>The new user text is forwarded to the LLM.</li>
  <li>The agent responds to the most recent user intent instead of finishing the previous message.</li>
</ul>

<h3>2.3 Meaningful Speech Detection</h3>
<ul>
  <li>Any non-filler speech while the agent is speaking is treated as a valid interruption.</li>
  <li>Examples:
    <ul>
      <li>“Actually, change the topic.” → interrupt + respond.</li>
      <li>“No, talk about pricing instead.” → interrupt + respond.</li>
    </ul>
  </li>
</ul>

<h3>2.4 Behaviour When Agent Is Silent</h3>
<ul>
  <li>When the agent is not speaking, <strong>all speech</strong> (including fillers) is treated as normal user input.</li>
  <li>This ensures that the interrupt layer does not over-filter when the user is simply thinking aloud.</li>
</ul>

<hr>

<h2>3. Known Issues & Edge Cases</h2>

<ul>
  <li>
    <strong>STT Misrecognition:</strong> In very noisy environments, short commands like “stop” may be misheard
    (e.g., as “top”). In those cases, the command will not be recognized as an interrupt.
  </li>
  <li>
    <strong>Very Low-Confidence Speech:</strong> Extremely quiet or mumbled speech may be dropped by the
    confidence filter as noise or filler.
  </li>
  <li>
    <strong>Model/Provider Limits:</strong>
    <ul>
      <li>Groq free-tier rate limits may apply under continuous rapid firing.</li>
      <li>TTS or STT behaviour may vary slightly depending on the microphone hardware and network conditions.</li>
    </ul>
  </li>
</ul>

<hr>

<h2>4. Steps to Test</h2>

<h3>4.1 Prerequisites</h3>
<ul>
  <li>Python <strong>3.12+</strong></li>
  <li>Virtual environment created and activated.</li>
  <li>All required keys present in <code>.env</code> (see Environment section below).</li>
</ul>

<h3>4.2 Install Dependencies</h3>

<pre><code>pip install "livekit-agents[deepgram,cartesia,groq,silero,turn-detector]~=1.0"
pip install python-dotenv
</code></pre>

<h3>4.3 Start the Agent (Console Mode)</h3>

<pre><code>python custom/run_agent.py console
</code></pre>

<h3>4.4 Test Scenarios</h3>

<table>
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Agent Speaking?</th>
      <th>Example Utterance</th>
      <th>Expected Behaviour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Filler while agent is talking</td>
      <td>Yes</td>
      <td>“umm… uh… hmm… acha... aree... hanji... haan...”</td>
      <td>Agent continues speaking, fillers are <strong>ignored</strong>.</td>
    </tr>
    <tr>
      <td>Explicit interrupt command</td>
      <td>Yes</td>
      <td>“stop”, “wait”, “no, not that one”,"listen","hold on","ruk jao","ek second"</td>
      <td>Current TTS is <strong>stopped immediately</strong>, agent replies to new intent.</td>
    </tr>
    <tr>
      <td>Meaningful correction</td>
      <td>Yes</td>
      <td>“actually, change the topic”</td>
      <td>Agent is interrupted and responds to updated request.</td>
    </tr>
    <tr>
      <td>Normal question</td>
      <td>No</td>
      <td>“can you explain that again?”</td>
      <td>Processed as normal user input, agent responds.</td>
    </tr>
    <tr>
      <td>Fillers when agent is silent</td>
      <td>No</td>
      <td>“umm so what about pricing?”</td>
      <td>Entire phrase is treated as user input and processed.</td>
    </tr>
  </tbody>
</table>

<p>
  For detailed behaviour, logs are written to:
  <code>logs/interrupt.log</code>.
</p>

<hr>

<h2>5. Environment Details & Configuration</h2>

<h3>5.1 Python & Runtime</h3>
<ul>
  <li>Python: <strong>3.12+</strong></li>
  <li>OS: Tested on Windows environment with virtualenv.</li>
</ul>

<h3>5.2 Required Python Packages</h3>

<pre><code>livekit-agents[deepgram,cartesia,groq,silero,turn-detector]~=1.0
python-dotenv
</code></pre>

<h3>5.3 Environment Variables (.env)</h3>

<p><strong>Note:</strong> <code>.env</code> must <strong>not</strong> be committed to version control.</p>

<pre><code>LIVEKIT_URL=wss://&lt;your-livekit-cloud-url&gt;
LIVEKIT_API_KEY=&lt;your_livekit_api_key&gt;
LIVEKIT_API_SECRET=&lt;your_livekit_api_secret&gt;

DEEPGRAM_API_KEY=&lt;your_deepgram_key&gt;
GROQ_API_KEY=&lt;your_groq_key&gt;
CARTESIA_API_KEY=&lt;your_cartesia_key&gt;
</code></pre>

<h3>5.4 Configuration Files</h3>

<ul>
  <li>
    <code>config/fillers_default.json</code><br />
    Contains filler phrases to ignore while the agent is speaking, for example:
    <pre><code>[
  "um",
  "umm",
  "uh",
  "uhh",
  "hmm",
  "erm"
]</code></pre>
  </li>

  <li>
    <code>config/commands_default.json</code><br />
    Contains user phrases that should be treated as <strong>hard interrupts</strong>, for example:
    <pre><code>[
  "stop",
  "wait",
  "hold on",
  "no",
  "stop that",
  "stop now"
]</code></pre>
  </li>
</ul>

<hr>

<h2>6. Design Goals & Rationale</h2>

<ul>
  <li><strong>Non-invasive:</strong> No changes are made to LiveKit’s internal VAD or transport; all logic is layered on top.</li>
  <li><strong>Configurable:</strong> Filler and command lists are externalized to JSON to support different languages or domains.</li>
  <li><strong>Scalable:</strong> Event-driven, async-safe architecture using <code>asyncio.create_task</code> to avoid blocking any I/O loops.</li>
  <li><strong>Cost-Efficient:</strong> Uses free-tier providers (Deepgram, Groq, Cartesia) to keep experimentation and testing inexpensive.</li>
</ul>

<p>
  Overall, this feature turns the agent into a more <strong>natural, interruption-aware conversational system</strong>,
  suitable for demos, interviews, and real-world voice assistant scenarios.
</p>
