# LiveKit Agents Examples

This folder contains a number of working examples for features and functionality of the LiveKit Agents framework. These examples can also be used to test the framework during development.

For even more examples, see the [LiveKit Agents Examples](https://github.com/livekit-examples/python-agents-examples) repository. For more information on the framework, see the [LiveKit Agents documentation](https://docs.livekit.io/agents/).

## Running the examples

To run the examples, you'll need:

- A [LiveKit Cloud](https://cloud.livekit.io) account or a local [LiveKit server](https://github.com/livekit/livekit)
- API keys for the model providers you want to use in a `.env` file
- A virtual environment (e.g. `venv`)

From the `examples` directory, create your `venv`:

```bash
cd examples
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Then install the local workspace with all plugins:

```bash
pip install -e ../livekit-agents
pip install -e ../livekit-plugins/livekit-plugins-*
```

Create a `.env` file in the `examples` directory and add your API keys:

```env
LIVEKIT_URL="wss://your-project.livekit.cloud"
LIVEKIT_API_KEY="your_api_key"
LIVEKIT_API_SECRET="your_api_secret"
OPENAI_API_KEY="sk-xxx" # or any other model provider API key
... # other model provider API keys as needed
```

Run an example agent:

```bash
python voice_agents/basic_agent.py console
```

Your agent is now running in the console. 

For frontend support, see the [Agents playground](https://docs.livekit.io/agents/start/playground/) and the [starter apps](https://docs.livekit.io/agents/start/frontend/#starter-apps).