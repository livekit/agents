# BitHuman Avatar Example

This example demonstrates how to integrate the BitHuman SDK with LiveKit Agents to create an interactive visual agent on local CPU device.

## Prerequisites

1. BitHuman API Secret
2. BitHuman Avatar Model (.imx file)

## Setup Instructions

1. Get API Secret from [bitHuman website](https://bithuman.io)
2. Download Avatar Model (.imx file). You can use a sample model as below
or create new models on the [platform](https://console.bithuman.io/imagineX) 
    ```bash
    wget https://repo.one.bithuman.io/resources/rt_models/samples/albert_einstein.imx
    ```
3. Install Dependencies
    ```bash
    pip install bithuman
    ```

### 4. Configuration

Create a `.env` file in the root directory with the following:

```
BITHUMAN_API_SECRET=your_api_secret_here
BITHUMAN_MODEL_PATH=/path/to/model.imx
```

## Running the Example

To run the agent with a BitHuman avatar (the first time loading on MacOS may take a while for warmup):

```bash
python examples/avatar_agents/bithuman/agent_worker.py dev
```

## How It Works

This example integrates BitHuman directly within the agent worker process:
- Audio from the agent is routed to the BitHuman runtime
- The BitHuman SDK processes the audio to generate realistic animations
- Synchronized audio and video are published to the LiveKit room

For more information about BitHuman SDK, refer to the [official documentation](https://docs.bithuman.io/api-reference/sdk/quick-start).
