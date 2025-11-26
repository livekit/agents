# IVR Navigation Agent Example

This example demonstrates how to build a Voice AI agent that can navigate IVR (Interactive Voice Response) systems. 

The goal is to show how an AI agent can act as a human caller: listening to menu prompts (e.g., "Press 1 for accounts..."), interpreting them, and sending DTMF tones (keypad presses) to navigate the tree and extract information or perform actions.

## Overview

The example consists of three parts:

1.  **The Navigator (`ivr_navigator_agent.py`)**: The AI agent that calls the bank. It listens to the IVR, interprets the prompts, and sends DTMF tones to navigate. **This is the main focus of the example.**
2.  **The Mock Bank (`ivr_system_agent.py`)**: A simulated banking IVR system. In a real application, this would be an actual phone number (like an airline or pharmacy). We provide this mock so you can test the navigator against a known system.
3.  **The Dialer (`dial_bank_agent.py`)**: A script that connects the two. It dispatches the Navigator agent and has it dial the Mock Bank via SIP.

## File Tour

-   **`ivr_navigator_agent.py`**: The agent logic for navigating the IVR. It receives a user intent (e.g., "Check my balance") and uses an LLM to determine which DTMF digits to press based on what it hears.
-   **`ivr_system_agent.py`**: The mock IVR system. It plays prompts and waits for DTMF input. It uses `MockBankService` to simulate real banking data.
-   **`dial_bank_agent.py`**: A helper script to start the flow. It creates a dispatch for the navigator agent and triggers an outbound SIP call to the IVR.
-   **`mock_bank_service.py`**: Serves read-only banking data from `data.json`.
-   **`data.json`**: Sample dataset with customers, balances, and transaction history.

## How it Works

1.  **Dispatch**: `dial_bank_agent.py` creates a job for the `ivr_navigator_agent.py` and passes a user request (e.g., "Check my checking balance") as metadata.
2.  **Call**: The script places a SIP call to the `ivr_system_agent.py`.
3.  **Navigation**:
    *   The **Bank (System)** answers and says: "Welcome to Horizon Bank. Press 1 for..."
    *   The **Navigator (Agent)** listens, consults its LLM with the user's request, and decides to press '1'.
    *   The Navigator sends the DTMF tone.
4.  **Result**: The interaction continues until the Navigator retrieves the information or completes the task, then it hangs up and logs the result.

## Run It Yourself

### Prerequisites
You need a LiveKit configured SIP trunk to allow the agents to receive inbound calls and make outbound calls. See the [LiveKit telephony integration guide](https://docs.livekit.io/agents/start/telephony/) for instructions on setting up SIP trunks.

1.  **Verify the dataset**
    Edit `examples/bank-ivr/data.json` if you want to customize the mock banking data.

2.  **Start the Mock Bank (The Target)**
    ```bash
    uv run python examples/bank-ivr/ivr_system_agent.py dev
    ```
    This agent acts as the IVR system waiting for calls.

3.  **Start the Navigator Agent (The Caller)**
    Open a new terminal. This agent will wait for a dispatch job to tell it to call.
    ```bash
    uv run python examples/bank-ivr/ivr_navigator_agent.py dev
    ```

4.  **Trigger the Call**
    Open a third terminal. This script tells the Navigator to call the Bank with a specific goal.
    ```bash
    uv run python examples/bank-ivr/dial_bank_agent.py --phone "+1234567890" --request "check balance for all accounts I have"
    ```
    *Note: Replace the phone number with the number that routes to your `ivr_system_agent.py` via your SIP setup.*

    The `--request` flag sets the goal for the Navigator. The script automatically appends the demo customer ID and PIN to the request so the agent can authenticate.

## Customizing the Navigator

To adapt this for a real-world use case (like calling an airline):
1.  Modify `ivr_navigator_agent.py`.
2.  Update the system prompt to describe the new persona and goal.
3.  The `send_dtmf_events` tool is the primary way the agent interacts with the world.

## Sample Scenarios

You can test different paths in the IVR by changing the `--request` passed to `dial_bank_agent.py`:

*   **Checking Balance**: `"Summarize jordan carter checking account"`
*   **Loan Status**: `"What are riley martinez loan obligations"`
*   **Credit Card Rewards**: `"Give me the platinum travel rewards card details for jordan carter"`

The navigator will interpret these natural language requests and navigate the menus accordingly.
