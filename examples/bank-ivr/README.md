# Bank IVR Example

This example reenvisions the LiveKit telecom demo as a retail banking assistant. It combines a scripted IVR tree, a DTMF helper agent, and a mock banking backend so you can experiment with full call flows—authentication, menu navigation, and data lookups—without touching production systems.

## What's Included

- Read-only banking data served by `MockBankService`, covering deposits, credit cards, loans, and rewards.
- A multi-agent IVR implemented with LiveKit Agents. The main `agent.py` orchestrates authentication, menu routing, and task execution.
- A DTMF-focused assistant (`dtmf_agent.py`) that reads the caller's intent from dispatch metadata and presses keypad digits on the caller's behalf.
- A dialer script (`dial_bank_agent.py`) that provisions a LiveKit dispatch, embeds the user's request, and places an outbound SIP call.

## File Tour

- `agent.py` - primary IVR worker. Hosts the authentication flow, menu prompts, and per-domain tasks.
- `dtmf_agent.py` - companion agent dedicated to DTMF entry. Pulls the request from `ctx.room.metadata`, drives the IVR with keypad events, and records the outcome.
- `dial_bank_agent.py` - command-line helper that triggers a dispatch and SIP call. Accepts `--phone` and `--request` so you can enqueue different intents.
- `mock_bank_service.py` - loads structured customer data from `data.json` into immutable dataclasses for safe read access.
- `data.json` - sample dataset featuring two customers with realistic balances, transactions, cards, loans, and rewards.
- `test_mock_bank_service.py` - pytest coverage for the data service and formatting helpers.

## How the Flow Works

1. `dial_bank_agent.py` creates a dispatch targeting `PHONE_TREE_AGENT_DISPATCH_NAME`, attaches the user's request as metadata, and dials the IVR through your LiveKit SIP trunk.
2. `agent.py` answers the call, authenticates the customer, and consults `MockBankService` to fulfill each IVR menu option.
3. `dtmf_agent.py` receives the same dispatch, sees the requested task in the metadata, sends keypad digits via `send_dtmf_events`, and logs progress using `record_task_result_and_hang_up`.
4. When the task finishes, the DTMF agent records a summary and gracefully ends the session, ensuring the call does not linger.

## IVR Flow

```mermaid
graph TD
    R[RootBankIVRAgent<br/>Authenticate & Main Menu]
    R --> A1((Collect Customer ID))
    A1 --> A2((Collect PIN))
    A2 --> RM[(Main Menu)]

    RM --> M1[1 · Deposit Accounts]
    RM --> M2[2 · Credit Cards]
    RM --> M3[3 · Loans & Mortgages]
    RM --> M4[4 · Rewards & Benefits]
    RM --> M5[5 · Switch Profile]

    subgraph "Deposit Accounts Task"
        M1 --> DA1[Balances]
        M1 --> DA2[Available Cash]
        M1 --> DA3[Recent Transactions]
        M1 --> DA4[Total Deposits]
        M1 --> DA9[9 · Return]
    end

    subgraph "Credit Cards Task"
        M2 --> CC1[Statement & Payment]
        M2 --> CC2[Rewards Rates]
        M2 --> CC3[Total Balances]
        M2 --> CC9[9 · Return]
    end

    subgraph "Loans Task"
        M3 --> LN1[Outstanding Balances]
        M3 --> LN2[Upcoming Payments]
        M3 --> LN3[Autopay Status]
        M3 --> LN9[9 · Return]
    end

    subgraph "Rewards Task"
        M4 --> RW1[Tier & Points]
        M4 --> RW2[Cashback]
        M4 --> RW3[Expiring Points]
        M4 --> RW9[9 · Return]
    end

    M5 --> SW((Re-Authenticate))
    SW --> RM
```

## Run It Yourself

1. **Verify the dataset** - edit `examples/bank-ivr/data.json` if you need different customers.
2. **Run the unit tests** - `uv run pytest examples/bank-ivr/test_mock_bank_service.py` confirms the mock service and helpers are healthy.
3. **Start the IVR worker** - `uv run python examples/bank-ivr/agent.py dev` hosts the IVR and webhook.
4. **Start the DTMF assistant** - open another terminal and run `uv run python examples/bank-ivr/dtmf_agent.py` so keypad commands are available.
5. **Place the outbound call** - `uv run python examples/bank-ivr/dial_bank_agent.py --phone "+12132896618" --request "check balance for all accounts I have"`. Both flags are optional; defaults dial the demo number and request a balance check. The request string lives in dispatch metadata and becomes visible to both agents.

Set `SIP_OUTBOUND_TRUNK_ID` to a valid LiveKit SIP trunk identifier before running the dialer. Without it the script logs an error and exits early.

## Customize the Demo
- Add more customer profiles or transactions in `data.json` and extend `MockBankService` if you need new fields.
- Dial the IVR to different tasks by passing a different `--request` to the dialer.
- Update the instructions inside `dtmf_agent.py` to pursue new flows or invoke additional tools.
- Extend `agent.py` with extra submenu options or automation (for example, bill payment or card activation) to explore advanced routing.

## Sample Requests & Ground Truth

Use these suggested `--request` values to validate the agent's responses. Each listing includes the expected numbers straight from `data.json` so you can compare transcripts against ground truth.

1. **"Summarize jordan carter checking account"**
   - Account `031890246` (Checking) with balance **$4,821.37** and available balance **$4,615.92**.
   - Recent transactions:
     • 2025-10-08 — PAYROLL DEP - HORIZON TECH — **+$3,250.00**
     • 2025-10-05 — ZELLE TO ALEX R — **-$120.45**
     • 2025-10-04 — COFFEE ROASTERS — **-$5.85**

2. **"What are riley martinez loan obligations"**
   - Auto Loan `AUTO-22901`: outstanding balance **$18,642.77**, next payment due **2025-10-10**, monthly payment **$415.17**, autopay **disabled**.
   - Private Student Loan `STUDENT-00218`: outstanding balance **$19,880.43**, next payment due **2025-10-28**, monthly payment **$290.10**, autopay **enabled**.

3. **"Give me the platinum travel rewards card details for jordan carter"**
   - Card `4485 1399 2211 0099` (Platinum Travel Rewards).
   - Statement balance **$2,150.76**, minimum due **$68.00**, payment due date **2025-10-18**.
   - Rewards earn rate **3x travel, 2x dining**.
   - Jordan Carter's rewards summary: tier **Platinum**, points **138,940**, cashback available **$182.55**, points expiring next statement **4,000**.

