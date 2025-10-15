# Horizon Bank IVR Example

This example mirrors the telecom IVR demo but shifts the narrative to a retail banking assistant. It showcases:

- A `MockBankService` that exposes read-only customer data for checking/savings accounts, credit cards, loans, and rewards.
- A multi-agent IVR tree where the root agent authenticates callers and hands off to submenu tasks implemented with `AgentTask` for each banking domain.
- Voice-ready prompts and DTMF collection using the LiveKit agents toolkit.

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

Run the mock service tests with:

```bash
uv run pytest examples/bank-ivr/test_mock_bank_service.py
```

Launch the IVR worker with:

```bash
uv run python examples/bank-ivr/agent.py dev
```

