# Restaurant Voice Agent Example

This example demonstrates a sophisticated multi-agent voice system for a restaurant using LiveKit Agents. The system can handle customer interactions for reservations, takeaway orders, and checkout through natural voice conversations.

## Overview

The restaurant voice agent consists of four specialized agents that work together to provide a complete restaurant service experience:

- **Greeter Agent**: Welcomes customers and routes them to appropriate services
- **Reservation Agent**: Handles table reservation requests
- **Takeaway Agent**: Manages food orders for pickup/delivery
- **Checkout Agent**: Processes payments and completes transactions

## Features

- 🗣️ **Natural Voice Interactions**: Full voice-to-voice conversations using state-of-the-art STT, TTS, and LLM technologies
- 🤖 **Multi-Agent System**: Specialized agents with seamless handoffs
- 📋 **Order Management**: Complete order taking and modification capabilities
- 📅 **Reservation System**: Time-based reservation booking
- 💳 **Payment Processing**: Secure credit card information collection
- 🔄 **Context Preservation**: Maintains conversation history across agent transfers
- 📊 **Customer Data Tracking**: Persistent customer information throughout the session

## Prerequisites

Before running this example, ensure you have:

1. **Python 3.8+** installed
2. **LiveKit Cloud account** or self-hosted LiveKit server
3. **API Keys** for the following services:
   - OpenAI (for LLM)
   - Deepgram (for Speech-to-Text)
   - Cartesia (for Text-to-Speech)

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/livekit/agents.git
   cd agents/examples/voice_agents/restaurant_agent
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   
   The easiest way is to use the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can install dependencies individually:
   ```bash
   pip install livekit-agents[openai,cartesia,deepgram,silero]
   pip install python-dotenv
   pip install pyyaml
   ```

4. **Set up environment variables**:
   Create a `.env` file in the same directory with the following variables:
   ```env
   LIVEKIT_URL=wss://your-livekit-server.com
   LIVEKIT_API_KEY=your-api-key
   LIVEKIT_API_SECRET=your-api-secret
   
   OPENAI_API_KEY=your-openai-api-key
   DEEPGRAM_API_KEY=your-deepgram-api-key
   CARTESIA_API_KEY=your-cartesia-api-key
   ```

## Running the Example

1. **Start the agent**:
   ```bash
   python restaurant_agent.py dev
   ```

2. **Connect to the room**: You can connect using any LiveKit client SDK or telephony integration. To get started quickly, try the [Agents Playground](https://agents-playground.livekit.io/).

## How It Works

### Agent Architecture

The system uses a state-based approach where different agents handle specific tasks:

```
Customer Call → Greeter → [Reservation | Takeaway] → [Checkout] → Complete
```

### Agent Responsibilities

#### Greeter Agent
- **Purpose**: Initial customer contact and routing
- **Capabilities**: 
  - Welcomes customers
  - Understands customer intent (reservation vs. takeaway)
  - Routes to appropriate specialized agent

#### Reservation Agent
- **Purpose**: Handle table reservations
- **Capabilities**:
  - Collect reservation time
  - Gather customer name and phone number
  - Confirm reservation details

#### Takeaway Agent
- **Purpose**: Manage food orders
- **Capabilities**:
  - Present menu options
  - Take and modify orders
  - Handle special requests
  - Transfer to checkout when ready

#### Checkout Agent
- **Purpose**: Process payments
- **Capabilities**:
  - Calculate order totals
  - Collect customer information
  - Process credit card details (number, expiry, CVV)
  - Complete transactions

### Data Management

The system maintains customer data throughout the conversation using the `UserData` class:

```python
@dataclass
class UserData:
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    reservation_time: Optional[str] = None
    order: Optional[list[str]] = None
    customer_credit_card: Optional[str] = None
    customer_credit_card_expiry: Optional[str] = None
    customer_credit_card_cvv: Optional[str] = None
    expense: Optional[float] = None
    checked_out: Optional[bool] = None
```

### Function Tools

Each agent has access to specialized function tools:

- **Common Tools**: `update_name()`, `update_phone()`, `to_greeter()`
- **Reservation Tools**: `update_reservation_time()`, `confirm_reservation()`
- **Takeaway Tools**: `update_order()`, `to_checkout()`
- **Checkout Tools**: `confirm_expense()`, `update_credit_card()`, `confirm_checkout()`

## Configuration

### Menu Customization
Update the menu in the `entrypoint()` function:
```python
menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"
```

### Voice Customization
Modify voice IDs in the `voices` dictionary to use different Cartesia voices:
```python
voices = {
    "greeter": "794f9389-aac1-45b6-b726-9d9369183238",
    "reservation": "156fb8d2-335b-4950-9cb3-a2d33befec77",
    "takeaway": "6f84f4b8-58a2-430c-8b37-e00b72059fdd",
    "checkout": "39b376fc-488e-4d0c-8b37-e00b72059fdd",
}
```

### Realtime Model Option
To use OpenAI's realtime model instead of separate STT/TTS/LLM components, uncomment the following lines in the `entrypoint()` function:
```python
# llm=openai.realtime.RealtimeModel(voice="alloy"),
```


This example is part of the LiveKit Agents framework. Please refer to the main repository for license information. 