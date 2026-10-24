# Drive-Thru Example

A complete drive-thru ordering system demonstrating interactive voice agents for food ordering with database integration and order management.

For setup instructions and more details, see the [main examples README](../README.md).

## Overview

This example simulates a fast food drive-thru. It is split across three files: `database.py` contains the menu and formats it as system prompt text, `order.py` holds Pydantic models for the three order types, and `agent.py` defines `DriveThruAgent` with dynamically built ordering tools.

The full menu is loaded once per session and injected directly into the agent's instructions, so the LLM has menu context without needing to call a tool.

### Menu Loading

At the start of each session, `new_userdata()` queries `FakeDB` for all item categories (drinks, combos, Happy Meals, regulars, sauces) and stores them in the `Userdata` dataclass alongside a fresh `OrderState`.
https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/drive-thru/agent.py#L382-L399
`DriveThruAgent.__init__` then formats each category using `menu_instructions()` and concatenates the results with `COMMON_INSTRUCTIONS` to build the full system prompt. This means the LLM sees the entire menu from the first turn and can answer questions or suggest items without any tool calls.
https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/drive-thru/agent.py#L55-L83
### Dynamic Tool Building

The three ordering tools are constructed by `build_combo_order_tool`, `build_happy_order_tool`, and `build_regular_order_tool`. Each method closes over the relevant item lists and injects their IDs as the `enum` constraint in the tool's JSON schema.

https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/drive-thru/agent.py#L85-L119

This restricts the LLM to known IDs at the schema layer before any runtime logic runs. `ToolError` handles the cases that can't be caught statically — for example, when a drink has multiple available sizes and the customer hasn't specified one yet, the tool raises a `ToolError` prompting the agent to ask for clarification before retrying.

### Order Types

`order.py` defines three Pydantic models: `OrderedCombo`, `OrderedHappy`, and `OrderedRegular` . A discriminated union `OrderedItem` is also defined. Each ordered item receives a random short `order_id` on creation via `order_uid()`. 

`OrderState` stores the current cart as a `dict[str, OrderedItem]` keyed by `order_id`, which the `remove_order_item` and `list_order_items` tools use to look up or modify existing items.
https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/drive-thru/order.py#L45-L56

### Managing the Order

Two tools handle cart management:

- `list_order_items` returns all current cart items with their `order_id`s. The agent is instructed to call this first when modifying or removing an item whose `order_id` is unknown.
- `remove_order_item` removes one or more items by `order_id`. Modifications (e.g., upsizing fries) are done by removing the old item and re-adding it with the new parameters.

`max_tool_steps=10` is set on the session to give the agent enough budget to call `list_order_items` followed by `remove_order_item` in a single turn when needed.

### Background Audio

`BackgroundAudioPlayer` plays an ambient drive-thru noise track (`bg_noise.mp3`) throughout the session to set the scene. 
https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/drive-thru/agent.py#L438-L443

### STT Tuning

The STT model is also initialized with `keyterm` hints for McDonald's brand names (e.g., `"Big Mac"`, `"McFlurry"`, `"McCrispy"`) to improve transcription accuracy.
https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/drive-thru/agent.py#L415-L430
