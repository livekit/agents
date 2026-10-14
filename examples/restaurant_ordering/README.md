# Restaurant Ordering Example

A pizzeria's phone line. Rosa answers for Zappi's Pizzeria: she takes pickup and delivery orders with sizes, sauces, toppings, and things left off, answers allergy questions from the menu data, steers around sold-out items, reads the whole menu when asked, offers breadsticks once, and reads back the total before the order goes to the kitchen. On the outbound call she rings a customer whose delivery is already on its way to say an item ran out, and sorts out the replacement or refund.

For setup instructions and more details, see the [main examples README](../README.md).

## Overview

The example is split so that the rules never depend on the model following its prompt:

- `backend.py` is the mock point of sale: the menu (with today's sold-out Buffalo cauliflower), the cart, pricing with tax and the delivery fee, the order it submits, and, for the outbound call, order #214, already out with the driver. Every rule lives here and refuses with a `PosError` whose message tells the agent what to do instead.
- `agent.py` defines `RosaAgent` (inbound) and `DeliveryUpdateAgent` (outbound), their tools, and the entrypoint. Each tool wraps one POS call; a refusal goes back to the LLM as a `ToolError` to relay.
- `call.py` holds what any phone agent here needs: the speaking rules, sentence pacing for Cartesia, background noise and voice cancellation (BVC for browser callers, BVCTelephony on phone calls), and hanging up or handing off without cutting the caller off.
- `grade.py` scores a simulation run from the POS's final state.

### Taking an order

Rosa adds each item with `add_item` as soon as it's named and asks only for what's missing: a pizza's size, a wing size and sauce, a drink size. Names are matched loosely ("a large", "hot sauce", "meat lover's pizza", "cokes"), and "no olives" is taken off the pizza at the same price.

The POS enforces the rest:

- People repeat themselves. An item already in the cart can't be added again, and a count only goes up when the caller's last words asked for more ("another", "make it three"), so "yeah, two pepperonis" never becomes four.
- Corrections change one line (`update_item`, `remove_item`), never rebuild the cart. "A dozen wings" is one order of twelve, not twelve orders.
- A sold-out item is refused with others from the same category to offer instead.
- Allergy questions go through `items_with_allergen`, one allergen per call. It names what the menu data lists, and never calls anything safe or allergen-free, since the kitchen is shared; a severe allergy gets an offer to transfer to staff. Every allergy mentioned stays in play, so a later item that contains it is flagged.
- `offer_breadsticks` is the one upsell, allowed once, and never when breadsticks are already in the order.
- Delivery goes anywhere with a street address, costs $4.99, and needs a $20 subtotal.
- The callback number has to be a real ten-digit number, however it's said ("four one five, five five five, oh one four two").
- An order is submitted only at a total that was priced and read back (`price_order`). Any change after the read-back needs a new one.

`end_call` refuses until the caller has heard the order number and when it'll be ready.

### The outbound delivery update

Order #214 (a large pepperoni, a dozen hot wings, and a 2-liter Sprite) is with Marco, about 5 minutes away, and the Sprite ran out after the order went in. Rosa opens with who she is and asks for the customer, saying nothing about the order until they confirm. A wrong number, a bad time, or a "no thanks" ends the call with `end_without_update`, sharing nothing.

With the customer on the line, she offers a same-price swap (a 2-liter Coke) or a refund of the item and its tax, and the customer can change their mind either way. She asks about drop-off details for the driver, can add breadsticks (the only thing the driver carries), refuses to cancel an order that's already out, and offers a note for the manager instead. `final_read_back` gives the changes, the new total, and the charge or refund; `end_call` refuses until the customer has heard it.

### Voice and turn-taking

The session runs on LiveKit Inference: Deepgram Nova-3, GPT-4.1 mini, and Cartesia Sonic 3.5, plus the turn detector, adaptive interruption (so "mm-hm" doesn't stop Rosa but "no, wait" does), and preemptive generation. A short break after each sentence keeps Cartesia from running them together.

Background noise is handled in two places. Before speech-to-text, enhanced noise cancellation removes background voices (a TV, people nearby) and noise: `BVC` for browser callers, and `BVCTelephony`, trained on narrowband call audio, for phone callers. Then adaptive interruption decides whether what's left is the caller cutting in or just a sound, so a cough or a passing voice doesn't stop Rosa mid-sentence. Leave Krisp off on the SIP trunk, since the agent already cleans the audio.

## Run it

```bash
python agent.py console                         # an order
CALL_DIRECTION=outbound python agent.py console # the delivery update
python agent.py dev
```

In `dev`, the job metadata picks the direction: `{"direction": "outbound"}` starts the delivery update with the person in the room as the customer. To ring a real phone, add the number and set an outbound SIP trunk:

```json
{"direction": "outbound", "dial": "+15105550100", "name": "Sam Carter"}
```

with `SIP_OUTBOUND_TRUNK_ID` set. `name` is who the order is under (Sam Carter if left out). Answering-machine detection decides who picked up: voicemail gets a short message with a callback number, and Rosa joins only for a person. Set `RESTAURANT_NUMBER` to the number customers should call back on.

A transfer to staff ends the call here, since there's no one to hand off to; see [warm-transfer](../warm-transfer/) for moving the caller to a real person.

Things to try: order two large pepperonis, then say "yeah, two pepperonis" again. Say "make that a medium, and drop the Coke" during the read-back. Ask for the whole menu. On the delivery update, take the refund, then change your mind.

## Demo page

`page.py` is the layer for the telephony demo on livekit.io, which dispatches this agent by name. The page joins the room as the browser caller (`caller-*`) or as a viewer (`viewer-*`) of a phone call, and draws what the agent sends it over the `demo.set_state` RPC: the call's stage, the POS's state (the cart, the receipt), a log of the tool calls, and a suggested next line for the caller. A browser caller's name is who the delivery update's order is under. From the job metadata the site sends, a call to a real phone takes `dial`, the watching `page`, and the line to call from (`trunk` or `from`, which win over `SIP_OUTBOUND_TRUNK_ID`).

Without a page watching (console, the playground, simulations, tests), it's inert: nothing is sent and no suggestions are written, so there are no extra LLM calls.

## Tests

```bash
cd examples/restaurant_ordering
python -m pytest                 # the POS's rules and grading (unit)
python -m pytest -m evals        # text-only conversations against LiveKit Inference
```

The evals need `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.

## Simulations

`simulations.yaml` holds 17 scenarios: the main orders, the policy twists (a sold-out side, delivery under the minimum, delivery out of town, allergies), callers who correct an earlier detail or talk in bursts, and the outbound paths. A run passes only if the conversation meets the scenario's expectations and `grade.py` finds the POS in the state its `expect` block names.

```bash
lk agent simulate examples/restaurant_ordering/agent.py \
  --scenarios examples/restaurant_ordering/simulations.yaml
```
