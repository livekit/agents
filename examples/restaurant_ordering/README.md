# Restaurant Ordering Example

A pizzeria's phone line. Gabby answers for Zappi's Pizzeria: she takes pickup and delivery orders with sizes, sauces, toppings, and things left off, answers allergy questions from the menu data, steers around sold-out items, reads the whole menu when asked, offers breadsticks once, takes a tip or a note for the driver when asked, and reads back the total before the order goes to the kitchen; she can cancel it, too. On the outbound call she rings a customer whose delivery is already on its way to say an item ran out, and sorts out the replacement or takes it off what's due at the door.

For setup instructions and more details, see the [main examples README](../README.md).

## Overview

The example is split so that the rules never depend on the model following its prompt:

- `backend.py` is the mock point of sale: the menu (with today's sold-out Buffalo cauliflower), the cart, pricing with tax and the delivery fee, the order it submits, and, for the outbound call, order #214, already out with the driver. Every rule lives here and refuses with a `PosError` whose message tells the agent what to do instead.
- `agent.py` defines `GabbyAgent` (inbound) and `DeliveryUpdateAgent` (outbound), their tools, and the entrypoint. Each tool wraps one POS call; a refusal goes back to the LLM as a `ToolError` to relay.
- `call.py` holds what any phone agent here needs: the speaking rules, background noise and voice cancellation (BVC for browser callers, BVCTelephony on phone calls), and hanging up without cutting the caller off.
- `grade.py` scores a simulation run from the POS's final state.

### Taking an order

Gabby adds each item with `add_item` as soon as it's named and asks only for what's missing: a pizza's size, a wing size and sauce, a drink size. Names are matched loosely ("a large", "hot sauce", "meat lover's pizza", "cokes"), and "no olives" is taken off the pizza at the same price.

The POS enforces the rest:

- People repeat themselves. An item already in the cart can't be added again, and a count only goes up when the caller's last words asked for more ("another", "make it three"), so "yeah, two pepperonis" never becomes four.
- Corrections change one line (`update_item`, `remove_item`), never rebuild the cart. `update_item` can also change what the line is ("make the Coke a Sprite"), keeping its count and size. "A dozen wings" is one order of twelve, not twelve orders.
- A sold-out item is refused with others from the same category to offer instead.
- Allergy questions go through `items_with_allergen`, one allergen per call. It names what the menu data lists, and never calls anything safe (or safer, or safest) or allergen-free, since the kitchen is shared; a severe allergy gets the same answer, said seriously. Every allergy mentioned stays in play, so a later item that contains it is flagged.
- `offer_breadsticks` is the one upsell, allowed once, and never when breadsticks are already in the order.
- Delivery goes anywhere with a street address and costs $4.99. An order under $15, pickup or delivery, has a $2.99 small-order fee (untaxed, like the delivery fee), which Gabby mentions once at the read-back without pushing the caller to add anything.
- Drop-off details for a delivery ("leave it at the door, don't ring the bell") are saved with `set_delivery_note`, read back with the order, and sent with it. A pickup order has no driver to take a note, and a note that's really an address goes through `set_delivery` instead.
- Zappi's is open every day, 11 a.m. to 10 p.m.; that's in the prompt, and anything else about the restaurant that isn't, Gabby says she doesn't know.
- Customers pay in person, cash or card: at pickup, or to the driver at the door. Gabby never takes card details over the phone, and stops a caller who starts reading a card number.
- A tip, only when the caller asks for one (`set_tip`, in dollars or a percent of the subtotal), goes on the total and is read back; changing it after the read-back needs a new one, like any change.
- The callback number has to be a real ten-digit number, however it's said ("four one five, five five five, oh one four two").
- An order is submitted only at a total that was priced and read back (`price_order`), and only on the caller's clear yes to that read-back: "whatever" or "sure, I guess" gets "Just to be sure, should I place it?" instead. Any change after the read-back needs a new one.
- `cancel_order` cancels the order placed on this call, after reading it back ("Cancel the whole order? Nothing will be due.") and only once the caller has answered yes; nothing is due after that. Before an order is placed, "never mind" just means nothing goes in.

`end_call` refuses until the caller has heard the order number and when it'll be ready (or, for a cancelled order, that it's cancelled).

### The outbound delivery update

Order #214 (a large pepperoni, a dozen hot wings, and a large Sprite) is with Marco, about 5 minutes away, and the Sprite ran out after the order went in. Gabby opens with who she is and asks for the customer by name, or, with no name to go on, whether they placed an order tonight (never a made-up name), saying nothing about the order until they confirm. A wrong number, a bad time, or a "no thanks" ends the call with `end_without_update`, sharing nothing.

The order is paid at the door, cash or card, so nothing is charged or refunded on the call: each change makes what's due at the door more or less. With the customer on the line, she offers a same-price swap (a large Coke) or taking the item off ($4.35 less at the door, the item and its tax), and the customer can change their mind either way. Once they've confirmed it's them, she reads them the order when asked. She asks about drop-off details for the driver (a note can't send the order to another address: the driver takes it to the one on it, and cancelling is the way out), can add breadsticks (the only thing the driver carries), and adds a tip for the driver if the customer asks. The customer can cancel the whole order even though it's on its way: once they confirm, Marco turns back and nothing is due. A complaint gets a sincere apology and only what she can actually do (the swap, taking the item off, or cancelling), with no other promises. `final_read_back` gives the changes and what's due at the door; `end_call` refuses until the customer has heard it.

### Voice and turn-taking

The session runs on LiveKit Inference: Deepgram Nova-3, GPT-4.1 mini, and Rime Coda with the voice Luna, plus the turn detector, adaptive interruption (so "mm-hm" doesn't stop Gabby but "no, wait" does), and preemptive generation. A small text transform has the voice say "pepperoni pizzas" rather than "pepperonis".

Background noise is handled in two places. Before speech-to-text, enhanced noise cancellation removes background voices (a TV, people nearby) and noise: `BVC` for browser callers, and `BVCTelephony`, trained on narrowband call audio, for phone callers. Then adaptive interruption decides whether what's left is the caller cutting in or just a sound, so a cough or a passing voice doesn't stop Gabby mid-sentence. Leave Krisp off on the SIP trunk, since the agent already cleans the audio.

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

with `SIP_OUTBOUND_TRUNK_ID` set. `name` is who the order is under; without it (or a browser caller's name), the order is under no name, and Gabby never says one. Answering-machine detection decides who picked up: voicemail gets a short message with a callback number, and Gabby joins only for a person. Set `RESTAURANT_NUMBER` to the number customers should call back on.

Gabby handles every call herself, with no handoff to staff, no notes for a manager, and no promised call backs: something the restaurant doesn't do, a discount code, a complaint, or an allergy the menu data doesn't track gets a plain answer from her. She never says she's noted or done something unless a tool did it. To add a handoff, see [warm-transfer](../warm-transfer/).

If the line goes quiet (neither side speaking for 20 seconds), Gabby asks once, "Are you still there?"; if it stays quiet, she says goodbye and hangs up (`hang_up_when_silent` in `call.py`). Silence never agrees to anything. It's off in simulations and text-only tests.

Things to try: order two large pepperonis, then say "yeah, two pepperonis" again. Say "make that a medium, and drop the Coke" during the read-back. Ask for the whole menu. Add a tip, then cancel the order once it's placed. On the delivery update, have the Sprite taken off, then change your mind, or cancel the whole order.

## Demo page

`page.py` is the layer for the telephony demo on livekit.io, which dispatches this agent by name. The page joins the room as the browser caller (`caller-*`) or as a viewer (`viewer-*`) of a phone call, and draws what the agent sends it over the `demo.set_state` RPC: the call's stage, the POS's state (the cart, the receipt with `quote.smallOrderFee` and `quote.tip`, `fulfillment.note` for the driver, `payment`, and `order.cancelled`; on the delivery update, `delivery.due`, `dueNow`, `balance`, `note`, and `cancelled`), a log of the tool calls, and a suggested next line for the caller. A browser caller's name is who the delivery update's order is under. From the job metadata the site sends, a call to a real phone takes `dial`, the watching `page`, and the line to call from (`trunk` or `from`, which win over `SIP_OUTBOUND_TRUNK_ID`).

Without a page watching (console, the playground, simulations, tests), it's inert: nothing is sent and no suggestions are written, so there are no extra LLM calls.

## Tests

```bash
cd examples/restaurant_ordering
python -m pytest                 # the POS's rules and grading (unit)
python -m pytest -m evals        # text-only conversations against LiveKit Inference
```

The evals need `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.

## Simulations

`simulations.yaml` holds 21 scenarios: the main orders, the policy twists (a sold-out side, a small order's fee, delivery out of town, allergies, a tip, a note for the driver, a cancellation), callers who correct an earlier detail or talk in bursts, and the outbound paths (including a cancellation and a complaint). A run passes only if the conversation meets the scenario's expectations and `grade.py` finds the POS in the state its `expect` block names.

```bash
lk agent simulate examples/restaurant_ordering/agent.py \
  --scenarios examples/restaurant_ordering/simulations.yaml
```
