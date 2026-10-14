# Vehicle Service Example

A car dealership's service desk on the phone. Leo answers for Bayside BMW: he talks a caller through what their car is doing, turns each answer into a note for the technician, catches the open recall, keeps wear items out of warranty, and books the drop-off with the shuttle or a loaner. On the follow-up call he rings the customer back with what the technician found, the price, what can wait, and when the car will be ready.

For setup instructions and more details, see the [main examples README](../README.md).

## Overview

The example is split so that the rules never depend on the model following its prompt:

- `backend.py` is the mock system of record: the customer's file (an in-warranty 2023 X3 with an open recall, and a 2019 330i out of warranty), five business days of advisor slots, the price menu, and, for the follow-up, the repair order for the X3 that's in the shop. Every rule lives here and refuses with a `ServiceError` whose message tells the agent what to do instead.
- `agent.py` defines `LeoAgent` (inbound) and `RepairCallAgent` (outbound), their tools, and the entrypoint. Each tool wraps one desk call; a refusal goes back to the LLM as a `ToolError` to relay.
- `call.py` holds what any phone agent here needs: the speaking rules, sentence pacing for Cartesia, background noise and voice cancellation (BVC for browser callers, BVCTelephony on phone calls), and hanging up without cutting the caller off.
- `grade.py` scores a simulation run from the desk's final state.

### The inbound consultation

Leo asks for a name and the phone number on the account, finds the file, and works out which car. For a problem, he asks one question at a time (the sound, where it comes from, when it happens, how long, any warning lights, anything recent) and logs each answer with `add_tech_note`. Once the sound and the moment point somewhere, the tool says what that often means and whether it's safe to drive, so Leo can say it as a possibility the technician will confirm, never a diagnosis.

The desk enforces the rest:

- Only menu items have prices (`quote_price`): an oil change, a tire rotation, or the $159 diagnostic fee. A brake job has no set price, so the agent can't invent one.
- A warranty repair needs a car under 3 years and 36,000 miles, and never covers wear items like brakes, tires, or wipers.
- Recalls must be checked before booking, and an open recall needs an answer.
- Loaners are only for warranty or recall visits over three hours; anyone can take the free shuttle or wait in the lounge.
- Every booking is read back first (`read_back`), and any change after the read-back needs another one.

A correction changes one field and re-checks what depends on it: a mileage that ends the warranty drops the warranty repair, a shorter visit drops the loaner, and a longer one drops a slot that no longer fits before closing. The tool result tells Leo what moved so he can say it.

Advisors are introduced by role ("Dana, one of our service advisors, will meet you at the service drive"), since the caller doesn't know who they are.

### The outbound follow-up

The X3 came in this morning for the grinding noise; the technician found the front pads worn to the metal and the rotors scored ($489, needs doing now) and the rear pads at 4 mm ($214, can wait). Leo opens with who he is and asks for the customer, and says nothing about the car until `confirm_customer` succeeds. Someone else answering, or a bad time, ends the call with `end_without_update` and a callback number, sharing nothing.

With the customer on the line, he goes over the cause, the price, and when the car would be ready, then records `approve_repair` or `decline_repair` (including "let me think about it"). Questions go through `repair_details`, so every price, part, and time comes from the findings. Once the urgent repair has an answer, `arrange_pickup` sets the shuttle or a drive-in, and `pickup_summary` gives the ready time and total. `end_call` refuses until the customer has heard both.

### Voice and turn-taking

The session runs on LiveKit Inference: Deepgram Nova-3, GPT-4.1 mini, and Cartesia Sonic 3.5 with an enthusiastic voice, plus the turn detector, adaptive interruption (so "mm-hm" doesn't stop Leo but "no, wait" does), and preemptive generation.

Background noise is handled in two places. Before speech-to-text, enhanced noise cancellation removes background voices (a TV, people nearby) and noise: `BVC` for browser callers, and `BVCTelephony`, trained on narrowband call audio, for phone callers. Then adaptive interruption decides whether what's left is the caller cutting in or just a sound, so a cough or a passing voice doesn't stop Leo mid-sentence. Leave Krisp off on the SIP trunk, since the agent already cleans the audio.

## Run it

```bash
python agent.py console                         # the inbound call
CALL_DIRECTION=outbound python agent.py console # the follow-up
python agent.py dev
```

In `dev`, the job metadata picks the direction: `{"direction": "outbound"}` starts the follow-up with the person in the room as the customer. To ring a real phone, add the number and set an outbound SIP trunk:

```json
{"direction": "outbound", "dial": "+15105550100"}
```

with `SIP_OUTBOUND_TRUNK_ID` set. Answering-machine detection decides who picked up: voicemail gets a short message with a callback number, and Leo joins only for a person. Set `SERVICE_DESK_NUMBER` to the number customers should call back on.

Things to try: describe a grinding when you brake, then correct where it comes from. Ask for a loaner on an oil change. On the follow-up, ask for a cheaper option, or say you'll think about it.

## Demo page

`page.py` is the layer for the telephony demo on livekit.io, which dispatches this agent by name. The page joins the room as the browser caller (`caller-*`) or as a viewer (`viewer-*`) of a phone call, and draws what the agent sends it over the `demo.set_state` RPC: the call's stage, the service desk's state (the customer's file, the booking, the findings), a log of the tool calls, and a suggested next line for the caller. A browser caller's name is who the file is under, and the visitor's `timezone` in the job metadata sets what "today" is. A call to a real phone takes `dial`, the watching `page`, and the line to call from (`trunk` or `from`, which win over `SIP_OUTBOUND_TRUNK_ID`).

Without a page watching (console, the playground, simulations, tests), it's inert: nothing is sent and no suggestions are written, so there are no extra LLM calls.

## Tests

```bash
cd examples/vehicle_service
python -m pytest                 # the desk's rules and grading (unit)
python -m pytest -m evals        # text-only conversations against LiveKit Inference
```

The evals need `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.

## Simulations

`simulations.yaml` holds 19 scenarios: the main bookings, the policy twists (a loaner refused, brakes under warranty, a lowered mileage), callers who correct an earlier detail, and the follow-up paths. A run passes only if the conversation meets the scenario's expectations and `grade.py` finds the desk in the state its `expect` block names.

```bash
lk agent simulate examples/vehicle_service/agent.py \
  --scenarios examples/vehicle_service/simulations.yaml
```
