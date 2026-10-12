# Fast brain, slow brain

A front desk talks to the person, by text and on the phone; its one tool delegates over A2A to a
back office that holds the system and does the work.

| Example | What |
| --- | --- |
| [`frontdesk/`](frontdesk/README.md) | A dental practice's scheduling: two tools and a booking that asks for what it is missing. |
| [`hotel-receptionist/`](hotel-receptionist/README.md) | A hotel's bookings, lookups and cancellations: guided flows, verification, approval on writes. |
| [`airline-support/`](airline-support/README.md) | An airline's customer care: a graph that asks for what is missing, EU261 in code, payment after approval. |

## Running them

Follow the [Getting started](../README.md#getting-started): install from PyPI, set the staging
project's keys, then `python main.py dev` in an example's directory (`python support_agent.py dev`
for the airline) and talk to it in the console.
