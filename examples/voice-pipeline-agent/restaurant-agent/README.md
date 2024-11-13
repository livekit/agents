# Restaurant Reservation Agent Example

This example demonstrates an agent that handles the reservation for Sphinx restaurant.

Features:

- Customer queries regarding the restaurant are answered through RAG provided context
- Checks availability based off of the `availability` file when a customer requests to make a reservation
- Creates a reservation

# Run

`cd rag`

`python build_data.py`

`cd ..`

`python assistant.py dev`
