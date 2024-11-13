# Restaurant Reservation Agent Example

This example demonstrates an agent that handles the reservation for Sphinx restaurant.

Features:

- Customer queries regarding the restaurant are answered through RAG provided context
- Checks availability based off of the `availability` file when a customer requests to make a reservation
- Lists alternate options if requested timeslot is not available
- Creates a reservation by updating the CSV file

# Run

`cd rag`

`python build_data.py`

`cd ..`

`python assistant.py dev`

## Disclaimers

- The CSV file only lists dates between November to December 2024 between 11 AM - 8 PM
