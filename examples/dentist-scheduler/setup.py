import aiohttp
import os

HEADERS = {
            "cal-api-version": "2024-06-14",
            "Authorization": "Bearer " + os.getenv("CAL_API_KEY")
        } 

async def get_event_id(slug: str) -> str | None:
    """ Searches for an event type. Returns the event ID if found, None if not 

        Args:
            slug (str): The unique identifier of the event type
    """
    payload = {"username": os.getenv("CAL_API_USERNAME"), "eventSlug": event}
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.cal.com/v2/event-types", params=payload, headers=HEADERS) as response:
            response = response.json()
            if response.status == "success" and response.data:
                return response.data[0].id 
            if response.status == "error":
                raise Exception(f"{response.error.details.statusCode} {response.error.code}: {response.error.message}")
            else:
                return None

async def create_event_type(*, title: str, slug: str) -> str:
    """ Creates specified event type and returns the event ID

        Args:
            title (str): The title of the event type
            slug (str): The unique identifier of the event type, typically with dashes in place of spaces
    """
    payload = {"lengthInMinutes": 60, "title": title, "slug": slug}
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.cal.com/v2/event-types", params=payload, headers=HEADERS) as response:
            response = response.json()
            if response.status == "success":
                return response.data.id
            else:
                raise Exception(f"Error: {response.error.code}")
            
async def setup_api() -> dict:
    """ Ensures that event types are set up correctly in Cal.com for this example. Returns a dictionary with event slugs and their respective IDs """
    event_ids = {}

    checkup_event_id = await get_event_id("routine-checkup")
    if not checkup_event_id:
        checkup_event_id = create_event_type(title="Routine Checkup", slug="routine-checkup")

    event_ids["routine-checkup"] = checkup_event_id
 
    extraction_event_id = await get_event_id("tooth-extraction")
    if not extraction_event_id:
        extraction_event_id = create_event_type(title="Tooth Extraction", slug="tooth-extraction")

    event_ids["tooth-extraction"] = extraction_event_id

    return event_ids