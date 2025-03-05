import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

HEADERS = {
    "cal-api-version": "2024-06-14",
    "Authorization": "Bearer " + os.getenv("CAL_API_KEY"),
}


async def get_event_id(slug: str) -> str | None:
    """Searches for an event type. Returns the event ID if found, None if not

    Args:
        slug (str): The unique identifier of the event type
    """
    payload = {"username": os.getenv("CAL_API_USERNAME"), "eventSlug": slug}
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.cal.com/v2/event-types", params=payload, headers=HEADERS
        ) as response:
            response = await response.json()
            if response["status"] == "success" and response["data"]:
                return response["data"][0]["id"]
            if response["status"] == "error":
                raise Exception("Error retrieving event type")
            else:
                return None


async def create_schedule() -> str:
    """Sets schedule for example, returns schedule ID"""
    payload = {
        "name": "LiveKit Dental Office Hours",
        "timeZone": "America/Los_Angeles",
        "availability": [
            {
                "days": [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                ],
                "startTime": "10:00",
                "endTime": "12:00",
            },
            {
                "days": [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                ],
                "startTime": "13:00",
                "endTime": "16:00",
            },
        ],
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.cal.com/v2/event-types", json=payload, headers=HEADERS
        ) as response:
            response = await response.json()
            if response["status"] == "success" and response["data"]:
                return response["data"][0]["id"]
            if response["status"] == "error":
                raise Exception("Error creating schedule")


async def create_event_type(*, title: str, slug: str, schedule_id: str) -> str:
    """Creates specified event type and returns the event ID

    Args:
        title (str): The title of the event type
        slug (str): The unique identifier of the event type, typically with dashes in place of spaces
    """
    payload = {
        "lengthInMinutes": 60,
        "title": title,
        "slug": slug,
        "scheduleId": schedule_id,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.cal.com/v2/event-types", json=payload, headers=HEADERS
        ) as response:
            response = await response.json()
            if response["status"] == "success":
                return response["data"]["id"]
            else:
                raise Exception(f"{response["error"]["code"]}")


async def setup_event_types() -> dict:
    """Ensures that the schedule and event types are set up correctly in Cal.com for this example. Returns a dictionary with event slugs and their respective IDs"""
    schedule_id = create_schedule()

    event_ids = {}

    checkup_event_id = await get_event_id("routine-checkup")
    if not checkup_event_id:
        checkup_event_id = await create_event_type(
            title="Routine Checkup", slug="routine-checkup", schedule_id=schedule_id
        )

    event_ids["routine-checkup"] = checkup_event_id

    extraction_event_id = await get_event_id("tooth-extraction")
    if not extraction_event_id:
        extraction_event_id = await create_event_type(
            title="Tooth Extraction", slug="tooth-extraction", schedule_id=schedule_id
        )

    event_ids["tooth-extraction"] = extraction_event_id

    return event_ids
