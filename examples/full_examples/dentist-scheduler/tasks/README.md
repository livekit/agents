# `AgentTask`
[AgentTask](https://github.com/yepher/livekit_info/blob/main/API_GUIDE.md#agenttask-class) allows you to establish a coherent workflow and customize agents to a greater level. This file breaks down each section of the agents.

Some of the agents share functions such as updating user information, and these functions are stored in `global_functions.py` to reduce redundancy. 

Breaking down the Agents:
- [Receptionist](#building-the-receptionist)
- [Scheduler](#building-the-scheduler)
- [Messenger](#building-the-messenger)

[Shared Functions](#shared-functions)

## Building the Receptionist 

The Receptionist handles simple inquiries about the office and transfers you to either the Scheduler or Messenger upon request.

### Initialization

First, initialize `AgentTask` with instructons, a new text-to-speech setting, and preset functions:
```
def __init__(self) -> None:
        super().__init__(
            instructions="""You are Alloy, a receptionist at the LiveKit Dental Office who answers inquiries and manages appointments for users. 
            If there is an inquiry that can't be answered, suggest to leave a message. Be brief and efficient, do not ask for unnecessary details. 
            When handling appointments or taking a message, you will transfer the user to another agent.""",
            tts=cartesia.TTS(emotion=["positivity:high"]),
            tools=[update_information, get_user_info],
        )
```


Using `on_enter()`, we define the agent's actions upon entering the session. This is also the best place to access the `userdata` field, as seen with the user's name:
```
async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=f"""Welcome the user to the LiveKit Dental Office and ask how you can assist."""
        )
```

### General Inquiries

This recipe uses functions to answer questions, specifically for hours and location inquiries. An alternative to this approach is appending the information to the instructions in `__init__()`, but it is better practice to create functions so the instructions aren't bulky.

The docstrings include a function description for `function_tool()` and helps determine when to call it. Inquiry answers are returned:

```
    @function_tool()
    async def hours_inquiry(self) -> str:
        """Answers user inquiries about the LiveKit dental office's hours of operation."""
        return "The LiveKit dental office is closed on Sundays but open 10 AM to 12 PM, and 1 PM to 4 PM otherwise."
       
```

### Transferring and Function Arguments

This example allows for full mobility between agents. Before transferring to `Scheduler()` and `Messenger()`, we want to collect the user's name before proceeding. For `Scheduler()`, we also want to collect which service (schedule/reschedule/cancel) is requested.

If the user's name was not previously defined, it will be updated. 

```
    @function_tool()
    async def manage_appointment(
        self, 
        name: Annotated[list[str], Field(description="The user's name")], 
        action: Annotated[list[str], Field(description="The appointment action requested, either 'schedule', 'reschedule', or 'cancel'")],
    ) -> tuple[Agent, str]:
        """
        This function allows for users to schedule, reschedule, or cancel an appointment by transferring to the scheduler. No specified date or time is required. 
        """
        if not self.session.userdata["userinfo"].name:
            self.session.userdata["userinfo"].name = name
        return self.session.userdata["agents"].scheduler(
            service=action
        ), "I'll be transferring you to our scheduler, Echo!"
```

 `service` is passed as an argument to `Scheduler()`, which is returned along with a string to ease the transition.

## Building the Scheduler

The Scheduler is able to schedule, reschedule, or cancel appointments via Cal.com API calls. Please note that when using an older model, faulty dates may be returned. If this is the case, modify the instructions to include the current year.  

### Initialization

The Cal.com API is case sensitive and dates must be correctly formatted. Add detailed instructions to avoid faulty API requests: 
```
def __init__(self, *, service: str) -> None:
        super().__init__(
            instructions="""You are Echo, a scheduler managing appointments for the LiveKit dental office. If the user's email is not given, ask for it before 
                            scheduling/rescheduling/canceling. Assume the letters are lowercase unless specified otherwise. When calling functions, return the user's email if already given.
                            Always confirm details with the user. Convert all times given by the user to ISO 8601 format in UTC timezone,
                            assuming the user is in America/Los Angeles, and do not mention the conversion or the UTC timezone to the user. Avoiding repeating words.""",
            tts=cartesia.TTS(voice="729651dc-c6c3-4ee5-97fa-350da1f88600"),
            tools=[update_information, get_user_info, transfer_to_receptionist, transfer_to_messenger],
        )
        self._service_requested = service
```
### API Calls

In this example, we only need to access upcoming appointments or schedule/reschedule/cancel them. Create an `Enum` object:
```
class APIRequests(Enum):
    GET_APPTS = "get_appts"
    CANCEL = "cancel"
    RESCHEDULE = "reschedule"
    SCHEDULE = "schedule"
```
Define a function to facilitate API calls. The request body is structured accordingly and filtered by GET/POST calls:

```
  async def send_request(
        self, *, request: APIRequests, uid: str = "", time: str = "", slug: str = ""
    ) -> dict:
        headers = {
            "cal-api-version": "2024-08-13",
            "Authorization": "Bearer " + os.getenv("CAL_API_KEY"),
        }
        async with aiohttp.ClientSession() as session:
            params = {}
            if request.value == "get_appts":
                payload = {
                    "attendeeEmail": self.session.userdata["userinfo"].email,
                    "attendeeName": self.session.userdata["userinfo"].name,
                    "status": "upcoming",
                }
                params = {
                    "url": "https://api.cal.com/v2/bookings",
                    "params": payload,
                    "headers": headers,
                }
            ...
            if request.value in ["schedule", "reschedule", "cancel"]:
                async with session.post(**params) as response:
                    data = await response.json()
            elif request.value == "get_appts":
                async with session.get(**params) as response:
                    data = await response.json()
            else:
                raise Exception("Cal.com API Communication Error")
            return data
```

### Scheduling Functionalities

If the time selected is not available, the Scheduler will suggest to pick another time. If no appointments are found, then the Scheduler will suggest to create one. If multiple appointments were made, the next upcoming one will be selected. 

    | Step | schedule(email, description, date) | reschedule(email, new_time) | cancel(email) |
    | ---- | ---------------------------------- | --------------------------- | ------------- |
    |   1  |    Send a SCHEDULE request        | Check for existing appointment | Check for existing appointment |
    |   2  |                                   | Send a RESCHEDULE request    |  Send a CANCEL request |

## Building the Messenger

The Messenger records user messages and sends it to Supabase. 

### Integrating Supabase

Create a class `SupabaseClient` to simplify Supabase operations:

```
class SupabaseClient:
    def __init__(self, supabase: AsyncClient) -> None:
        self._supabase = supabase

    @classmethod
    async def initiate_supabase(supabase):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        supabase_client: AsyncClient = await create_async_client(url, key)
        return supabase(supabase_client)

    async def insert_msg(self, name: str, message: str, phone: str) -> list:
        data = await (
            self._supabase.table("messages")
            .insert({"name": name, "message": message, "phone_number": phone})
            .execute()
        )
```

This recipe uses Supabase's `AsyncClient`. 

### Initialization

`SupabaseClient` will be initiated once the user meets the messenger. This happens in `on_enter()`:

```
async def on_enter(self) -> None:
        self._supabase = await SupabaseClient.initiate_supabase()

        await self.session.generate_reply(
            instructions=f"""Introduce yourself and ask {self.session.userdata["userinfo"].name} for their phone number if not given. Then, ask for the message they want to leave for the office."""
        )
```

We ensure that the phone number is given before sending the message. If given previously, `Messenger()` will access `get_user_info` to retrieve it. The agent will return the phone number and message, submit to Supabase, and return a confirmation message if successful.

```
@function_tool()
    async def record_message(
        self,
        phone_number: Annotated[str, Field(description="The user's phone number")],
        message: Annotated[
            str, Field(description="The user's message to be left for the office")
        ],
    ) -> str:
        """Records the user's message to be left for the office and the user's phone number."""
        self.session.userdata["userinfo"].phone = phone_number
        self.session.userdata["userinfo"].message = message
        try:
            data = await self._supabase.insert_msg(
                name=self.session.userdata["userinfo"].name,
                message=message,
                phone=phone_number,
            )
            if data:
                return "Your message has been submitted."
        except Exception as e:
            raise Exception(f"Error sending data to Supabase: {e}")
```

## Shared Functions

To have shared functionalities across agents, create `global_functions.py` to define them.

For example, each agent can update information on record:
```
@function_tool()
async def update_information(
    field: Annotated[str, Field(description="The type of information to be updated, either 'phone_number', 'email', or 'name'")], 
    info: Annotated[str, Field(description="The new user provided information")], 
    context: RunContext,
    ) -> str:
    """ 
    Updates information on record about the user. The only fields to update are names, phone numbers, and emails.
    """
    userinfo = context.userdata["userinfo"]
    if field == "name":
        userinfo.name = info
    elif field == "phone_number":
        userinfo.phone = info
    elif field == "email":
        userinfo.email = info
    
    return "Got it, thank you!"
```

Since these functions are disconnected from an `Agent` class, add `context: RunContext` as a parameter to access the session's `userdata`.


