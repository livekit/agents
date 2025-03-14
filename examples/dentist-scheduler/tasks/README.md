# `AgentTask`
[AgentTask](https://github.com/yepher/livekit_info/blob/main/API_GUIDE.md#agenttask-class) allows you to establish a coherent workflow and customize agents to a greater level. This file breaks down each section of the agents.

Breaking down the Agents:
- [Receptionist](#building-the-receptionist)
- [Scheduler](#building-the-scheduler)
- [Messenger](#building-the-messenger)

## Building the Receptionist 

The Receptionist handles simple inquiries about the office and transfers you to either the Scheduler or Messenger upon request.

### Initialization

First, initialize `AgentTask` with instructons and a new text-to-speech setting:
```
def __init__(self) -> None:
        super().__init__(
            instructions="""You are Alloy, a receptionist at the LiveKit Dental Office who answers inquiries and manages appointments for users. 
            If there is an inquiry that can't be answered, suggest to leave a message. Be brief and efficient.""",
            tts=cartesia.TTS(emotion=["positivity:high"]),
        )
```


Using `on_enter()`, we define the agent's actions upon entering the session. This is also the best place to access the `userdata` field, as seen with the user's name:
```
async def on_enter(self) -> None:
        await self.agent.generate_reply(
            instructions=f"""Welcome the user to the LiveKit Dental Office and ask how you can assist.
                            The user's name is {self.agent.userdata["userinfo"].name}."""
        )
```

### General Inquiries

This recipe uses functions to answer questions, specifically for hours and location inquiries. An alternative to this approach is appending the information to the instructions in `__init__()`, but it is better practice to create functions so the instructions aren't bulky.

The docstrings include a function description for `ai_function()` and helps determine when to call it. Inquiry answers are included in the instructions of `generate_reply()`:

```
@ai_function()
    async def hours_inquiry(self) -> None:
        """Answers user inquiries about the LiveKit dental office's hours of operation."""
        await self.agent.current_speech.wait_for_playout()
        await self.agent.generate_reply(
            instructions="Inform the user that the LiveKit dental office is closed on Sundays but open 10 AM to 12 PM, and 1 PM to 4 PM otherwise."
        )
```

### Transferring and Function Arguments

This example allows for full mobility between agents. Before transferring to `Scheduler()` and `Messenger()`, we want to collect the user's name before proceeding. For `Scheduler()`, we also want to collect which service (schedule/reschedule/cancel) is requested.

The docstrings contain function parameter descriptions as well:
```
    @ai_function()
    async def manage_appointment(self, name: str, service: str):
        """
        This function allows for users to schedule, reschedule, or cancel an appointment by transferring to the scheduler.
        The user's name will be confirmed with the user by spelling it out.
        Args:
            name: The user's name
            service: Either "schedule", "reschedule", or "cancel"
        """
        self.agent.userdata["userinfo"].name = name
        return self.agent.userdata["tasks"].scheduler(
            service=service
        ), "I'll be transferring you to our scheduler, Echo!"
```

Once the function is called with the required arguments, `userdata` is updated with the user's name for the other agents to access. `service` is passed as an argument to `Scheduler()`, which is returned along with a string to ease the transition. The string is only processed when `TTS()` is defined.

## Building the Scheduler

The Scheduler is able to schedule, reschedule, or cancel appointments via Cal.com API calls.

### Initialization

### API Calls


### Scheduling Functionalities

## Building the Messenger

The Messenger records user messages and sends it to Supabase. 

### Integrating Supabase
