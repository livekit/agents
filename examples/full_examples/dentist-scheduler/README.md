# :tooth: Building a Dentist Scheduler Assistant
Build an AI-powered dentist assistant that manages appointments, takes messages, and answers inquiries

This example integrates the Cal.com and Supabase APIs with LiveKit and exhibits the functionality of a multi-agent framework. To curate a complete assistant, this recipe includes a Recepionist, Scheduler, and Messenger.

# Prerequisites
For this example, you need to create an VoiceAgent using the [Voice Agent quickstart](https://docs.livekit.io/agents/quickstarts/voice-agent/).

### Setting up the Cal API
You need to [create an account](https://app.cal.com/signup) and enroll in a plan. The `.env.local` file should have your API key and Cal.com username:
```
CAL_API_KEY="<cal_api_key>"
CAL_API_USERNAME="<cal_api_username>"
```

We'll set up the details for this example for you in-house upon running the agent, this includes the schedule and necessary event types.

### Setting up Supabase
1. [Create an account](https://supabase.com/dashboard/sign-up) and a new project for this example. 
2. After configuring your RLS settings (SELECT and INSERT policies for this example), create a table in the public schema. 

    Your table should have these text columns: `name`, `message`, and `phone_number`. You can optionally add a datetime column that     automatically records the time the message was received.
  
    Here is what your table should look like:
    | date | name | message | phone_number |
    | ---- | ---- | ------- | ------------ |
  

3. Add your project API key and URL into the `.env.local` file:
```
SUPABASE_API_KEY="<supabase_api_key>"
SUPABASE_URL="<supabase_project_url>"
```

# Storing data across agents
Agents can access shared data through `VoiceAgent.userdata`. You can store information about the user, tasks, and API information. 

In this example, we create `UserInfo` to store user information:
```
@dataclass
class UserInfo:
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    message: str | None = None
```

We also use `Agents` as an agents bank for smooth transfers:
```
@dataclass
class Agents:
    @property
    def receptionist(self) -> Agent:
        return Receptionist()

    @property
    def messenger(self) -> Agent:
        return Messenger()

    def scheduler(self, service: str) -> Agent:
        return Scheduler(service=service)

```


This example stores event IDs from setting up the Cal API, an instance of `UserInfo`, and an instance of `Agents()`. Modify `VoiceAgent`'s arguments to include `userdata`:
```
userdata = {"event_ids": event_ids, "userinfo": UserInfo(), "agents": Agents()}
    agent = VoiceAgent(
        task=Receptionist(),
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )
```

# Running the example
After setting up the environment, run `python dental_agent.py dev` to meet the Receptionist.

Try out:
- Scheduling a new appointment 
- Asking about the office hours and location
- Leaving a message for the office
- Transferring between agents




