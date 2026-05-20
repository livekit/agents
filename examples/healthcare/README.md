# Healthcare Example

A full healhcare assistant providing secure appointment management, billing handling, and lab result retrieval.

For setup instructions and more details, see the [main examples README](https://github.com/livekit/agents/blob/main/examples/README.md).

## Overview

The healthcare agent utilizes a variety of `AgentTasks` to achieve structured workflows to collect information. This example is modality-agnostic, where users can interact via text or voice and switch seamlessly. If the conversation heads out of the scope of the agent, the user will be transfered to a human.

### Profile Authentication

Before any sensitive information is queried, the user must go through an authentication process. This process will only occur once per call. If the user provides a name and birthday existing in the database, the process is fast-forwarded. This is possible via task completion callbacks:
https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/healthcare/healthcare_agent.py#L574-L588
Otherwise, the user will also be asked for their phone number and insurance provider
to create a profile.

https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/healthcare/healthcare_agent.py#L590-L641

After the profile is created, the agent will be given a tool to update the patient's record as needed.

### Appointment Management

Function tools are added dynamically, so the LLM cannot hallucinate parameters or call tools prematurely. The user is given options for compatible doctors based on their insurance:

https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/healthcare/healthcare_agent.py#L306-L333

After the doctor is chosen, the appointment scheduling tool is built dynamically with the availabilities. 
https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/healthcare/healthcare_agent.py#L734-L780

Finally, the visit reason is collected, and once confirmed the database will be updated accordingly (the doctor's availability will be removed).

Users are also able to modify existing appointments. If the user wishes to reschedule an appointment, the appointment is canceled and `ScheduleAppointmentTask` is reused. 

https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/healthcare/healthcare_agent.py#L506-L543

### Billing Handling

`GetCreditCardTask()` is showcased here. The user's details are verified, and a balance is generated and connected to their profile.
https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/healthcare/healthcare_agent.py#L734-L748

### Lab Result Retrieval

We use OpenAI's provider tool, `FileSearch`, to read through an uploaded lab report (mock_checkup_report.pdf). 
https://github.com/livekit/agents/blob/8283a5a5c9863a07bcf030ee90e8ab780e1e569b/examples/healthcare/healthcare_agent.py#L734-L780

