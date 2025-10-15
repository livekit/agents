from ... import function_tool
from ...job import get_job_context
from ..workflows.utils import DtmfEvent, dtmf_event_to_code


@function_tool
async def send_dtmf_events(
    events: list[DtmfEvent],
) -> None:
    """
    Send a list of DTMF events to the telephony provider.

    Call when:
    - User wants to send DTMF events
    """
    job_ctx = get_job_context()

    for event in events:
        try:
            code = dtmf_event_to_code(event)
            await job_ctx.room.local_participant.publish_dtmf(code=code, digit=event.value)
        except Exception as e:
            return f"Failed to send DTMF event: {event.value}. Error: {str(e)}"

    return f"Successfully sent DTMF events: {', '.join(events)}"
