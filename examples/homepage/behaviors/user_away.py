"""Check in when the user goes quiet.

The session marks the user "away" after ``user_away_timeout`` seconds of
silence (an ``AgentSession`` option, 15s by default); the state only returns
to "listening" once the user speaks again, so the check-in fires once per
quiet spell rather than repeating.
"""

from prompts import prompt

from livekit.agents import AgentSession, UserStateChangedEvent

CHECK_IN_INSTRUCTIONS = prompt("user_away")


def check_in_when_user_away(session: AgentSession) -> None:
    @session.on("user_state_changed")
    def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
        if ev.new_state == "away":
            session.generate_reply(
                instructions=CHECK_IN_INSTRUCTIONS,
                allow_interruptions=True,
            )
