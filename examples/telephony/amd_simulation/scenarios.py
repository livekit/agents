from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Voice = Literal["machine", "human"]
Category = Literal[
    "uncertain", "machine-screening", "machine-vm", "machine-ivr", "human", "machine-unavailable"
]

AGENT_INSTRUCTIONS = (
    "You are Alex from Acme Dental. You are calling Sam to confirm "
    "a dental appointment tomorrow at 10 AM. Keep replies brief. "
    "If asked to leave a message, give the appointment details and "
    "ask Sam to call the office to confirm. Do not invent a phone number."
)


@dataclass(frozen=True)
class Clip:
    text: str
    voice: Voice = "machine"
    pause_after: float = 0.0


@dataclass(frozen=True)
class Step:
    name: str
    clips: tuple[Clip, ...]
    category: Category | None
    reply: str = ""
    dtmf: str = ""
    pause_before: float = 0.0
    observe_for: float = 1.0
    advance_on_start: bool = False


@dataclass(frozen=True)
class Scenario:
    name: str
    steps: tuple[Step, ...]
    category: Category
    reason: str = "finished"
    voicemail_played: bool | None = None
    disconnect: bool = False
    timeout: float = 120.0
    idle_timeout: float = 10.0


SCREEN = Step(
    "screen-name",
    (Clip("Hi, this is Call Assist. Please state your name and why you are calling."),),
    "machine-screening",
    reply="Identify Alex from Acme Dental and explain the appointment confirmation call for Sam.",
)
SCREEN_AGAIN = Step(
    "screen-purpose",
    (Clip("Before I connect you, please tell me what appointment this call is about."),),
    "machine-screening",
    reply="Say the dental appointment is tomorrow at 10 AM. Answer the screener briefly.",
)
HUMAN = Step(
    "human-pickup",
    (Clip("Hello, this is Sam. Yes, I can hear you. What are you calling about?", "human"),),
    "human",
    reply="Address Sam directly and explain the dental appointment tomorrow at 10 AM.",
)
FOLLOWUP = Step(
    "human-confirmation",
    (Clip("Yes, tomorrow at ten works for me. I'll be there.", "human"),),
    None,
    reply="Acknowledge Sam's confirmation naturally. Do not keep answering a screener or menu.",
)
VM = Step(
    "voicemail",
    (Clip("Your call has been forwarded to voicemail. Please leave your message after the tone."),),
    "machine-vm",
    reply=(
        "Leave one self-contained message: Alex from Acme Dental, Sam's dental appointment "
        "tomorrow at 10 AM, and a request to call the office to confirm. Invent no phone number."
    ),
)
VM_AGAIN = Step(
    "voicemail-repeat",
    (Clip("Please continue recording your voicemail message after the tone."),),
    "machine-vm",
    observe_for=3.0,
)
MENU = Step(
    "ivr-appointments",
    (Clip("For billing, press one. To speak to Sam about appointments, press two."),),
    "machine-ivr",
    dtmf="2",
)
MENU_AGAIN = Step(
    "ivr-submenu",
    (Clip("To speak with Sam now, press three. To return to the main menu, press nine."),),
    "machine-ivr",
    dtmf="3",
)
SAVE = Step(
    "voicemail-submit",
    (
        Clip(
            "To send your recorded message and connect to Sam, press one. To delete it, press two."
        ),
    ),
    "machine-ivr",
    dtmf="1",
)
UNAVAILABLE = Step(
    "unavailable",
    (Clip("This number is no longer in service. Your call cannot be completed. Goodbye."),),
    "machine-unavailable",
    observe_for=2.0,
)
REJECTED = Step(
    "screen-rejected",
    (Clip("The person you called has declined this call. No message can be left. Goodbye."),),
    "machine-unavailable",
    observe_for=2.0,
)
MAILBOX_FULL = Step(
    "mailbox-full",
    (Clip("The mailbox is full. Your message cannot be recorded or saved. Goodbye."),),
    "machine-unavailable",
    observe_for=2.0,
)
AMBIGUOUS = Step(
    "ambiguous-fragment",
    (Clip("Uh... well..."),),
    "uncertain",
    reply="A brief clarification or introduction is acceptable. Do not invent facts.",
)

SCENARIOS = (
    Scenario("direct-human", (HUMAN, FOLLOWUP), "human", voicemail_played=False),
    Scenario("direct-unavailable", (UNAVAILABLE,), "machine-unavailable", voicemail_played=False),
    Scenario("direct-mailbox-full", (MAILBOX_FULL,), "machine-unavailable", voicemail_played=False),
    Scenario("screening-human", (SCREEN, SCREEN_AGAIN, HUMAN, FOLLOWUP), "human"),
    Scenario("screening-voicemail", (SCREEN, VM), "machine-vm", "idle_timeout", True),
    Scenario(
        "screening-rejected", (SCREEN, REJECTED), "machine-unavailable", voicemail_played=False
    ),
    Scenario("voicemail-idle", (VM,), "machine-vm", "idle_timeout", True),
    Scenario("voicemail-no-duplicate", (VM, VM_AGAIN), "machine-vm", "idle_timeout", True),
    Scenario("voicemail-failure", (VM, MAILBOX_FULL), "machine-unavailable", voicemail_played=True),
    Scenario("voicemail-menu-human", (VM, SAVE, HUMAN, FOLLOWUP), "human", voicemail_played=True),
    Scenario("ivr-human", (MENU, HUMAN, FOLLOWUP), "human"),
    Scenario("ivr-nested-human", (MENU, MENU_AGAIN, HUMAN, FOLLOWUP), "human"),
    Scenario("ivr-voicemail", (MENU, VM), "machine-vm", "idle_timeout", True),
    Scenario("ivr-unavailable", (MENU, UNAVAILABLE), "machine-unavailable"),
    Scenario(
        "voicemail-rerecord",
        (
            VM,
            Step(
                "rerecord-menu",
                (Clip("Your recording was too quiet. To record your message again, press two."),),
                "machine-ivr",
                dtmf="2",
            ),
            Step("rerecord", VM.clips, "machine-vm", reply=VM.reply),
        ),
        "machine-vm",
        "idle_timeout",
        True,
    ),
    Scenario(
        "human-interrupts-voicemail",
        (
            Step(
                "voicemail-start",
                VM.clips,
                "machine-vm",
                reply="Start a voicemail. An interrupted fragment need not contain the full message.",
                advance_on_start=True,
            ),
            Step(
                "human-interrupts",
                (
                    Clip(
                        "Hello, it's Sam! I picked up. Please stop the message and talk to me.",
                        "human",
                    ),
                ),
                "human",
                reply="Respond to Sam directly about the appointment. Do not restart the voicemail.",
                pause_before=0.8,
            ),
            FOLLOWUP,
        ),
        "human",
        voicemail_played=False,
    ),
    Scenario(
        "split-screening-to-voicemail",
        (
            SCREEN,
            Step(
                "split-rollover",
                (
                    Clip("Okay. They cannot take the call.", pause_after=0.8),
                    Clip("Please leave a voicemail message after the tone."),
                ),
                "machine-vm",
                reply=VM.reply,
            ),
        ),
        "machine-vm",
        "idle_timeout",
        True,
    ),
    Scenario(
        "delayed-voicemail-menu",
        (VM, Step("delayed-menu", SAVE.clips, "machine-ivr", dtmf="1", pause_before=20), HUMAN),
        "human",
        voicemail_played=True,
    ),
    Scenario("uncertain-then-human", (AMBIGUOUS, HUMAN, FOLLOWUP), "human"),
    Scenario(
        "uncertainty-limit",
        (
            AMBIGUOUS,
            Step("ambiguous-again", (Clip("Um... uh..."),), "uncertain", reply=AMBIGUOUS.reply),
            Step("ambiguous-last", (Clip("Hmm... well..."),), "uncertain", reply=AMBIGUOUS.reply),
        ),
        "uncertain",
        "max_uncertain_turns",
    ),
    Scenario("silent-call", (), "uncertain", "idle_timeout", False),
    Scenario("screening-idle", (SCREEN,), "machine-screening", "idle_timeout"),
    Scenario("ivr-idle", (MENU,), "machine-ivr", "idle_timeout"),
    Scenario(
        "screening-disconnect",
        (SCREEN,),
        "machine-screening",
        "participant_disconnected",
        disconnect=True,
    ),
    Scenario("silent-disconnect", (), "uncertain", "participant_disconnected", disconnect=True),
    Scenario(
        "overall-deadline",
        (SCREEN, VM),
        "machine-vm",
        "timeout",
        timeout=40.0,
        idle_timeout=45.0,
    ),
)
