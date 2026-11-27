"""livekit-plugins-rabel: the wrapper's rules on a scripted speech-to-text.

The wrapper runs against an in-memory stand-in of the Rabel API (version 3 of its conversation).
The tests check its rules, that it fails open (API down, slow, refusing, erroring: the events are
exactly the inner speech-to-text's), and, in a session made of LiveKit's own fakes, that a right
guess starts the agent's reply sooner and a wrong one never later.

No network: the stand-in is handed to the plugin as its `http_session`, and its WebSocket is a pair
of queues. The tests run on the virtual-time loop (tests/virtual_time.py), whose clock the plugin
reads through `time.perf_counter()`. The scripts' words are made up.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AgentStateChangedEvent,
    LanguageCode,
    UserStateChangedEvent,
    llm,
    stt,
)
from livekit.agents.testing import fake_job_context
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, TimedString
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.plugins import rabel
from livekit.plugins.rabel import stt as rabel_stt
from livekit.plugins.rabel.stt import words_of

from .fake_llm import FakeLLM
from .fake_session import FakeActions, create_session, run_session

SESSION_TIMEOUT = 60.0

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]


@pytest.fixture
def four_early_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The defaults the plugin assumes outside a job, set at the README's `max_retries` 4.

    Every test that streams through the wrapper uses it, except the two that check LiveKit's own
    defaults.
    """
    monkeypatch.setattr(rabel_stt, "_DEFAULT_LIMITS", rabel_stt._HostLimits(max_starts=4))


T = stt.SpeechEventType


# -------------------------------------------------- a scripted inner speech-to-text

# in a script: LiveKit commits a turn, which the wrapper sees as the caller's message added to the
# conversation
COMMIT = "livekit_commit"


def ev(at: float, typ: Any, text: str | None = None, conf: float | None = None) -> tuple:
    return (at, typ, text, conf)


class ScriptedSTT(stt.STT):
    """Emits its script in order.

    Each entry: seconds after the first audio frame, event type, text, last word's confidence.
    """

    def __init__(self, script: list[tuple], language: str = "en") -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self.script = script
        self.language = language

    @property
    def provider(self) -> str:
        return "Scripted"

    @property
    def model(self) -> str:
        return "Script-1"

    async def _recognize_impl(
        self, buffer: Any, *, language: Any = NOT_GIVEN, conn_options: Any = None
    ) -> stt.SpeechEvent:
        raise NotImplementedError

    def stream(
        self, *, language: Any = NOT_GIVEN, conn_options: Any = DEFAULT_API_CONNECT_OPTIONS
    ) -> ScriptedStream:
        return ScriptedStream(stt=self, conn_options=conn_options)


class ScriptedStream(stt.RecognizeStream):
    async def _run(self) -> None:
        await self._input_ch.recv()
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for at, typ, text, conf in self._stt.script:  # type: ignore[attr-defined]
            if typ == COMMIT:
                continue
            wait = t0 + at - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            alts = []
            if text is not None:
                words = [TimedString(w, confidence=0.99) for w in text.split()]
                if words and conf is not None:
                    words[-1] = TimedString(str(words[-1]), confidence=conf)
                alts = [
                    stt.SpeechData(
                        language=LanguageCode(getattr(self._stt, "language", "en")),
                        text=text,
                        words=words,
                        confidence=0.9,
                    )
                ]
            self._event_ch.send_nowait(stt.SpeechEvent(type=typ, alternatives=alts))
        async for _ in self._input_ch:
            pass


def user_item(text: str) -> ConversationItemAddedEvent:
    return ConversationItemAddedEvent(item=llm.ChatMessage(role="user", content=[text or ""]))


# -------------------------------------------------- an in-memory stand-in of the Rabel API


class FakeWS:
    """The client's end of one conversation: what it sends goes to the stand-in at once, what the
    stand-in sends waits in a queue for `receive()`."""

    def __init__(self, api: FakeAPI) -> None:
        self._api = api
        self._in: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.closed = False
        self.close_code: int | None = None
        self.got_hello = False
        # half-open: the line died without a close; sends and pings vanish and nothing comes back
        self.half_open = False

    async def send_str(self, data: str) -> None:
        if self.closed:
            raise ConnectionResetError("closed")
        if not self.half_open:
            self._api.on_message(self, json.loads(data))

    async def receive(self) -> aiohttp.WSMessage:
        return await self._in.get()

    async def ping(self, message: bytes = b"") -> None:
        if self.closed:
            raise ConnectionResetError("closed")
        if not self.half_open:  # the API's WebSocket server answers a ping at once
            self._in.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.PONG, message, None))

    async def pong(self, message: bytes = b"") -> None:
        if self.closed:
            raise ConnectionResetError("closed")

    async def close(self) -> bool:
        self._end()
        return True

    def push(self, m: dict | str) -> None:
        if not self.closed:
            data = m if isinstance(m, str) else json.dumps(m)
            self._in.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, data, None))

    def _end(self) -> None:
        if not self.closed:
            self.closed = True
            self.close_code = aiohttp.WSCloseCode.OK
            self._in.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None))


class FakeSession:
    """Stands in for the `aiohttp.ClientSession` the plugin is given. With no API, connecting fails
    as a refused connection does."""

    closed = False

    def __init__(self, api: FakeAPI | None) -> None:
        self._api = api

    async def ws_connect(self, url: str, **kw: Any) -> FakeWS:
        if self._api is None:
            raise aiohttp.ClientConnectionError("connection refused")
        return self._api.connect()


class FakeAPI:
    """v: 3 answers as the API serves today (`ready` with `"v": 3`); 2 as an API that answers an
    older version (`ready` with `"v": 2`). min_chance: the `ready`'s, else the hello's. mode: "ok"
    (guess `answers[heard words]` after `delay` s), "refuse" (an error message, then close),
    "garbage" (broken JSON, a guess for words never sent and a 400), "half-open" (`ready`, then
    the connection goes half-open: nothing sent arrives, pings go unanswered, no close comes).
    `plan` gives a mode for each conversation in turn, the last repeating."""

    def __init__(
        self,
        mode: str = "ok",
        answers: dict[str, str] | None = None,
        delay: float = 0.02,
        status: int = 402,
        error: str = "no_seconds",
        retry_after: float = 60,
        ready_after: float = 0.0,
        v: int = 3,
        chance: float = 0.86,
        send_chance: bool = True,
        min_chance: float | None = None,
        plan: list[str] | None = None,
    ) -> None:
        self.answers, self.delay = answers or {}, delay
        self.status, self.error, self.retry_after = status, error, retry_after
        self.ready_after, self.v = ready_after, v
        self.chance, self.send_chance, self.min_chance = chance, send_chance, min_chance
        self.plan = plan or [mode]
        self.received: list[dict] = []
        self.hello: dict | None = None
        self.connected_at: list[float] = []

    def connect(self) -> FakeWS:
        self.connected_at.append(asyncio.get_running_loop().time())
        return FakeWS(self)

    def _mode(self) -> str:
        return self.plan[min(len(self.connected_at), len(self.plan)) - 1]

    def _ready(self) -> dict:
        # `min_chance` as the hello set it, null when it set none: Bell's balanced setting
        min_chance = self.min_chance
        if min_chance is None:
            min_chance = (self.hello or {}).get("min_chance")
        return {"type": "ready", "v": self.v, "min_chance": min_chance, "hold_line": 0.9}

    def _guess(self, text: str, heard: str) -> dict:
        g = {"type": "guess", "turn": 0, "id": 1, "text": text, "heard": heard}
        if self.send_chance:
            g["chance"] = self.chance
        return g

    def on_message(self, ws: FakeWS, m: dict) -> None:
        loop = asyncio.get_running_loop()
        mode = self._mode()
        if not ws.got_hello:
            ws.got_hello = True
            self.hello = m
            if mode == "refuse":
                ws.push(
                    {
                        "type": "error",
                        "status": self.status,
                        "error": self.error,
                        "message": "refused",
                        "retry_after_s": self.retry_after,
                    }
                )
                ws._end()
            else:
                loop.call_later(self.ready_after, ws.push, self._ready())
                if mode == "half-open":
                    loop.call_later(self.ready_after, setattr, ws, "half_open", True)
            return
        self.received.append(m)
        if mode == "garbage":
            ws.push("{not json")
            ws.push(self._guess("words never heard", "nothing"))
            ws.push({"type": "error", "status": 400, "error": "bad_message", "message": "x"})
            ws._end()
            return
        if m.get("type") == "heard" and m["text"] in self.answers:
            loop.call_later(self.delay, ws.push, self._guess(self.answers[m["text"]], m["text"]))


async def collect(s: stt.STT, settle: float = 0.3) -> list[tuple]:
    stream = s.stream()
    if isinstance(s, rabel.STT):
        await asyncio.sleep(0.2)  # the API's conversation is ready before the first words
    stream.push_frame(
        rtc.AudioFrame(
            data=b"\x00\x00" * 160, sample_rate=16000, num_channels=1, samples_per_channel=160
        )
    )
    out: list[tuple] = []

    async def read() -> None:
        async for e in stream:
            out.append((e.type, e.alternatives[0].text if e.alternatives else None))

    reader = asyncio.create_task(read())
    script = s.wrapped_stt.script if isinstance(s, rabel.STT) else s.script  # type: ignore[attr-defined]
    if isinstance(s, rabel.STT):
        loop = asyncio.get_running_loop()
        for at, typ, text, _ in script:
            if typ == COMMIT:
                loop.call_later(at, s._push_conversation_item, user_item(text))
    await asyncio.sleep(max(at for at, *_ in script) + settle)
    stream.end_input()
    await asyncio.wait_for(reader, 5)
    await stream.aclose()
    return out


def wrapper(script: list[tuple], api: FakeAPI | None, **kw: Any) -> rabel.STT:
    return rabel.STT(
        ScriptedSTT(script),
        api_key=kw.pop("api_key", "test"),
        base_url="https://api.example",
        http_session=FakeSession(api),  # type: ignore[arg-type]
        **kw,
    )


async def wrapped_run(script: list[tuple], api: FakeAPI | None, settle: float = 0.3, **kw: Any):
    """The events out, the stream's counts and summary, and the API's end_of_turn messages."""
    w = wrapper(script, api, **kw)
    out = await collect(w, settle=settle)
    stream = next(iter(w._streams), None)
    ends = [m for m in api.received if m["type"] == "end_of_turn"] if api else []
    return out, (stream.counts if stream else {}), (stream.summary if stream else {}), ends


async def plain_run(script: list[tuple]) -> list[tuple]:
    return await collect(ScriptedSTT(script))


TURN = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like", 0.95),
    ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my", 0.95),
    ev(0.80, T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.", 0.95),
    ev(0.95, T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
    ev(0.95, T.END_OF_SPEECH),
]
TWO_TURNS = [
    *TURN,
    ev(1.40, T.START_OF_SPEECH),
    ev(1.45, T.INTERIM_TRANSCRIPT, "and my", 0.95),
    ev(1.70, T.INTERIM_TRANSCRIPT, "and my savings", 0.95),
    ev(1.90, T.FINAL_TRANSCRIPT, "And my savings."),
    ev(1.90, T.END_OF_SPEECH),
]
# the same with LiveKit's commit of the first turn before the second begins (an early reply waits
# for that commit, and rule 8 holds a guess until it)
TWO_TURNS_COMMITTED = sorted(
    [*TWO_TURNS, ev(1.20, COMMIT, "I'd like to check my balance.")], key=lambda e: e[0]
)
GUESS_BALANCE = "i'd like to check my balance"
AT_CHECK = {"I'd like to check": GUESS_BALANCE}


def hello_v3(**kw: Any) -> dict:
    """the hello the hookup sends: `"type": "hello"` and `"v": 3`; `min_chance` only when the
    developer set one"""
    return {
        "key": "test",
        "type": "hello",
        "v": 3,
        "agent": "default",
        "host": "livekit",
        "learn": True,
        **kw,
    }


# -------------------------------------------------- the hello and the options


@pytest.mark.usefixtures("four_early_starts")
async def test_hello_announces_version_3_and_options_change_from_the_next_stream() -> None:
    api = FakeAPI()
    w = wrapper(TURN, api, agent="front-desk")
    await collect(w)
    assert api.hello == hello_v3(agent="front-desk")
    w.update_options(agent="demo-bank", min_chance=0.9, learn=False)
    await collect(w)
    assert api.hello == hello_v3(agent="demo-bank", min_chance=0.9, learn=False)


@pytest.mark.parametrize(
    "kw",
    [
        {"agent": "Front Desk"},
        {"min_chance": "0.8"},
        {"answer_timeout": 0},
        {"learn": "no"},
        {"report": None},
    ],
    ids=lambda kw: "-".join(kw),
)
def test_options_are_checked(kw: dict) -> None:
    with pytest.raises(ValueError):
        rabel.STT(ScriptedSTT(TURN), api_key="test", **kw)
    w = rabel.STT(ScriptedSTT(TURN), api_key="test", agent="demo-bank")
    with pytest.raises(ValueError):
        w.update_options(**kw)
    assert w._agent == "demo-bank"


# -------------------------------------------------- the rules


@pytest.mark.usefixtures("four_early_starts")
async def test_guess_sent_once_beginnings_held_eager_equal_held() -> None:
    api = FakeAPI(answers=AT_CHECK)
    out, counts, summary, ends = await wrapped_run(TURN, api)
    # one preflight from the guess; the interim that is a beginning of it and the speech-to-text's
    # own preflight with the same words are held back; the final and end of speech pass
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "I'd like"),
        (T.INTERIM_TRANSCRIPT, "I'd like to check"),
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance."),
        (T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["guess_sent"] == 1 and counts["interim_held_back"] == 1
    assert counts["eager_held_back"] == 1
    heard = [m for m in api.received if m["type"] == "heard"]
    assert all(set(m) == {"type", "text"} for m in heard), heard  # only the words
    # sent at 0.32 s (the words at 0.30 s, the answer 0.02 s later), the final at 0.95 s
    assert ends == [
        {
            "type": "end_of_turn",
            "text": "I'd like to check my balance.",
            "guess": "kept",
            "head_start_ms": 630,
        }
    ]
    assert summary == {"turns": 1, "guesses": 1, "kept": 1, "dropped": 0, "head_start_ms": 630}


@pytest.mark.parametrize(
    "api_kw, stt_kw, dropped",
    [
        pytest.param(
            {"chance": 0.75, "min_chance": 0.8}, {}, "guess_below_min_chance", id="under-ready"
        ),
        pytest.param({"send_chance": False}, {}, "guess_no_chance", id="no-chance"),
        pytest.param({"delay": 0.3}, {"answer_timeout": 0.1}, "guess_slow", id="slow"),
    ],
)
@pytest.mark.usefixtures("four_early_starts")
async def test_wrong_or_slow_guess_not_used(
    api_kw: dict, stt_kw: dict, dropped: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    out, counts, _, ends = await wrapped_run(TURN, FakeAPI(answers=AT_CHECK, **api_kw), **stt_kw)
    assert out == await plain_run(TURN)
    assert counts[dropped] == 1 and "guess_sent" not in counts
    assert [m["guess"] for m in ends] == ["none"]
    lines = [r.getMessage() for r in caplog.records if r.name == "livekit.plugins.rabel"]
    assert len(lines) == 1, lines


@pytest.mark.usefixtures("four_early_starts")
async def test_a_final_after_the_end_of_speech_ends_the_turn() -> None:
    # OpenAI's realtime models with local VAD end speech first and send the final after it
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
        ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my", 0.95),
        ev(0.80, T.END_OF_SPEECH),
        ev(0.95, T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
    ]
    out, _, summary, ends = await wrapped_run(script, FakeAPI(answers=AT_CHECK))
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "I'd like"),
        (T.INTERIM_TRANSCRIPT, "I'd like to check"),
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance."),
        (T.END_OF_SPEECH, None),
        (T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
    ]
    # one turn, its final words, and Bell's guess kept
    assert ends == [
        {
            "type": "end_of_turn",
            "text": "I'd like to check my balance.",
            "guess": "kept",
            "head_start_ms": 630,
        }
    ]
    assert summary["turns"] == 1 and summary["kept"] == 1


@pytest.mark.usefixtures("four_early_starts")
async def test_a_turn_of_finals_only_reaches_bell() -> None:
    # a speech-to-text that sends no interims (Gnani): no heard, but the turn's end with its words
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.40, T.FINAL_TRANSCRIPT, "Please check my balance."),
        ev(0.40, T.END_OF_SPEECH),
    ]
    api = FakeAPI()
    out, _, _, ends = await wrapped_run(script, api)
    assert out == [(typ, text) for _, typ, text, _ in script]
    assert [m["type"] for m in api.received] == ["end_of_turn"]
    assert ends == [{"type": "end_of_turn", "text": "Please check my balance.", "guess": "none"}]


@pytest.mark.usefixtures("four_early_starts")
async def test_contradicting_interim_passes_and_eager_replaces() -> None:
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "check my", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "check my savings", 0.95),
        ev(0.55, T.PREFLIGHT_TRANSCRIPT, "Check my savings.", 0.95),
        ev(0.70, T.FINAL_TRANSCRIPT, "Check my savings."),
        ev(0.70, T.END_OF_SPEECH),
    ]
    out, counts, _, ends = await wrapped_run(
        script, FakeAPI(answers={"check my": "check my balance"})
    )
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "check my"),
        (T.PREFLIGHT_TRANSCRIPT, "check my balance."),
        (T.INTERIM_TRANSCRIPT, "check my savings"),
        (T.PREFLIGHT_TRANSCRIPT, "Check my savings."),
        (T.FINAL_TRANSCRIPT, "Check my savings."),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["guess_cancelled_by_words"] == 1
    assert [(m["guess"], "head_start_ms" in m) for m in ends] == [("dropped", False)]


@pytest.mark.parametrize(
    "final, re_sent", [("check my balance.", True), ("Check my balance.", False)]
)
@pytest.mark.usefixtures("four_early_starts")
async def test_final_equal_to_a_cancelled_guess_goes_first_as_an_early_start(
    final: str, re_sent: bool
) -> None:
    # LiveKit remembers the cancelled guess's text and starts no early reply at a final equal to it
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "check my", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "check my bills", 0.95),
        ev(0.50, T.INTERIM_TRANSCRIPT, "check my balance", 0.95),
        ev(0.70, T.FINAL_TRANSCRIPT, final),
        ev(0.70, T.END_OF_SPEECH),
    ]
    out, counts, _, _ = await wrapped_run(script, FakeAPI(answers={"check my": "check my balance"}))
    tail = [(T.PREFLIGHT_TRANSCRIPT, final)] if re_sent else []
    assert out[-(len(tail) + 2) :] == [*tail, (T.FINAL_TRANSCRIPT, final), (T.END_OF_SPEECH, None)]
    assert counts.get("final_sent_as_early_start", 0) == len(tail) and counts["guess_sent"] == 1


# LiveKit starts its own early reply on BEFORE's final, which waits for its end of turn
BEFORE = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(0.30, T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
    ev(0.30, T.END_OF_SPEECH),
]
AFTER = [
    ev(0.60, T.START_OF_SPEECH),
    ev(0.65, T.INTERIM_TRANSCRIPT, "and my", 0.95),
    ev(0.95, T.INTERIM_TRANSCRIPT, "and my savings", 0.95),
    ev(1.20, T.FINAL_TRANSCRIPT, "And my savings please."),
    ev(1.20, T.END_OF_SPEECH),
]


@pytest.mark.parametrize(
    "commit_at, sent, count",
    [
        pytest.param(None, False, "held_guess_not_sent_turn_over", id="livekit-reply-waits"),
        pytest.param(0.80, False, "held_guess_not_sent_commit", id="commit-after-the-guess"),
        pytest.param(0.45, True, None, id="commit-before-the-guess"),
    ],
)
@pytest.mark.usefixtures("four_early_starts")
async def test_guess_held_while_livekits_own_reply_waits(
    commit_at: float | None, sent: bool, count: str | None
) -> None:
    script = [*BEFORE, *AFTER]
    if commit_at is not None:
        script = sorted(
            [*script, ev(commit_at, COMMIT, "I'd like to check my balance.")], key=lambda e: e[0]
        )
    out, counts, _, _ = await wrapped_run(
        script, FakeAPI(answers={"and my": "and my savings please"})
    )
    guesses = [e for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT]
    if sent:
        assert guesses == [(T.PREFLIGHT_TRANSCRIPT, "and my savings please.")]
    else:
        assert guesses == [] and counts["guess_held_while_livekit_waits"] == 1
        assert count is not None and counts[count] == 1
        assert out == [(typ, text) for _, typ, text, _ in [*BEFORE, *AFTER]]


@pytest.mark.usefixtures("four_early_starts")
async def test_no_guess_as_third_early_start_and_none_equal_to_the_running_one() -> None:
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.PREFLIGHT_TRANSCRIPT, "yes", 0.95),
        ev(0.20, T.PREFLIGHT_TRANSCRIPT, "yes please", 0.95),
        ev(0.35, T.INTERIM_TRANSCRIPT, "yes please do", 0.95),
        ev(0.70, T.FINAL_TRANSCRIPT, "Yes, please do it."),
        ev(0.70, T.END_OF_SPEECH),
    ]
    api = FakeAPI(answers={"yes please do": "yes please do it", "yes": "yes"})
    out, counts, _, _ = await wrapped_run(script, api)
    # the guess "yes" equals the speech-to-text's own running early start; the later one would take
    # LiveKit's spare
    assert counts["guess_no_spare_start"] == 1 and counts["guess_same_as_running"] == 1
    assert out == await plain_run(script)


# -------------------------------------------------- fails open


@pytest.mark.parametrize(
    "case, lines_expected",
    [
        ("down", 1),
        ("disabled", 0),
        ("no-key", 1),
        ("slow-ready", 0),
        ("slow-guess", 0),
        ("refused-402", 1),
        ("refused-503", 1),
        ("garbage", 2),
    ],
)
@pytest.mark.usefixtures("four_early_starts")
async def test_fails_open(case: str, lines_expected: int, caplog: pytest.LogCaptureFixture) -> None:
    plain = await plain_run(TURN)
    answers = {"I'd like": GUESS_BALANCE, **AT_CHECK}
    api: FakeAPI | None = {
        "down": None,
        "slow-ready": FakeAPI(answers=answers, ready_after=5.0),
        "slow-guess": FakeAPI(answers=answers, delay=2.0),
        "refused-402": FakeAPI(mode="refuse", status=402, error="no_seconds"),
        "refused-503": FakeAPI(mode="refuse", status=503, error="waking", retry_after=60),
        "garbage": FakeAPI(mode="garbage"),
    }.get(case, FakeAPI(answers=answers))
    kw: dict[str, Any] = {"enabled": False} if case == "disabled" else {}
    if case == "no-key":
        kw["api_key"] = ""
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    settle = {"slow-guess": 2.5, "down": 3.0}.get(case, 0.5)
    out, *_ = await wrapped_run(TURN, api, settle=settle, **kw)
    assert out == plain, case
    lines = [r.getMessage() for r in caplog.records if r.name == "livekit.plugins.rabel"]
    assert len(lines) == lines_expected, lines


@pytest.mark.usefixtures("four_early_starts")
async def test_reopens_after_503_with_backoff_and_not_after_400() -> None:
    api = FakeAPI(plan=["refuse", "ok"], status=503, error="waking", retry_after=0)
    await wrapped_run(TWO_TURNS, api)
    at = [t - api.connected_at[0] for t in api.connected_at]
    assert len(at) == 2 and at[1] == pytest.approx(1.0, abs=0.01), at
    assert ("end_of_turn", "And my savings.") in [(m["type"], m.get("text")) for m in api.received]
    api = FakeAPI(plan=["refuse"], status=400, error="bad_message", retry_after=0)
    await wrapped_run(TWO_TURNS, api)
    assert len(api.connected_at) == 1


LATE_TURN = [(at + 20.0, *rest) for at, *rest in TURN]  # a turn after 20 s of a quiet line


@pytest.mark.usefixtures("four_early_starts")
async def test_a_dead_connection_after_ready_is_noticed_and_reopened(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    # a healthy line that stays quiet for 20 s answers the pings, stays open and guesses
    api = FakeAPI(answers=AT_CHECK)
    late, *_ = await wrapped_run(LATE_TURN, api)
    assert late != await plain_run(LATE_TURN) and len(api.connected_at) == 1
    assert not [r for r in caplog.records if r.name == "livekit.plugins.rabel"]
    # a half-open one: the first turn (committed) passes through unchanged; no pong within the
    # liveness timeout (15 s after ready) gives one "down" line, and the back-off's first wait
    # (1 s) later a new conversation guesses the late turn as the healthy line did
    api = FakeAPI(plan=["half-open", "ok"], answers=AT_CHECK)
    first = [*TURN, ev(1.20, COMMIT, "I'd like to check my balance.")]
    out, *_, ends = await wrapped_run([*first, *LATE_TURN], api)
    at = [t - api.connected_at[0] for t in api.connected_at]
    assert len(at) == 2 and at[1] == pytest.approx(16.0, abs=0.01), at
    assert [e["text"] for e in ends] == ["I'd like to check my balance."]
    assert out == await plain_run(TURN) + late
    lines = [r.getMessage() for r in caplog.records if r.name == "livekit.plugins.rabel"]
    assert len(lines) == 1 and "APIConnectionError" in lines[0], lines


# -------------------------------------------------- the report and the two log lines


@pytest.mark.parametrize("report", [True, False], ids=["report-on", "report-off"])
@pytest.mark.usefixtures("four_early_starts")
async def test_log_lines_written_with_the_report_on_or_off_and_carry_no_words(
    report: bool, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="livekit.plugins.rabel")
    answers = {**AT_CHECK, "and my": "and my savings please"}
    api = FakeAPI(answers=answers)
    out, _, summary, ends = await wrapped_run(TWO_TURNS_COMMITTED, api, report=report)
    if report:
        assert [m["guess"] for m in ends] == ["kept", "dropped"]
    else:  # the API is told nothing of the guesses
        assert ends == [
            {"type": "end_of_turn", "text": "I'd like to check my balance."},
            {"type": "end_of_turn", "text": "And my savings."},
        ]
    recs = [r for r in caplog.records if r.name == "livekit.plugins.rabel"]
    kept = [r for r in recs if r.getMessage() == "Bell's guess kept"]
    assert len(kept) == 1 and kept[0].levelno == logging.DEBUG and kept[0].head_start_ms == 630
    totals = [r for r in recs if "head start in all" in r.getMessage()]
    assert len(totals) == 1 and totals[0].levelno == logging.INFO
    assert (
        totals[0].getMessage()
        == "Bell: 2 turns, 2 guesses, 1 kept, 1 dropped, 630 ms head start in all"
    )
    assert summary == {"turns": 2, "guesses": 2, "kept": 1, "dropped": 1, "head_start_ms": 630}
    said: set[str] = set()
    for _, _, text, _ in TWO_TURNS:
        said |= set(words_of(text))
    for g in answers.values():
        said |= set(words_of(g))
    base = set(vars(logging.makeLogRecord({})))
    for r in recs:
        assert not set(words_of(r.getMessage())) & said, r.getMessage()
        for k, val in vars(r).items():
            if k not in base and isinstance(val, str):
                assert not set(words_of(val)) & said, (k, val)


@pytest.mark.usefixtures("four_early_starts")
async def test_report_switched_off_at_once() -> None:
    api = FakeAPI(answers=AT_CHECK)
    w = wrapper(TWO_TURNS, api)
    stream = w.stream()
    await asyncio.sleep(0.2)
    stream.push_frame(
        rtc.AudioFrame(
            data=b"\x00\x00" * 160, sample_rate=16000, num_channels=1, samples_per_channel=160
        )
    )

    async def read() -> None:
        async for _ in stream:
            pass

    reader = asyncio.create_task(read())
    await asyncio.sleep(1.2)  # the first turn has ended; the second has not begun
    w.update_options(report=False)
    await asyncio.sleep(1.0)
    stream.end_input()
    await asyncio.wait_for(reader, 5)
    await stream.aclose()
    ends = [m for m in api.received if m["type"] == "end_of_turn"]
    assert [m.get("guess") for m in ends] == ["kept", None]


# -------------------------------------------------- guessing again, and the end bet

# Bell's first guess "...check my balance" is cancelled by the words ("...check my savings"); Bell
# guesses again from those words, and the new guess is the next early start
AGAIN = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check my savings", 0.95),
    ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my savings account", 0.95),
    ev(0.80, T.FINAL_TRANSCRIPT, "I'd like to check my savings account."),
    ev(0.80, T.END_OF_SPEECH),
]
AGAIN_ANSWERS = {
    "I'd like to check": GUESS_BALANCE,
    "I'd like to check my savings": "i'd like to check my savings account",
}


@pytest.mark.usefixtures("four_early_starts")
async def test_an_api_that_answers_a_version_2_ready_leaves_the_events_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    api = FakeAPI(answers={"I'd like": GUESS_BALANCE, **AT_CHECK}, v=2)
    out, counts, _, _ = await wrapped_run(TURN, api, settle=1.5)
    assert out == await plain_run(TURN)
    assert "guess_sent" not in counts
    # no words are sent and no other conversation is opened
    assert api.received == [] and len(api.connected_at) == 1
    lines = [r.getMessage() for r in caplog.records if r.name == "livekit.plugins.rabel"]
    assert len(lines) == 1, lines


@pytest.mark.usefixtures("four_early_starts")
async def test_a_new_guess_after_the_words_cancelled_the_first_is_sent_and_kept() -> None:
    out, counts, summary, ends = await wrapped_run(AGAIN, FakeAPI(answers=AGAIN_ANSWERS))
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "I'd like to check"),
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance."),
        (T.INTERIM_TRANSCRIPT, "I'd like to check my savings"),
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my savings account."),
        (T.INTERIM_TRANSCRIPT, "I'd like to check my savings account"),
        (T.FINAL_TRANSCRIPT, "I'd like to check my savings account."),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["guess_sent"] == 2 and counts["guess_cancelled_by_words"] == 1
    # the report: the last guess kept, the one before it dropped
    assert ends[-1]["guess"] == "kept" and ends[-1]["dropped_before"] == 1
    assert (summary["guesses"], summary["kept"], summary["dropped"]) == (2, 1, 1)


@pytest.mark.usefixtures("four_early_starts")
async def test_one_guess_alive_and_never_the_last_early_start() -> None:
    # a second guess that comes while the first is the running early start, the words still agreeing
    # with it
    api = FakeAPI(answers={**AT_CHECK, "I'd like to check my": "i'd like to check my card"})
    out, counts, _, _ = await wrapped_run(TURN, api)
    assert [e for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT] == [
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.")
    ]
    assert counts["guess_sent"] == 1 and counts["guess_while_live"] == 1
    # two guesses each cancelled by the words have taken two of LiveKit's three early starts: a
    # third is not sent, so LiveKit has one left for the final
    script = [
        *AGAIN[:3],
        ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my savings balance", 0.95),
        ev(0.80, T.FINAL_TRANSCRIPT, "I'd like to check my savings balance today."),
        ev(0.80, T.END_OF_SPEECH),
    ]
    answers = {
        **AGAIN_ANSWERS,
        "I'd like to check my savings balance": "i'd like to check my savings balance today",
    }
    out, counts, _, _ = await wrapped_run(script, FakeAPI(answers=answers))
    assert counts["guess_sent"] == 2 and counts["guess_cancelled_by_words"] == 2
    assert counts["guess_no_spare_start"] == 1
    assert (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my savings balance today.") not in out


async def test_at_livekits_default_bell_guesses_once_a_turn() -> None:
    # LiveKit's default of 3: one guess leaves two starts (the final's and a spare); a second would
    # leave one
    out, counts, _, _ = await wrapped_run(AGAIN, FakeAPI(answers=AGAIN_ANSWERS))
    assert counts["guess_sent"] == 1 and counts["guess_no_spare_start"] == 1


@pytest.mark.parametrize("committed", [False, True], ids=["not-committed", "committed"])
@pytest.mark.usefixtures("four_early_starts")
async def test_the_count_runs_until_livekit_commits(committed: bool) -> None:
    # a wrong guess (start 1) and LiveKit's own start at the first final (start 2); the caller
    # pauses and goes on. Not committed, LiveKit's turn runs on with two starts used and a guess
    # would leave it one; committed, the count starts again
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check my savings", 0.95),
        ev(0.50, T.FINAL_TRANSCRIPT, "I'd like to check my savings."),
        ev(0.50, T.END_OF_SPEECH),
        *([ev(0.70, COMMIT, "I'd like to check my savings.")] if committed else []),
        ev(0.90, T.START_OF_SPEECH),
        ev(0.95, T.INTERIM_TRANSCRIPT, "and my", 0.95),
        ev(1.20, T.INTERIM_TRANSCRIPT, "and my card", 0.95),
        ev(1.40, T.FINAL_TRANSCRIPT, "And my card."),
        ev(1.40, T.END_OF_SPEECH),
    ]
    answers = {"I'd like to check": "i'd like to check my balance", "and my": "and my card"}
    out, counts, _, _ = await wrapped_run(script, FakeAPI(answers=answers))
    if committed:
        assert counts["guess_sent"] == 2 and "guess_no_spare_start" not in counts
    else:
        assert counts["guess_sent"] == 1 and counts["guess_no_spare_start"] == 1


@pytest.mark.usefixtures("four_early_starts")
async def test_the_end_bet_is_sent_on_the_words_so_far() -> None:
    # a guess equal to the words it was made from: Bell bets the turn is over there. It is sent as
    # the early start on the words exactly as the speech-to-text showed them, and kept when the turn
    # ends on them
    said = "I'd like to check my balance."
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, said, 0.95),
        ev(0.40, T.FINAL_TRANSCRIPT, said),
        ev(0.40, T.END_OF_SPEECH),
    ]
    out, counts, _, ends = await wrapped_run(script, FakeAPI(answers={said: GUESS_BALANCE}))
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, said),
        (T.PREFLIGHT_TRANSCRIPT, said),
        (T.FINAL_TRANSCRIPT, said),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["end_bet_sent"] == 1 and ends[-1]["guess"] == "kept"


# -------------------------------------------------- the session's own settings (rule 6)


def in_session(monkeypatch: pytest.MonkeyPatch, session: Any) -> None:
    """The job's primary session, as `get_job_context` gives it to the plugin inside a running
    job."""
    monkeypatch.setattr(
        rabel_stt,
        "get_job_context",
        lambda *, required=True: SimpleNamespace(primary_session=session),
    )


def session_with(**turn_handling: Any) -> AgentSession:
    return AgentSession(turn_handling=turn_handling)  # type: ignore[arg-type]


def session_like(session: AgentSession, **kw: Any) -> SimpleNamespace:
    """What the plugin reads of a session, with the parts a test sets (an agent, its reply under
    way)."""
    base: dict[str, Any] = {
        "options": session.options,
        "turn_detection": session.turn_detection,
        "llm": None,
        "current_agent": None,
        "current_speech": None,
    }
    return SimpleNamespace(**{**base, **kw})


async def test_the_sessions_settings_are_read_and_the_agents_come_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits, read = rabel_stt._HostLimits, rabel_stt._host_limits
    defaults = limits(early_replies=True, max_starts=3, max_speech_s=10.0, agent_speaking=False)
    # no job (a developer's own script): LiveKit's defaults
    assert read() == defaults
    in_session(monkeypatch, session_with())
    assert read() == defaults

    in_session(
        monkeypatch,
        session_with(preemptive_generation={"max_retries": 1, "max_speech_duration": 4}),
    )
    assert read() == limits(max_starts=1, max_speech_s=4.0)

    # where LiveKit starts no early reply: preemptive generation off, turns ended by hand, a
    # realtime model
    in_session(monkeypatch, session_with(preemptive_generation={"enabled": False}))
    assert read().early_replies is False
    in_session(monkeypatch, session_with(turn_detection="manual"))
    assert read().early_replies is False
    # turns ended by the speech-to-text
    in_session(monkeypatch, session_with(turn_detection="stt"))
    assert read().stt_turns is True and read().early_replies is True
    not_an_llm = object()  # stands for a realtime model: anything that is not an llm.LLM
    in_session(monkeypatch, session_like(session_with(), llm=not_an_llm))
    assert read().early_replies is False

    # the running agent's own settings come before the session's, as LiveKit resolves them
    agent = Agent(
        instructions="x",
        turn_handling={"preemptive_generation": {"max_retries": 1}, "turn_detection": "manual"},
    )
    session = session_with(preemptive_generation={"max_retries": 2})
    in_session(monkeypatch, session_like(session, current_agent=agent))
    assert read() == limits(early_replies=False, max_starts=1)

    # the agent's reply under way, until the caller interrupts it
    in_session(
        monkeypatch, session_like(session, current_speech=SimpleNamespace(interrupted=False))
    )
    assert read().agent_speaking is True
    in_session(monkeypatch, session_like(session, current_speech=SimpleNamespace(interrupted=True)))
    assert read().agent_speaking is False

    # a job with no session started, or settings that cannot be read: LiveKit's defaults, never an
    # error
    class NoSession:
        @property
        def primary_session(self) -> AgentSession:
            raise RuntimeError("No AgentSession was started for this job")

    monkeypatch.setattr(rabel_stt, "get_job_context", lambda *, required=True: NoSession())
    assert read() == defaults
    in_session(monkeypatch, object())
    assert read() == defaults


@pytest.mark.parametrize("turn_detection", ["stt", "vad"])
async def test_a_final_within_the_turn_holds_a_guess_only_where_livekit_replies_on_it(
    monkeypatch: pytest.MonkeyPatch, turn_detection: str
) -> None:
    in_session(
        monkeypatch,
        session_with(turn_detection=turn_detection, preemptive_generation={"max_retries": 4}),
    )
    # a speech-to-text that finalizes a segment before the turn's end (Deepgram)
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
        ev(0.40, T.FINAL_TRANSCRIPT, "I'd like to check."),
        ev(0.55, T.INTERIM_TRANSCRIPT, "my", 0.95),
        ev(0.80, T.FINAL_TRANSCRIPT, "my balance."),
        ev(0.80, T.END_OF_SPEECH),
    ]
    api = FakeAPI(answers={"I'd like to check. my": GUESS_BALANCE})
    out, counts, _, _ = await wrapped_run(script, api)
    guesses = [e for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT]
    if turn_detection == "stt":
        # with the speech-to-text's endpointing LiveKit starts no early reply on that final, so
        # none waits and the guess is sent
        assert guesses == [(T.PREFLIGHT_TRANSCRIPT, "my balance.")]
        assert "guess_held_while_livekit_waits" not in counts
    else:
        # with VAD or the turn detector LiveKit's own early reply on that final waits (rule 8)
        assert guesses == [] and counts["guess_held_while_livekit_waits"] == 1


@pytest.mark.parametrize(
    "session_kw, count",
    [
        ({"preemptive_generation": {"enabled": False}}, "guess_no_early_reply"),
        ({"turn_detection": "manual"}, "guess_no_early_reply"),
        # a guess would take the turn's only early start; were it wrong, LiveKit could start none on
        # the final words
        ({"preemptive_generation": {"max_retries": 1}}, "guess_no_spare_start"),
        ({"preemptive_generation": {"max_speech_duration": 0.2}}, "guess_past_10s"),
    ],
    ids=[
        "early-replies-off",
        "turns-ended-by-hand",
        "one-early-start-a-turn",
        "past-the-limit-of-speech",
    ],
)
@pytest.mark.usefixtures("four_early_starts")
async def test_no_guess_where_livekit_would_start_no_early_reply(
    session_kw: dict, count: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_session(monkeypatch, session_with(**session_kw))
    api = FakeAPI(answers=AT_CHECK)
    out, counts, _, ends = await wrapped_run(TURN, api)
    assert out == await plain_run(TURN)
    assert counts[count] == 1 and "guess_sent" not in counts
    # Bell still hears the turn, and the report says no guess was passed on
    assert [m["type"] for m in api.received].count("heard") == 4
    assert ends == [
        {"type": "end_of_turn", "text": "I'd like to check my balance.", "guess": "none"}
    ]


@pytest.mark.parametrize("interrupted", [False, True], ids=["speaking", "interrupted"])
@pytest.mark.usefixtures("four_early_starts")
async def test_no_guess_while_the_agents_reply_is_under_way(
    interrupted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a guess then would start nothing, and its words would count as the caller's where LiveKit
    # counts the words that interrupt the agent (`min_words`); once the caller has interrupted the
    # reply, a guess is an early start again
    speech = SimpleNamespace(interrupted=interrupted)
    in_session(monkeypatch, session_like(session_with(), current_speech=speech))
    out, counts, _, _ = await wrapped_run(TURN, FakeAPI(answers=AT_CHECK))
    if interrupted:
        assert counts["guess_sent"] == 1 and "guess_while_agent_speaks" not in counts
    else:
        assert out == await plain_run(TURN)
        assert counts["guess_while_agent_speaks"] == 1 and "guess_sent" not in counts


# -------------------------------------------------- in a session: the reply's start
#
# LiveKit's own fakes (the VAD, the LLM, the voice, the audio in and out) with the plugin around a
# scripted speech-to-text, inside a job, as `tests/test_agent_session.py` measures preemptive
# generation. The caller speaks from 0.5 s to 2.0 s; the final transcript comes 0.1 s after; the LLM
# takes 1.0 s and the voice 0.3 s to its first audio; LiveKit ends the turn 0.5 s after the caller
# stops.
# From the caller's last word to the agent's first audio:
#   Without Bell the LLM starts at the final:                 0.1 + 1.0 + 0.3 = 1.4 s.
#   On a right guess the LLM started 0.78 s before that word: 0.5 + 0.3       = 0.8 s.

SAID = "I'd like to check my balance."
SPOKEN = [
    ev(0.50, T.START_OF_SPEECH),
    ev(0.80, T.INTERIM_TRANSCRIPT, "I'd like", 0.95),
    ev(1.20, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(2.10, T.FINAL_TRANSCRIPT, SAID),
    ev(2.10, T.END_OF_SPEECH),
]
WRONG = "I'd like to check my card."


async def reply_latency(stt_: stt.STT, **turn_handling: Any) -> tuple[float, int]:
    """Seconds from the caller's last word to the agent's first audio, and the LLM requests made."""
    actions = FakeActions()
    actions.add_user_speech(0.5, 2.0, SAID, stt_delay=0.1)
    actions.add_llm("Sure, one moment.", input=SAID, ttft=0.9, duration=1.0)
    actions.add_llm("Which card?", input=WRONG, ttft=0.9, duration=1.0)
    actions.add_tts(1.0, input="Sure, one moment.", ttfb=0.3)
    actions.add_tts(1.0, input="Which card?", ttfb=0.3)
    session = create_session(actions, turn_handling=dict(turn_handling), with_stt=False)
    llm_ = session.llm
    assert isinstance(llm_, FakeLLM)
    requests = 0
    chat = llm_.chat

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal requests
        requests += 1
        return chat(*args, **kwargs)

    llm_.chat = counted  # type: ignore[method-assign]
    agent_states: list[AgentStateChangedEvent] = []
    user_states: list[UserStateChangedEvent] = []
    session.on("agent_state_changed", agent_states.append)
    session.on("user_state_changed", user_states.append)
    with fake_job_context():
        await asyncio.wait_for(
            run_session(session, Agent(instructions="x", stt=stt_)), timeout=SESSION_TIMEOUT
        )
    stopped = next(e for e in user_states if e.new_state == "listening").speech_timestamp
    spoke = next(e for e in agent_states if e.new_state == "speaking").created_at
    assert stopped is not None
    return spoke - stopped, requests


@pytest.mark.no_concurrent
@pytest.mark.usefixtures("four_early_starts")
async def test_in_a_session_a_right_guess_starts_the_reply_sooner_and_a_wrong_one_never_later() -> (
    None
):
    plain, plain_requests = await reply_latency(ScriptedSTT(SPOKEN))
    assert plain == pytest.approx(1.4, abs=0.2) and plain_requests == 1

    right = wrapper(SPOKEN, FakeAPI(answers=AT_CHECK))
    latency, requests = await reply_latency(right)
    assert latency == pytest.approx(0.8, abs=0.2) and requests == 1
    assert next(iter(right._streams)).counts["guess_sent"] == 1

    # a wrong guess: LiveKit drops its reply at the final and starts on the caller's words, as it
    # does without Bell, at the cost of one more LLM request
    wrong = wrapper(SPOKEN, FakeAPI(answers={"I'd like to check": "i'd like to check my card"}))
    latency, requests = await reply_latency(wrong)
    assert latency == pytest.approx(plain, abs=0.05) and requests == 2


@pytest.mark.no_concurrent
@pytest.mark.usefixtures("four_early_starts")
async def test_in_a_session_with_one_early_start_a_turn_a_wrong_guess_costs_nothing() -> None:
    # `max_retries` 1: a guess would take the turn's only early start, and LiveKit could start none
    # on the final words; the reply would start at LiveKit's end of turn (0.5 + 1.0 + 0.3 = 1.8 s).
    # The plugin reads the session's limit through the job and sends no guess.
    one = {"preemptive_generation": {"max_retries": 1}}
    plain, _ = await reply_latency(ScriptedSTT(SPOKEN), **one)
    wrong = wrapper(SPOKEN, FakeAPI(answers={"I'd like to check": "i'd like to check my card"}))
    latency, requests = await reply_latency(wrong, **one)
    assert latency == pytest.approx(plain, abs=0.05) and requests == 1
    counts = next(iter(wrong._streams)).counts
    assert counts["guess_no_spare_start"] == 1 and "guess_sent" not in counts


async def run_in(
    language: str, script: list[tuple], api: FakeAPI
) -> tuple[list[tuple], list[dict]]:
    """wrapped_run with a speech-to-text that says its words are in `language`"""
    w = rabel.STT(
        ScriptedSTT(script, language=language),
        api_key="test",
        base_url="https://api.example",
        http_session=FakeSession(api),  # type: ignore[arg-type]
    )
    out = await collect(w)
    return out, [m for m in api.received if m["type"] == "end_of_turn"]


# -------------------------------------------------- a language other than English
#
# Under LiveKit's default turn detection an early reply is kept only when the final transcript is
# the same string as the early start's, so the guess is written the way the final will be. For
# English that is `final_form`'s rules. For any other language it is the call's own form: how this
# call's final transcripts have been written so far.

FIRST_HI = "\u092e\u0947\u0930\u093e \u0928\u093e\u092e \u0930\u093e\u092e \u0939\u0948\u0964"
HEARD_HI = "\u092e\u0941\u091d\u0947 \u0905\u092a\u0928\u093e \u092c\u0948\u0932\u0947\u0902\u0938"
REST_HI = "\u091a\u0947\u0915 \u0915\u0930\u0928\u093e \u0939\u0948"
SAID_HI = (
    HEARD_HI + " " + REST_HI + "\u0964"
)  # the turn as the speech-to-text writes it, closed with Hindi's own mark
HINDI = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, FIRST_HI[:8], 0.95),
    ev(0.30, T.FINAL_TRANSCRIPT, FIRST_HI),
    ev(0.30, T.END_OF_SPEECH),
    ev(0.45, COMMIT, FIRST_HI),
    ev(0.60, T.START_OF_SPEECH),
    ev(0.65, T.INTERIM_TRANSCRIPT, HEARD_HI, 0.95),
    ev(1.10, T.FINAL_TRANSCRIPT, SAID_HI),
    ev(1.10, T.END_OF_SPEECH),
]


@pytest.mark.parametrize("language, kept", [("hi", True), ("en", False), ("multi", False)])
@pytest.mark.usefixtures("four_early_starts")
async def test_another_language_is_written_as_the_calls_own_finals_were(language, kept) -> None:
    api = FakeAPI(answers={HEARD_HI: HEARD_HI + " " + REST_HI})
    out, ends = await run_in(language, HINDI, api)
    guess = [text for typ, text in out if typ == T.PREFLIGHT_TRANSCRIPT]
    if kept:
        # the call's first final closed with Hindi's own mark, so the guess does, and it is the
        # final's string
        assert guess == [SAID_HI] and ends[-1]["guess"] == "kept"
    else:
        # English, or a speech-to-text that names no one language: the English rules, as before
        assert guess == [HEARD_HI + " " + REST_HI + "."] and ends[-1]["guess"] == "dropped"


def test_the_calls_own_form() -> None:
    from livekit.plugins.rabel._form import CallForm, final_form, written

    form = CallForm()
    # before any final transcript there is nothing to copy: the guess is written as `final_form`
    # writes it
    assert written("Je voudrais", ("un", "billet"), form) == final_form(
        "Je voudrais", ("un", "billet")
    )
    form.final("Je voudrais v\u00e9rifier mon solde.")
    form.final("Pouvez-vous l'emmener \u00e0 l'h\u00f4tel ?")
    # a word as this call wrote it (its apostrophe back), and the turn closed as it closed after the
    # same opening word
    assert written("Pouvez-vous", ("lemmener",), form) == "Pouvez-vous l'emmener ?"
    assert written("Je voudrais", ("un", "billet"), form) == "Je voudrais un billet."
    # an opening word not heard yet: the way most of the call's turns have closed
    form.final("Combien co\u00fbte le billet ?")
    assert written("Quel est", ("le", "prix"), form) == "Quel est le prix ?"
    # a guess that adds nothing (the end bet) is the speech-to-text's own text
    assert written("Je voudrais un billet", (), form) == "Je voudrais un billet"
    # a speech-to-text that writes no capitals and no marks is copied as it is
    plain = CallForm()
    plain.final("mera naam ram hai")
    assert written("", ("mujhe", "balance"), plain) == "mujhe balance"


def test_a_script_without_spaces_gets_none_between_the_added_words() -> None:
    from livekit.plugins.rabel._form import CallForm, written

    # LiveKit's comparison takes kana and ideographs one character at a time, so Bell's added words
    # are single characters; they are written as the call's finals are, with no space between
    ja = CallForm()
    ja.final("\u6b8b\u9ad8\u3092\u78ba\u8a8d\u3057\u305f\u3044\u3067\u3059\u3002")
    # the shown words and the added ones: the final's string exactly
    said = "\u6b8b\u9ad8\u3092\u78ba\u8a8d\u3002"
    assert written("\u6b8b\u9ad8", ("\u3092", "\u78ba", "\u8a8d"), ja) == said
    assert words_of(said) == ("\u6b8b", "\u9ad8", "\u3092", "\u78ba", "\u8a8d")
    # an added word of such a script gets no space after a word of a spaced script either, as
    # Japanese writes it; between two words of a spaced script the space stays
    assert written("ATM", ("\u3067",), ja) == "ATM\u3067\u3002"
    zh = CallForm()
    zh.final("\u6211\u60f3\u67e5\u4f59\u989d\u3002")
    assert written("\u6211\u60f3", ("\u67e5", "\u4f59", "\u989d"), zh) == (
        "\u6211\u60f3\u67e5\u4f59\u989d\u3002"
    )
    fr = CallForm()
    fr.final("Je voudrais un billet.")
    assert written("Je voudrais", ("un", "billet"), fr) == "Je voudrais un billet."


# -------------------------------------------------- the wrapped stream's flush


class FlushedStream(ScriptedStream):
    """A stream that needs a flush to finalize the turn while its STT's capability says none is
    needed, as a FallbackAdapter's stream does when its live provider needs one and another in the
    list does not."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.flushes = 0

    @property
    def manual_flush(self) -> bool:
        return True

    async def _run(self) -> None:
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                self.flushes += 1


class FlushedSTT(ScriptedSTT):
    def __init__(self) -> None:
        super().__init__([])
        self.made: list[FlushedStream] = []

    def stream(
        self, *, language: Any = NOT_GIVEN, conn_options: Any = DEFAULT_API_CONNECT_OPTIONS
    ) -> FlushedStream:
        s = FlushedStream(stt=self, conn_options=conn_options)
        self.made.append(s)
        return s


async def test_the_wrapped_streams_flush_need_is_reported_and_its_flush_passed_on() -> None:
    inner = FlushedSTT()
    assert inner.capabilities.manual_flush is False
    w = rabel.STT(
        inner,
        api_key="test",
        base_url="https://api.example",
        http_session=FakeSession(None),  # type: ignore[arg-type]
    )
    stream = w.stream()
    stream.push_frame(
        rtc.AudioFrame(
            data=b"\x00\x00" * 160, sample_rate=16000, num_channels=1, samples_per_channel=160
        )
    )
    await asyncio.sleep(0.05)
    # LiveKit reads the stream's need before it flushes at the end of a turn
    # (audio_recognition's manual_flush)
    assert stream.manual_flush is True
    stream.flush()
    await asyncio.sleep(0.05)
    assert [s.flushes for s in inner.made] == [1]
    await stream.aclose()
