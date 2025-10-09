from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mock_dashboard import (  # noqa: E402
    CloudAgentRecord,
    LatencyMetrics,
    MockLiveKitDashboard,
    format_currency,
    format_list,
)

from livekit.agents import (
    Agent,
    AgentSession,
    AgentTask,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.beta.workflows.dtmf_inputs import GetDtmfTask
from livekit.agents.llm.tool_context import ToolError
from livekit.plugins import deepgram, elevenlabs, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()


logger = logging.getLogger("agentic-ivr")


AGENTIC_IVR_DISPATCH_NAME = os.getenv("AGENTIC_IVR_DISPATCH_NAME", "livekit-agentic-ivr")


class TaskOutcome(str, Enum):
    RETURN_TO_ROOT = "return_to_root"
    END_SESSION = "end_session"
    TRANSFER_SUPPORT = "transfer_support"


@dataclass
class SessionState:
    account_id: Optional[str] = None  # noqa: UP007
    account_label: Optional[str] = None  # noqa: UP007
    project_id: Optional[str] = None  # noqa: UP007
    project_label: Optional[str] = None  # noqa: UP007
    usage_cache: dict[str, dict[str, object]] = field(default_factory=dict)
    telephony_cache: dict[str, dict[str, object]] = field(default_factory=dict)
    performance_cache: dict[str, dict[str, object]] = field(default_factory=dict)
    audit_log: list[str] = field(default_factory=list)


async def speak(agent: Agent, instructions: str, *, allow_interruptions: bool = False) -> None:
    logger.debug("prompt: %s", instructions)
    await agent.session.generate_reply(
        instructions=instructions, allow_interruptions=allow_interruptions
    )


async def collect_digits(
    agent: Agent,
    *,
    prompt: str,
    num_digits: int,
    confirmation: bool = True,
) -> str:
    while True:
        try:
            result = await GetDtmfTask(
                num_digits=num_digits,
                ask_for_confirmation=confirmation,
                chat_ctx=agent.chat_ctx.copy(exclude_instructions=True, exclude_function_call=True),
                extra_instructions=(
                    "You are confirming keyed digits for a LiveKit Cloud caller. "
                    f"Prompt them as follows and wait for {num_digits} digits.\n"
                    f"<prompt>{prompt}</prompt>"
                ),
            )
        except ToolError as exc:
            await speak(agent, exc.message if hasattr(exc, "message") else str(exc))
            continue

        return result.user_input


def build_menu_prompt(base_prompt: str, options: dict[str, str]) -> str:
    option_lines = [f"Press {digit} for {description}." for digit, description in options.items()]
    return (
        f"{base_prompt} "
        + " ".join(option_lines)
        + " After you choose, press the matching digit on your keypad."
    )


async def run_menu(
    agent: Agent,
    *,
    prompt: str,
    options: dict[str, str],
    invalid_message: str = "I did not catch that selection. Let's try again.",
) -> str:
    normalized_options = dict(options.items())

    while True:
        try:
            result = await GetDtmfTask(
                num_digits=1,
                ask_for_confirmation=False,
                chat_ctx=agent.chat_ctx.copy(exclude_instructions=True, exclude_function_call=True),
                extra_instructions=build_menu_prompt(prompt, normalized_options),
            )
        except ToolError as exc:
            await speak(agent, exc.message if hasattr(exc, "message") else str(exc))
            continue

        selection = result.user_input
        if selection in normalized_options:
            logger.debug("menu selection: %s -> %s", selection, normalized_options[selection])
            return selection

        await speak(agent, invalid_message)


def format_seconds(seconds: int) -> str:
    minutes, remaining = divmod(seconds, 60)
    parts: list[str] = []
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if remaining:
        parts.append(f"{remaining} second{'s' if remaining != 1 else ''}")
    if not parts:
        return "0 seconds"
    return " and ".join(parts)


def format_agents(agents: Iterable[CloudAgentRecord]) -> str:
    lines: list[str] = []
    for agent in agents:
        lines.append(
            f"{agent.name} is {agent.status} in {agent.region} with {agent.active_sessions} active sessions and {agent.uptime_hours} hours of uptime."
        )
    return " ".join(lines) if lines else "No agents are currently deployed for this project."


class RootIVRAgent(Agent):
    def __init__(self, *, dashboard: MockLiveKitDashboard, state: SessionState) -> None:
        super().__init__(
            instructions=(
                "You are LiveKit Cloud's automated customer service IVR. "
                "Your job is to authenticate the caller, present a layered phone tree, "
                "and collect digits for navigation. Be succinct, professional, and always remind callers "
                "how to return to the main menu or end the call."
            ),
        )
        self._dashboard = dashboard
        self._state = state

    async def on_enter(self) -> None:
        await speak(
            self,
            "Thank you for calling LiveKit Cloud Support. This automated system can review hosted agents, usage, telephony, and support status. Let's confirm your account to begin.",
        )
        await self._ensure_account_binding()
        await self._main_menu_loop()

    async def _ensure_account_binding(self) -> None:
        while not self._state.account_id:
            digits = await collect_digits(
                self,
                prompt=(
                    "Please enter your six digit LiveKit account number now."
                    " Say the digits clearly or use the keypad."
                ),
                num_digits=6,
            )
            candidate = f"ACCT-{digits}"
            if not self._dashboard.account_exists(candidate):
                await speak(
                    self,
                    "That account number was not found. Check the number on your LiveKit invoice and try again.",
                )
                continue

            self._state.account_id = candidate
            self._state.account_label = (
                self._dashboard.get_account_label(candidate) or "your organization"
            )
            self._state.audit_log.append(f"account_verified:{candidate}")
            await speak(
                self,
                f"Thanks. I located the account for {self._state.account_label}.",
            )

        while not self._state.project_id:
            digits = await collect_digits(
                self,
                prompt=(
                    "Enter the six digit project ID you would like to manage."
                    " If you are unsure, check the LiveKit Cloud dashboard."
                ),
                num_digits=6,
            )
            candidate = f"PRJ-{digits}"
            if not self._dashboard.project_exists(self._state.account_id, candidate):
                available = self._dashboard.list_projects(self._state.account_id)
                names = [
                    f"{project_id[-6:]} for {self._dashboard.describe_project(self._state.account_id, project_id)}"
                    for project_id in available
                ]
                await speak(
                    self,
                    "I did not find that project. Available project codes are "
                    + format_list(names)
                    + ". Let's try again.",
                )
                continue

            self._apply_project(candidate)
            await speak(
                self,
                f"Great. We'll work with the project {self._state.project_label}.",
            )

    async def _main_menu_loop(self) -> None:
        options = {
            "1": "Cloud-hosted agent operations",
            "2": "Usage and billing",
            "3": "Telephony and SIP",
            "4": "Performance metrics",
            "5": "Support services",
            "6": "Switch project",
            "7": "End this call",
        }

        while True:
            prompt = (
                f"Main menu for {self._state.project_label}. "
                "You can press nine during any submenu to return here."
            )
            choice = await run_menu(self, prompt=prompt, options=options)

            if choice == "1":
                outcome = await CloudAgentsTask(state=self._state, dashboard=self._dashboard)
                if await self._handle_task_outcome(outcome):
                    return
            elif choice == "2":
                outcome = await UsageBillingTask(state=self._state, dashboard=self._dashboard)
                if await self._handle_task_outcome(outcome):
                    return
            elif choice == "3":
                outcome = await TelephonyOpsTask(state=self._state, dashboard=self._dashboard)
                if await self._handle_task_outcome(outcome):
                    return
            elif choice == "4":
                outcome = await PerformanceMetricsTask(state=self._state, dashboard=self._dashboard)
                if await self._handle_task_outcome(outcome):
                    return
            elif choice == "5":
                outcome = await SupportServicesTask(state=self._state, dashboard=self._dashboard)
                if await self._handle_task_outcome(outcome):
                    return
            elif choice == "6":
                await self._switch_project()
            elif choice == "7":
                await self._farewell()
                return

    async def _handle_task_outcome(self, outcome: TaskOutcome) -> bool:
        if outcome == TaskOutcome.RETURN_TO_ROOT:
            return False
        if outcome == TaskOutcome.TRANSFER_SUPPORT:
            await speak(
                self,
                "I'll alert a LiveKit specialist and send them your session transcript. Please hold while we connect you.",
            )
        if outcome == TaskOutcome.END_SESSION:
            await self._farewell()
        return True

    async def _switch_project(self) -> None:
        assert self._state.account_id is not None, "account not set"

        projects = self._dashboard.list_projects(self._state.account_id)
        options: dict[str, str] = {}
        mapping: dict[str, str] = {}

        for idx, project_id in enumerate(projects, start=1):
            digit = str(idx)
            if idx > 8:
                break
            mapping[digit] = project_id
            label = self._dashboard.describe_project(self._state.account_id, project_id)
            options[digit] = f"Switch to {label}"
        options["9"] = "Cancel and stay on current project"

        prompt = "Select a project to manage. Each option references its primary label."

        selection = await run_menu(self, prompt=prompt, options=options)

        if selection == "9":
            await speak(self, "No changes made. Remaining on the current project.")
            return

        project_id = mapping[selection]
        self._apply_project(project_id)
        await speak(
            self,
            f"Switched to {self._state.project_label}. Returning to the main menu.",
        )

    def _apply_project(self, project_id: str) -> None:
        assert self._state.account_id is not None, "account not set"

        self._state.project_id = project_id
        self._state.project_label = self._dashboard.describe_project(
            self._state.account_id, project_id
        )
        self._state.usage_cache.pop(project_id, None)
        self._state.telephony_cache.pop(project_id, None)
        self._state.performance_cache.pop(project_id, None)
        self._state.audit_log.append(f"project_selected:{project_id}")

    async def _farewell(self) -> None:
        await speak(
            self,
            "Thank you for using LiveKit's automated support. A summary of this session will be available in your dashboard. Goodbye!",
            allow_interruptions=False,
        )


class BaseSubmenuTask(AgentTask[TaskOutcome]):
    def __init__(
        self,
        *,
        state: SessionState,
        dashboard: MockLiveKitDashboard,
        menu_name: str,
    ) -> None:
        super().__init__(
            instructions=(
                f"You operate the {menu_name} submenu for the LiveKit IVR. "
                "Speak concisely, present numbered options, and respect digit commands."
            )
        )
        self.state = state
        self.dashboard = dashboard
        self.menu_name = menu_name

    @property
    def account_id(self) -> str:
        if not self.state.account_id:
            raise RuntimeError("account not set")
        return self.state.account_id

    @property
    def project_id(self) -> str:
        if not self.state.project_id:
            raise RuntimeError("project not set")
        return self.state.project_id

    async def speak(self, message: str) -> None:
        await speak(self, message)


class CloudAgentsTask(BaseSubmenuTask):
    def __init__(self, *, state: SessionState, dashboard: MockLiveKitDashboard) -> None:
        super().__init__(state=state, dashboard=dashboard, menu_name="cloud agent operations")

    async def on_enter(self) -> None:
        await self.speak(
            f"Cloud agent menu for {self.state.project_label}. Press nine to return to the main menu or zero to end the call."
        )
        await self._loop()

    async def _loop(self) -> None:
        options = {
            "1": "Hear an overall agent summary",
            "2": "list each agent with status and uptime",
            "3": "Check active session totals",
            "4": "Inspect a specific agent",
            "5": "Review deployment regions",
            "9": "Return to the main menu",
            "0": "End the call",
        }

        while True:
            choice = await run_menu(
                self,
                prompt="Choose an agent operations option.",
                options=options,
            )

            if choice == "1":
                await self._agent_summary()
            elif choice == "2":
                await self._list_agents()
            elif choice == "3":
                await self._active_sessions()
            elif choice == "4":
                await self._agent_detail()
            elif choice == "5":
                await self._region_summary()
            elif choice == "9":
                await self.speak("Returning to the main menu.")
                self.complete(TaskOutcome.RETURN_TO_ROOT)
                return
            elif choice == "0":
                self.complete(TaskOutcome.END_SESSION)
                return

    async def _agent_summary(self) -> None:
        summary = self.dashboard.summarize_cloud_agents(self.account_id, self.project_id)
        running_names = self.dashboard.get_running_agent_names(self.account_id, self.project_id)
        message = (
            f"{self.state.project_label} has {summary['total_agents']} deployed cloud agents. "
            f"{summary['running']} are running, {summary['idle']} idle, and {summary['paused']} paused."
        )
        if running_names:
            message += f" Running agents include {format_list(running_names)}."

        sessions = self.dashboard.aggregate_agent_sessions(self.account_id, self.project_id)
        message += f" Combined active sessions total {sessions}."

        await self.speak(message)
        self.state.audit_log.append("cloud_agents:summary")

    async def _list_agents(self) -> None:
        agents = self.dashboard.list_cloud_agents(self.account_id, self.project_id)
        await self.speak(format_agents(agents))
        self.state.audit_log.append("cloud_agents:list")

    async def _active_sessions(self) -> None:
        sessions = self.dashboard.aggregate_agent_sessions(self.account_id, self.project_id)
        uptime = self.dashboard.aggregate_uptime_hours(self.account_id, self.project_id)
        await self.speak(
            f"Across all agents there are {sessions} active sessions with a cumulative uptime of {uptime} hours."
        )
        self.state.audit_log.append("cloud_agents:sessions")

    async def _agent_detail(self) -> None:
        agents = self.dashboard.list_cloud_agents(self.account_id, self.project_id)
        if not agents:
            await self.speak("There are no agents to inspect right now.")
            return

        menu: dict[str, str] = {}
        mapping: dict[str, CloudAgentRecord] = {}
        for idx, agent in enumerate(agents, start=1):
            digit = str(idx)
            if idx > 8:
                break
            menu[digit] = f"Details for {agent.name}"
            mapping[digit] = agent
        menu["9"] = "Go back"
        menu["0"] = "End the call"

        choice = await run_menu(
            self,
            prompt="Choose an agent to inspect.",
            options=menu,
        )

        if choice == "9":
            await self.speak("Back to the previous options.")
            return
        if choice == "0":
            self.complete(TaskOutcome.END_SESSION)
            return

        agent = mapping[choice]
        message = (
            f"Agent {agent.name} operates in {agent.region} as a {agent.modality} service. "
            f"Status is {agent.status} with {agent.active_sessions} live sessions and {agent.uptime_hours} uptime hours."
            " Alerts will trigger if uptime exceeds service level targets."
        )
        await self.speak(message)
        self.state.audit_log.append(f"cloud_agents:detail:{agent.name}")

    async def _region_summary(self) -> None:
        summary = self.dashboard.summarize_cloud_agents(self.account_id, self.project_id)
        regions = summary["regions"]
        message = (
            f"Deployments span {len(regions)} regions: {format_list(regions)}. "
            "Latency-sensitive workloads are automatically pinned to the caller's nearest region."
        )
        await self.speak(message)
        self.state.audit_log.append("cloud_agents:regions")


class UsageBillingTask(BaseSubmenuTask):
    def __init__(self, *, state: SessionState, dashboard: MockLiveKitDashboard) -> None:
        super().__init__(state=state, dashboard=dashboard, menu_name="usage and billing")

    async def on_enter(self) -> None:
        await self.speak(
            f"Usage and billing for {self.state.project_label}. Press nine to return to the main menu or zero to end."
        )
        await self._loop()

    async def _loop(self) -> None:
        options = {
            "1": "Hear current LLM, TTS, and STT spend",
            "2": "Check remaining balance",
            "3": "Projected burn rate and plan guidance",
            "4": "Email a usage summary",
            "9": "Return to the main menu",
            "0": "End the call",
        }

        while True:
            choice = await run_menu(
                self,
                prompt="Select a billing insight.",
                options=options,
            )

            if choice == "1":
                await self._spend_breakdown()
            elif choice == "2":
                await self._remaining_balance()
            elif choice == "3":
                await self._burn_rate()
            elif choice == "4":
                await self._email_summary()
            elif choice == "9":
                await self.speak("Returning to the main menu.")
                self.complete(TaskOutcome.RETURN_TO_ROOT)
                return
            elif choice == "0":
                self.complete(TaskOutcome.END_SESSION)
                return

    def _usage(self) -> dict[str, object]:
        if cached := self.state.usage_cache.get(self.project_id):
            return cached
        snapshot = self.dashboard.get_usage_snapshot(self.account_id, self.project_id)
        self.state.usage_cache[self.project_id] = snapshot
        return snapshot

    async def _spend_breakdown(self) -> None:
        usage = self._usage()
        message = (
            f"Since the start of billing cycle {usage['billing_cycle']}, "
            f"LLM usage totals {format_currency(usage['llm_cost'])}, "
            f"text-to-speech totals {format_currency(usage['tts_cost'])}, "
            f"and speech-to-text totals {format_currency(usage['stt_cost'])}."
        )
        await self.speak(message)
        self.state.audit_log.append("usage:spend")

    async def _remaining_balance(self) -> None:
        usage = self._usage()
        balance = format_currency(usage["balance_remaining"])
        await self.speak(
            f"Your remaining pre-paid balance is {balance}. You can top up anytime from the LiveKit dashboard or upgrade your plan."
        )
        self.state.audit_log.append("usage:balance")

    async def _burn_rate(self) -> None:
        usage = self._usage()
        burn = format_currency(usage["burn_rate_per_day"])
        llm = format_currency(usage["llm_cost"])
        tts = format_currency(usage["tts_cost"])
        stt = format_currency(usage["stt_cost"])
        advice = (
            "Consider the Scale plan"
            if usage["balance_remaining"] < 1000
            else "Current allotment looks healthy"
        )
        await self.speak(
            f"Daily burn rate is {burn}. Current cycle spend is {llm} on language, {tts} on voice output, and {stt} on transcription. {advice}."
        )
        self.state.audit_log.append("usage:burn")

    async def _email_summary(self) -> None:
        await self.speak(
            "I've queued a usage summary to be emailed to the billing contacts on file."
        )
        self.state.audit_log.append("usage:email")


class TelephonyOpsTask(BaseSubmenuTask):
    def __init__(self, *, state: SessionState, dashboard: MockLiveKitDashboard) -> None:
        super().__init__(state=state, dashboard=dashboard, menu_name="telephony operations")

    async def on_enter(self) -> None:
        await self.speak(
            f"Telephony overview for {self.state.project_label}. Press nine to return or zero to end the call."
        )
        await self._loop()

    async def _loop(self) -> None:
        options = {
            "1": "Inbound and outbound volumes",
            "2": "Queue depth and handle time",
            "3": "SIP trunk health",
            "4": "Voicemail and callbacks",
            "9": "Return to the main menu",
            "0": "End the call",
        }

        while True:
            choice = await run_menu(
                self,
                prompt="Choose a telephony report.",
                options=options,
            )

            if choice == "1":
                await self._volume_summary()
            elif choice == "2":
                await self._queue_status()
            elif choice == "3":
                await self._sip_trunks()
            elif choice == "4":
                await self._voicemail()
            elif choice == "9":
                await self.speak("Returning to the main menu.")
                self.complete(TaskOutcome.RETURN_TO_ROOT)
                return
            elif choice == "0":
                self.complete(TaskOutcome.END_SESSION)
                return

    def _telephony(self) -> dict[str, object]:
        if cached := self.state.telephony_cache.get(self.project_id):
            return cached
        stats = self.dashboard.get_telephony_stats(self.account_id, self.project_id)
        self.state.telephony_cache[self.project_id] = stats
        return stats

    async def _volume_summary(self) -> None:
        stats = self._telephony()
        inbound = stats["inbound_calls"]
        outbound = stats["outbound_calls"]
        await self.speak(
            f"Inbound calls this cycle total {inbound}, outbound calls {outbound}. LiveKit auto-scales telephony capacity based on daily peaks."
        )
        self.state.audit_log.append("telephony:volume")

    async def _queue_status(self) -> None:
        stats = self._telephony()
        queued = stats["queued_calls"]
        handle_time = format_seconds(stats["avg_handle_time_seconds"])
        await self.speak(
            f"There are currently {queued} callers waiting. Average handle time sits at {handle_time}."
        )
        self.state.audit_log.append("telephony:queue")

    async def _sip_trunks(self) -> None:
        stats = self._telephony()
        trunks = stats["sip_trunks"]
        await self.speak(
            f"SIP trunk health shows {trunks['healthy']} healthy, {trunks['degraded']} degraded, and {trunks['offline']} offline connections."
        )
        self.state.audit_log.append("telephony:trunks")

    async def _voicemail(self) -> None:
        stats = self._telephony()
        voicemails = stats["voicemail_count"]
        await self.speak(
            f"You have {voicemails} new voicemail messages. Callback automation can be toggled in the dashboard if you need faster follow-ups."
        )
        self.state.audit_log.append("telephony:voicemail")


class PerformanceMetricsTask(BaseSubmenuTask):
    def __init__(self, *, state: SessionState, dashboard: MockLiveKitDashboard) -> None:
        super().__init__(state=state, dashboard=dashboard, menu_name="performance metrics")

    async def on_enter(self) -> None:
        await self.speak(
            f"Performance metrics for {self.state.project_label}. Press nine to return or zero to end."
        )
        await self._loop()

    async def _loop(self) -> None:
        options = {
            "1": "Language model latency",
            "2": "Text-to-speech latency",
            "3": "Speech-to-text latency",
            "4": "Recent incidents",
            "5": "Comprehensive performance summary",
            "9": "Return to the main menu",
            "0": "End the call",
        }

        while True:
            choice = await run_menu(
                self,
                prompt="Select a performance insight.",
                options=options,
            )

            if choice in {"1", "2", "3"}:
                await self._latency_report(choice)
            elif choice == "4":
                await self._incident_history()
            elif choice == "5":
                await self._comprehensive_summary()
            elif choice == "9":
                await self.speak("Returning to the main menu.")
                self.complete(TaskOutcome.RETURN_TO_ROOT)
                return
            elif choice == "0":
                self.complete(TaskOutcome.END_SESSION)
                return

    def _performance(self) -> dict[str, object]:
        if cached := self.state.performance_cache.get(self.project_id):
            return cached
        metrics = self.dashboard.get_performance_metrics(self.account_id, self.project_id)
        self.state.performance_cache[self.project_id] = metrics
        return metrics

    async def _latency_report(self, option: str) -> None:
        performance = self._performance()
        modality_map = {
            "1": ("language model", performance["llm"]),
            "2": ("text-to-speech", performance["tts"]),
            "3": ("speech-to-text", performance["stt"]),
        }
        label, metrics_obj = modality_map[option]
        metrics_obj = metrics_obj  # type: ignore[assignment]
        if not isinstance(metrics_obj, LatencyMetrics):
            raise RuntimeError("unexpected latency metrics structure")
        await self.speak(
            f"Current {label} latency shows {MockLiveKitDashboard.format_latency(metrics_obj)}."
        )
        self.state.audit_log.append(f"performance:{label}")

    async def _incident_history(self) -> None:
        performance = self._performance()
        incidents = performance["last_incidents"]
        if incidents:
            await self.speak("Recent incidents include " + format_list(incidents) + ".")
        else:
            await self.speak("No incidents have been reported in the last 30 days.")
        self.state.audit_log.append("performance:incidents")

    async def _comprehensive_summary(self) -> None:
        performance = self._performance()
        llm = performance["llm"]
        tts = performance["tts"]
        stt = performance["stt"]
        await self.speak(
            "Overall system health is steady. "
            f"LLM latency: {MockLiveKitDashboard.format_latency(llm)}. "
            f"TTS latency: {MockLiveKitDashboard.format_latency(tts)}. "
            f"STT latency: {MockLiveKitDashboard.format_latency(stt)}."
        )
        self.state.audit_log.append("performance:summary")


class SupportServicesTask(BaseSubmenuTask):
    def __init__(self, *, state: SessionState, dashboard: MockLiveKitDashboard) -> None:
        super().__init__(state=state, dashboard=dashboard, menu_name="support services")

    async def on_enter(self) -> None:
        await self.speak(
            f"Support services for {self.state.project_label}. Press nine to return or zero to end."
        )
        await self._loop()

    async def _loop(self) -> None:
        options = {
            "1": "Open ticket status",
            "2": "Request a callback",
            "3": "Escalate to a LiveKit specialist",
            "4": "Get documentation recommendations",
            "9": "Return to the main menu",
            "0": "End the call",
        }

        while True:
            choice = await run_menu(
                self,
                prompt="Select a support option.",
                options=options,
            )

            if choice == "1":
                await self._ticket_status()
            elif choice == "2":
                await self._request_callback()
            elif choice == "3":
                self.complete(TaskOutcome.TRANSFER_SUPPORT)
                return
            elif choice == "4":
                await self._documentation_links()
            elif choice == "9":
                await self.speak("Returning to the main menu.")
                self.complete(TaskOutcome.RETURN_TO_ROOT)
                return
            elif choice == "0":
                self.complete(TaskOutcome.END_SESSION)
                return

    async def _ticket_status(self) -> None:
        support = self.dashboard.get_support_overview(self.account_id, self.project_id)
        tickets = support["open_tickets"]
        await self.speak(
            f"You have {tickets} open tickets. SLA tier is {support['sla_tier']}. Last contact was {support['last_agent_contact']}."
        )
        self.state.audit_log.append("support:tickets")

    async def _request_callback(self) -> None:
        support = self.dashboard.get_support_overview(self.account_id, self.project_id)
        await self.speak(
            "A callback request has been logged. A LiveKit engineer will reach out using the preferred contact on file."
        )
        support["pending_callbacks"] = support.get("pending_callbacks", 0) + 1
        self.state.audit_log.append("support:callback")

    async def _documentation_links(self) -> None:
        await self.speak(
            "For deeper guidance, review the LiveKit Cloud agent deployment playbook and the SIP integration cookbook available in your dashboard's documentation tab."
        )
        self.state.audit_log.append("support:docs")


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    dashboard = MockLiveKitDashboard()
    state = SessionState()

    session: AgentSession[SessionState] = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-4.1-mini"),
        stt=deepgram.STT(model="nova-3"),
        tts=elevenlabs.TTS(model="eleven_multilingual_v2"),
        turn_detection=MultilingualModel(),
        userdata=state,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage() -> None:
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=RootIVRAgent(dashboard=dashboard, state=state),
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name=AGENTIC_IVR_DISPATCH_NAME,
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
