from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from mock_bank_service import (
    CreditCard,
    DepositAccount,
    LoanAccount,
    MockBankService,
    RewardsSummary,
    format_currency,
    format_transactions,
)

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    cli,
    inference,
    metrics,
)
from livekit.agents.beta.workflows.dtmf_inputs import GetDtmfTask
from livekit.agents.llm.tool_context import ToolError
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()


logger = logging.getLogger("bank-ivr")


BANK_IVR_DISPATCH_NAME = os.getenv("BANK_IVR_DISPATCH_NAME", "bank-ivr-agent")


server = AgentServer()


class TaskOutcome(str, Enum):
    RETURN_TO_ROOT = "return_to_root"
    END_SESSION = "end_session"


@dataclass
class SessionState:
    customer_id: Optional[str] = None  # noqa: UP007
    customer_name: Optional[str] = None  # noqa: UP007
    branch_name: Optional[str] = None  # noqa: UP007
    deposit_cache: dict[str, tuple[DepositAccount, ...]] = field(default_factory=dict)
    card_cache: dict[str, tuple[CreditCard, ...]] = field(default_factory=dict)
    loan_cache: dict[str, tuple[LoanAccount, ...]] = field(default_factory=dict)
    rewards_cache: dict[str, RewardsSummary] = field(default_factory=dict)
    audit_log: list[str] = field(default_factory=list)


def speak(agent: Agent, instructions: str) -> None:
    agent.session.say(text=instructions, allow_interruptions=False)


async def collect_digits(
    agent: Agent,
    *,
    prompt: str,
    num_digits: int,
    confirmation: bool = False,
) -> str:
    while True:
        try:
            result = await GetDtmfTask(
                num_digits=num_digits,
                ask_for_confirmation=confirmation,
                chat_ctx=agent.chat_ctx.copy(
                    exclude_instructions=True,
                    exclude_function_call=True,
                    exclude_handoff=True,
                    exclude_config_update=True,
                ),
                extra_instructions=(
                    "You are gathering keypad digits from a bank customer. "
                    f"Prompt them with: {prompt}."
                ),
            )
        except ToolError as exc:
            speak(agent, exc.message if hasattr(exc, "message") else str(exc))
            continue

        return result.user_input.replace(" ", "")


async def add_event_message(agent: Agent, *, content: str) -> None:
    agent.chat_ctx.copy().add_message(
        role="user", content=f"<system_event>{content}</system_event>"
    )
    await agent.update_chat_ctx(agent.chat_ctx)


async def run_menu(
    agent: Agent,
    *,
    prompt: str,
    options: dict[str, str],
    invalid_message: str = "I did not catch that selection. Let's try again.",
) -> str:
    normalized_options = dict(options.items())

    while True:
        instructions_text = f"{prompt} " + " ".join(
            f"Press {digit} for {label}." for digit, label in normalized_options.items()
        )

        try:
            result = await GetDtmfTask(
                num_digits=1,
                ask_for_confirmation=False,
                extra_instructions=instructions_text,
            )
        except ToolError as exc:
            speak(agent, exc.message if hasattr(exc, "message") else str(exc))
            continue

        if result.user_input in normalized_options:
            choice = result.user_input
            logger.debug("menu selection: %s -> %s", choice, normalized_options[choice])
            return choice

        await add_event_message(
            agent, content=f"User entered invalid menu selection: {result.user_input}"
        )
        speak(agent, invalid_message)


class RootBankIVRAgent(Agent):
    def __init__(self, *, service: MockBankService, state: SessionState) -> None:
        super().__init__(
            instructions=(
                "You are the automated telephone assistant for Horizon Federal Bank. "
                "Authenticate callers, guide them through the banking menu, and remind them they can press 9 to return to the main menu."
            ),
        )
        self._service = service
        self._state = state

    async def on_enter(self) -> None:
        await self._authenticate_customer()
        await self._main_menu_loop()

    async def _authenticate_customer(self) -> None:
        while True:
            customer_id = await collect_digits(
                self,
                prompt="Please enter your eight digit customer ID",
                num_digits=8,
                confirmation=False,
            )
            # customer_id = "10000001"
            await add_event_message(self, content=f"User entered customer ID: {customer_id}")
            pin = await collect_digits(
                self,
                prompt="Now enter your four digit telephone banking PIN",
                num_digits=4,
                confirmation=False,
            )
            # pin = "0000"
            await add_event_message(self, content=f"User entered PIN: {pin}")

            if self._service.authenticate(customer_id, pin):
                profile = self._service.get_profile(customer_id)
                self._state.customer_id = customer_id
                self._state.customer_name = profile.full_name
                self._state.branch_name = profile.branch_name
                self._state.deposit_cache[customer_id] = profile.deposit_accounts
                self._state.card_cache[customer_id] = profile.credit_cards
                self._state.loan_cache[customer_id] = profile.loans
                self._state.rewards_cache[customer_id] = profile.rewards
                self._state.audit_log.append(f"auth_success:{customer_id}")
                await add_event_message(
                    self, content=f"Authentication successful for {profile.full_name}"
                )
                speak(
                    self,
                    f"Thank you {profile.full_name}. You're connected with the {profile.branch_name} branch menu.",
                )
                return

            await add_event_message(self, content="Authentication failed")
            speak(
                self,
                "I'm sorry, that ID and PIN combination was not recognized. Let's try again.",
            )
            self._state.audit_log.append("auth_failed")

    async def _main_menu_loop(self) -> None:
        options = {
            "1": "deposit accounts",
            "2": "credit cards",
            "3": "loans and mortgages",
            "4": "rewards and benefits",
            "5": "switch profile",
        }

        choice_to_task: dict[str, type[SubmenuTaskType]] = {
            "1": DepositAccountsTask,
            "2": CreditCardsTask,
            "3": LoansTask,
            "4": RewardsTask,
        }

        while True:
            prompt = (
                f"Main menu for {self._state.customer_name}. "
                "Press 1 for deposit accounts, 2 for credit cards, 3 for loans, 4 for rewards, or 5 to switch profile. Always say out loud the menu options to user first before asking for selection."
            )
            choice = await run_menu(self, prompt=prompt, options=options)

            if choice in choice_to_task:
                task = choice_to_task[choice](state=self._state, service=self._service)
                outcome = await task
                if await self._handle_task_outcome(outcome):
                    return
                continue

            if choice == "5":
                await self._switch_profile()
                continue

    async def _handle_task_outcome(self, outcome: TaskOutcome) -> bool:
        if outcome == TaskOutcome.RETURN_TO_ROOT:
            return False
        if outcome == TaskOutcome.END_SESSION:
            await self._farewell()
        return True

    async def _switch_profile(self) -> None:
        customer_ids = self._service.list_customer_ids()
        options: dict[str, str] = {}
        mapping: dict[str, str] = {}

        for index, customer_id in enumerate(customer_ids, start=1):
            digit = str(index)
            profile = self._service.get_profile(customer_id)
            options[digit] = f"Switch to {profile.full_name}"
            mapping[digit] = customer_id

        options["9"] = "Cancel"

        selection = await run_menu(
            self,
            prompt="Choose which customer profile you'd like to access.",
            options=options,
        )

        if selection == "9":
            speak(self, "Staying with the current profile.")
            return

        chosen_id = mapping[selection]
        original_id = self._state.customer_id
        original_name = self._state.customer_name
        original_branch = self._state.branch_name

        speak(self, "Okay, let's verify that profile now.")

        while True:
            pin = await collect_digits(
                self,
                prompt=f"Please enter the four digit PIN for customer ID {chosen_id}",
                num_digits=4,
                confirmation=False,
            )
            if self._service.authenticate(chosen_id, pin):
                profile = self._service.get_profile(chosen_id)
                self._state.customer_id = chosen_id
                self._state.customer_name = profile.full_name
                self._state.branch_name = profile.branch_name
                self._state.deposit_cache[chosen_id] = profile.deposit_accounts
                self._state.card_cache[chosen_id] = profile.credit_cards
                self._state.loan_cache[chosen_id] = profile.loans
                self._state.rewards_cache[chosen_id] = profile.rewards
                speak(self, f"Verified. You're now managing accounts for {profile.full_name}.")
                return

            speak(self, "That PIN did not match. Let's try again or press 9 to cancel.")
            retry = await run_menu(
                self,
                prompt="Press 1 to try again or 9 to cancel switching.",
                options={"1": "Retry", "9": "Cancel"},
            )
            if retry == "9":
                speak(self, "Returning to the previous customer profile.")
                self._state.customer_id = original_id
                self._state.customer_name = original_name
                self._state.branch_name = original_branch
                return

    async def _farewell(self) -> None:
        speak(
            self,
            "Thanks for banking with Horizon Federal Bank. Goodbye!",
        )


class BaseBankTask(AgentTask[TaskOutcome]):
    def __init__(
        self,
        *,
        state: SessionState,
        service: MockBankService,
        menu_name: str,
    ) -> None:
        super().__init__(
            instructions=(
                f"You are handling the {menu_name} submenu for Horizon Federal Bank. "
                "Speak professionally, cite balances precisely, and remind callers that pressing 9 returns to the main menu."
            )
        )
        self.state = state
        self.service = service
        self.menu_name = menu_name

    @property
    def customer_id(self) -> str:
        if not self.state.customer_id:
            raise RuntimeError("Customer ID not set")
        return self.state.customer_id

    @property
    def customer_name(self) -> str:
        return self.state.customer_name or "the customer"

    def speak(self, message: str) -> None:
        speak(self, message)


class DepositAccountsTask(BaseBankTask):
    def __init__(self, *, state: SessionState, service: MockBankService) -> None:
        super().__init__(state=state, service=service, menu_name="deposit accounts")

    async def on_enter(self) -> None:
        await self._loop()

    async def _loop(self) -> None:
        options = {
            "1": "Hear balances for each account",
            "2": "Review available cash",
            "3": "Listen to recent transactions",
            "4": "Total deposits across accounts",
            "9": "Return to the main menu",
        }

        while True:
            choice = await run_menu(
                self,
                prompt="Deposit accounts menu",
                options=options,
            )

            if choice == "1":
                await self._account_balances()
            elif choice == "2":
                await self._available_cash()
            elif choice == "3":
                await self._recent_transactions()
            elif choice == "4":
                await self._total_deposits()
            elif choice == "9":
                self.speak("Returning to the main menu now.")
                self.complete(TaskOutcome.RETURN_TO_ROOT)
                return

    def _accounts(self) -> tuple[DepositAccount, ...]:
        cached = self.state.deposit_cache.get(self.customer_id)
        if cached is not None:
            return cached
        accounts = self.service.list_deposit_accounts(self.customer_id)
        self.state.deposit_cache[self.customer_id] = accounts
        return accounts

    async def _account_balances(self) -> None:
        lines: list[str] = []
        for acct in self._accounts():
            lines.append(
                f"{acct.account_type} ending in {acct.account_number[-4:]} has a balance of {format_currency(acct.balance)}."
            )
        self.speak(" ".join(lines))
        self.state.audit_log.append("deposit:balances")

    async def _available_cash(self) -> None:
        lines: list[str] = []
        for acct in self._accounts():
            delta = acct.available_balance
            lines.append(
                f"Available funds for {acct.account_type} ending in {acct.account_number[-4:]} are {format_currency(delta)}."
            )
        self.speak(" ".join(lines))
        self.state.audit_log.append("deposit:available")

    async def _recent_transactions(self) -> None:
        accounts = self._accounts()
        menu: dict[str, str] = {
            str(i + 1): f"{acct.account_type} ending in {acct.account_number[-4:]}"
            for i, acct in enumerate(accounts)
        }
        menu["9"] = "Go back"

        selection = await run_menu(
            self,
            prompt="Choose an account to hear recent activity.",
            options=menu,
        )

        if selection == "9":
            return

        index = int(selection) - 1
        account = accounts[index]
        activity = format_transactions(account.recent_transactions)
        self.speak(
            f"Recent activity for {account.account_type} ending in {account.account_number[-4:]}:\n{activity}"
        )
        self.state.audit_log.append(f"deposit:transactions:{account.account_number}")

    async def _total_deposits(self) -> None:
        total = self.service.calculate_total_deposits(self.customer_id)
        self.speak(f"Total deposits across your accounts come to {format_currency(total)}.")
        self.state.audit_log.append("deposit:total")


class CreditCardsTask(BaseBankTask):
    def __init__(self, *, state: SessionState, service: MockBankService) -> None:
        super().__init__(state=state, service=service, menu_name="credit cards")

    async def on_enter(self) -> None:
        await self._loop()

    async def _loop(self) -> None:
        options = {
            "1": "Statement and payment details",
            "2": "Rewards earning rates",
            "3": "Total card balances",
            "9": "Return to main menu",
        }

        while True:
            choice = await run_menu(self, prompt="Credit card menu", options=options)
            if choice == "1":
                await self._statement_details()
            elif choice == "2":
                await self._rewards_rates()
            elif choice == "3":
                await self._total_balances()
            elif choice == "9":
                self.speak("Returning you to the main menu now.")
                self.complete(TaskOutcome.RETURN_TO_ROOT)
                return

    def _cards(self) -> tuple[CreditCard, ...]:
        cached = self.state.card_cache.get(self.customer_id)
        if cached is not None:
            return cached
        cards = self.service.list_credit_cards(self.customer_id)
        self.state.card_cache[self.customer_id] = cards
        return cards

    async def _statement_details(self) -> None:
        lines: list[str] = []
        for card in self._cards():
            lines.append(
                " ".join(
                    [
                        f"Card ending in {card.card_number[-4:]} has a statement balance of {format_currency(card.statement_balance)}.",
                        f"Minimum due is {format_currency(card.minimum_due)} on {card.payment_due_date}.",
                    ]
                )
            )
        self.speak(" ".join(lines))
        self.state.audit_log.append("cards:statement")

    async def _rewards_rates(self) -> None:
        lines = [f"{card.product_name} earns {card.rewards_earn_rate}." for card in self._cards()]
        self.speak(" ".join(lines))
        self.state.audit_log.append("cards:rewards")

    async def _total_balances(self) -> None:
        balance = self.service.calculate_total_card_balance(self.customer_id)
        self.speak(f"Total statement balances across your cards are {format_currency(balance)}.")
        self.state.audit_log.append("cards:total")


class LoansTask(BaseBankTask):
    def __init__(self, *, state: SessionState, service: MockBankService) -> None:
        super().__init__(state=state, service=service, menu_name="loans and mortgages")

    async def on_enter(self) -> None:
        await self._loop()

    async def _loop(self) -> None:
        options = {
            "1": "Outstanding balances",
            "2": "Upcoming payments",
            "3": "Autopay status",
            "9": "Return to main menu",
        }

        while True:
            choice = await run_menu(self, prompt="Loans menu", options=options)
            if choice == "1":
                await self._balances()
            elif choice == "2":
                await self._upcoming_payments()
            elif choice == "3":
                await self._autopay_status()
            elif choice == "9":
                self.speak("Returning to the main menu now.")
                self.complete(TaskOutcome.RETURN_TO_ROOT)
                return

    def _loans(self) -> tuple[LoanAccount, ...]:
        cached = self.state.loan_cache.get(self.customer_id)
        if cached is not None:
            return cached
        loans = self.service.list_loans(self.customer_id)
        self.state.loan_cache[self.customer_id] = loans
        return loans

    async def _balances(self) -> None:
        lines = [
            f"{loan.loan_type} ending in {loan.loan_id[-4:]} has {format_currency(loan.outstanding_balance)} remaining."  # noqa: E501
            for loan in self._loans()
        ]
        self.speak(" ".join(lines))
        self.state.audit_log.append("loans:balance")

    async def _upcoming_payments(self) -> None:
        payments = self.service.upcoming_payments(self.customer_id)
        lines = [
            f"Loan {loan_id[-4:]} payment of {format_currency(amount)} is due on {due}."
            for loan_id, amount, due in payments
        ]
        self.speak(" ".join(lines))
        self.state.audit_log.append("loans:payments")

    async def _autopay_status(self) -> None:
        lines = [
            (
                f"Autopay is {'enabled' if loan.autopay_enabled else 'not enabled'} for {loan.loan_type} ending in {loan.loan_id[-4:]}."
            )
            for loan in self._loans()
        ]
        self.speak(" ".join(lines))
        self.state.audit_log.append("loans:autopay")


class RewardsTask(BaseBankTask):
    def __init__(self, *, state: SessionState, service: MockBankService) -> None:
        super().__init__(state=state, service=service, menu_name="rewards and benefits")

    async def on_enter(self) -> None:
        await self._loop()

    async def _loop(self) -> None:
        options = {
            "1": "Rewards balance and tier",
            "2": "Cashback available",
            "3": "Expiring points",
            "9": "Return to main menu",
        }

        while True:
            choice = await run_menu(self, prompt="Rewards menu", options=options)
            if choice == "1":
                await self._balance_and_tier()
            elif choice == "2":
                await self._cashback()
            elif choice == "3":
                await self._expiring_points()
            elif choice == "9":
                self.speak("Returning to the main menu now.")
                self.complete(TaskOutcome.RETURN_TO_ROOT)
                return

    def _rewards(self) -> RewardsSummary:
        cached = self.state.rewards_cache.get(self.customer_id)
        if cached is not None:
            return cached
        rewards = self.service.get_rewards(self.customer_id)
        self.state.rewards_cache[self.customer_id] = rewards
        return rewards

    async def _balance_and_tier(self) -> None:
        rewards = self._rewards()
        self.speak(
            f"You are in the {rewards.tier} tier with {rewards.points_balance:,} points available."
        )
        self.state.audit_log.append("rewards:balance")

    async def _cashback(self) -> None:
        rewards = self._rewards()
        self.speak(
            f"You have {format_currency(rewards.cashback_available)} in cashback ready to redeem."
        )
        self.state.audit_log.append("rewards:cashback")

    async def _expiring_points(self) -> None:
        rewards = self._rewards()
        points = rewards.expiring_next_statement
        if points:
            self.speak(
                f"{points:,} points will expire on your next statement. Consider redeeming soon."
            )
        else:
            self.speak("Great news—no points are set to expire on your next statement.")
        self.state.audit_log.append("rewards:expiring")


SubmenuTaskType = DepositAccountsTask | CreditCardsTask | LoansTask | RewardsTask


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=BANK_IVR_DISPATCH_NAME)
async def bank_ivr_session(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    service = MockBankService()
    state = SessionState()

    session: AgentSession[SessionState] = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=inference.LLM("openai/gpt-4.1"),
        stt=inference.STT("deepgram/nova-3"),
        tts=inference.TTS("cartesia/sonic-3"),
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
        agent=RootBankIVRAgent(service=service, state=state),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
