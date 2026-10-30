# SPDX-License-Identifier: Apache-2.0
"""Acknowledged Prism history changes shared by startup, updates, and restart."""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from rimelabs_sdk import realtime as r

from livekit.agents import llm


class PrismHistoryUpdateError(llm.RealtimeError):
    """An edit failure with a detached, confirmed projection and no transcript in its message."""

    def __init__(
        self,
        message: str,
        *,
        status: Literal["rejected", "partial", "uncertain"] = "rejected",
        confirmed_chat_ctx: llm.ChatContext | None = None,
        needs_reconciliation: bool = True,
    ):
        super().__init__(message, code=f"history_update_{status}")
        self.status = status
        self.recoverable = status != "uncertain"
        self.confirmed_chat_ctx = confirmed_chat_ctx
        self.needs_reconciliation = needs_reconciliation

    @property
    def partial(self) -> bool:
        return self.status == "partial"


HistoryUpdateError = PrismHistoryUpdateError


@dataclass(frozen=True)
class HistoryUpdateNotice:
    session_id: str
    status: Literal["rejected", "partial", "uncertain"]
    recoverable: bool
    confirmed_chat_ctx: llm.ChatContext | None = field(repr=False)
    needs_reconciliation: bool = True


@dataclass
class HistorySyncResult:
    submitted: list[tuple[r.ToolCallRef, llm.FunctionCallOutput]] = field(default_factory=list)
    failure: PrismHistoryUpdateError | None = None
    committed: bool = False
    created: set[str] = field(default_factory=set)
    history_read: bool = False


def _items(ctx: llm.ChatContext) -> list[llm.ChatItem]:
    projected = ctx.copy(exclude_handoff=True, exclude_config_update=True)
    return projected.items


def _wire(item: llm.ChatItem, *, validate_text: bool = True) -> r.HistoryItem:
    if isinstance(item, llm.ChatMessage):
        if item.role not in ("user", "assistant", "system") or any(
            not isinstance(c, str) for c in item.content
        ):
            raise HistoryUpdateError("Prism history requires user, assistant, or system text")
        text = item.text_content or ""
        if validate_text and len(text) > 4000:
            raise HistoryUpdateError("Prism history text must contain at most 4000 characters")
        result: r.HistoryItem = r.HistoryMessage(role=item.role, text=text)
    elif isinstance(item, llm.FunctionCall):
        if not item.call_id or not item.name:
            raise HistoryUpdateError("Past tool calls require a call ID and name")
        result = r.HistoryFunctionCall(
            call_id=item.call_id, name=item.name, arguments=item.arguments
        )
    elif isinstance(item, llm.FunctionCallOutput):
        if not item.call_id:
            raise HistoryUpdateError("Tool results require a call ID")
        output = item.output
        if item.is_error:
            output = json.dumps(
                {
                    "status": "failed",
                    "reason": "tool_error",
                    "retryable": False,
                    "message": output,
                    "result": None,
                }
            )
        result = r.HistoryFunctionResult(call_id=item.call_id, output=output)
    else:
        raise HistoryUpdateError("Unsupported Prism history item")
    # Validate encoding before the first operation of a batch reaches the SDK.
    for value in vars(result).values():
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise HistoryUpdateError("Prism history must be valid UTF-8") from error
    return result


class PrismHistory:
    """Keep local identity stable while applying changes through the actual SDK.

    Own the chat record, tool references, server IDs, and history update lock.
    Event observations remain synchronous so acknowledgments can keep arriving
    while an edit holds the lock. Callers receive independent chat snapshots.

    Reads are snapshots, not replacements for the local conversation record.
    Deletion-only changes and missing imported results preserve existing turns.
    Changes to order or content rebuild completed history, avoiding guesses about
    whole-turn insertion anchors.
    This is a sequence of acknowledged operations, not a server transaction.
    """

    def __init__(self, sdk: r.RealtimeSession):
        self._sdk = sdk
        self._chat = llm.ChatContext.empty()
        self._calls: OrderedDict[str, tuple[r.ToolCallRef, llm.FunctionCall]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._failure: HistoryUpdateError | None = None
        self._ids: dict[str, str] = {}
        self._active: dict[str, set[str]] = {}
        self._response_revision = 0
        self._revision = 0
        self._variants: dict[str, set[str]] = {}
        self._played: dict[str, str] = {}
        self._holds: dict[str, Any] = {}

    @property
    def chat_ctx(self) -> llm.ChatContext:
        """Return a snapshot whose items cannot change the acknowledged record."""
        return llm.ChatContext([item.model_copy(deep=True) for item in self._chat.items])

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def needs_reconciliation(self) -> bool:
        return self._failure is not None

    def projection(self, local: llm.ChatContext) -> llm.ChatContext:
        result = self.chat_ctx
        for item in result.items:
            if isinstance(item, llm.ChatMessage) and item.id in self._played:
                item.content = [self._played[item.id]]
        result.items.extend(
            item.model_copy(deep=True)
            for item in local.items
            if not isinstance(item, (llm.ChatMessage, llm.FunctionCall, llm.FunctionCallOutput))
        )
        return result

    async def reconcile(self, target: llm.ChatContext, revision: int) -> None:
        async with self._lock:
            if revision != self._revision or _items(target) != _items(self.projection(target)):
                self._failure = HistoryUpdateError(
                    "History changed during reconciliation",
                    confirmed_chat_ctx=self.projection(llm.ChatContext.empty()),
                )
                raise self._failure
            self._failure = None

    def release_response(self, response_id: str) -> None:
        hold = self._holds.pop(response_id, None)
        if hold is not None:
            hold.release()

    def close(self) -> None:
        for response_id in list(self._holds):
            self.release_response(response_id)

    async def get_history(self) -> tuple[r.HistoryItem, ...]:
        """Read completed server history after any pending history update."""
        async with self._lock:
            return await self._sdk.get_history()

    async def wait_for_updates(self) -> None:
        """Wait for pending edits without holding a lock during reply admission."""
        async with self._lock:
            if self._failure is not None:
                raise self._failure

    def last_call(self, response: r.ResponseRef) -> r.ToolCallRef | None:
        """Find the latest retained call for a tool-continuation fallback."""
        return next(
            (
                ref
                for ref, _ in reversed(self._calls.values())
                if ref.response_id == response.response_id
            ),
            None,
        )

    def has_pending_calls(self, response: r.ResponseRef) -> bool:
        """Whether a response still has tool calls without acknowledged results."""
        completed = {i.call_id for i in self._chat.items if isinstance(i, llm.FunctionCallOutput)}
        return any(
            call_id not in completed
            for call_id, (ref, _) in self._calls.items()
            if ref.response_id == response.response_id
        )

    def record_tool_call(self, payload: r.ToolCall) -> llm.FunctionCall:
        """Record an allowed live call and return a separate item for execution."""
        call = llm.FunctionCall(
            id=payload.item_id,
            call_id=payload.call.call_id,
            name=payload.name,
            arguments=json.dumps(payload.arguments),
        )
        if payload.call.call_id in self._calls:
            return self._calls[payload.call.call_id][1].model_copy(deep=True)
        hold = getattr(self._sdk, "hold_response", None)
        if hold is not None and payload.call.response_id not in self._holds:
            self._holds[payload.call.response_id] = hold(payload.call)
        self._calls[payload.call.call_id] = (payload.call, call)
        if len(self._calls) > 1024:
            completed = {
                i.call_id for i in self._chat.items if isinstance(i, llm.FunctionCallOutput)
            }
            for key in list(self._calls):
                if len(self._calls) <= 1024:
                    break
                if key in completed and self._calls[key][0].response_id not in self._holds:
                    del self._calls[key]
        self._chat.items.append(call)
        self._revision += 1
        return call.model_copy(deep=True)

    def record_assistant(self, message_id: str, transcript: str, *, played: bool = False) -> None:
        """Record completed or interrupted output, including an unheard message."""
        self._revision += 1
        self._variants.setdefault(message_id, set()).add(transcript)
        if played:
            self._played[message_id] = transcript
            return
        self._chat._upsert_item(
            llm.ChatMessage(id=message_id, role="assistant", content=[transcript])
        )

    @staticmethod
    def validate(
        ctx: llm.ChatContext, *, existing: set[str] | None = None, unchanged: set[str] | None = None
    ) -> None:
        items = _items(ctx)
        ids: set[str] = set()
        calls: dict[str, str] = {}
        results: set[str] = set()
        existing = existing or set()
        for item in items:
            if item.id in ids:
                raise HistoryUpdateError("Prism history item IDs must be unique")
            ids.add(item.id)
            _wire(item, validate_text=item.id not in (unchanged or set()))
            if isinstance(item, llm.FunctionCall):
                if item.call_id in calls:
                    raise HistoryUpdateError("Prism history call IDs must be unique")
                calls[item.call_id] = item.id
            if isinstance(item, llm.FunctionCallOutput):
                if item.call_id not in calls or item.call_id in results:
                    raise HistoryUpdateError("A tool result must follow its call exactly once")
                results.add(item.call_id)
        if any(
            item_id not in existing and call_id not in results for call_id, item_id in calls.items()
        ):
            raise HistoryUpdateError("Imported tool calls require completed results")

    def observe(
        self,
        payload: r.InputEvent | r.ResponseEvent | r.ItemTruncated | r.FaultEvent,
        *,
        turn_started_at: float | None = None,
    ) -> None:
        # This method never awaits: event consumption must not wait on a history lock.
        if isinstance(payload, r.ResponseStarted):
            self._response_revision += 1
            ids = set(payload.input_item_ids)
            if payload.input_item_id:
                ids.add(payload.input_item_id)
            self._active[payload.response.response_id] = ids
        elif isinstance(payload, r.InputCommitted):
            self._ids.setdefault(payload.item_id, payload.item_id)
            if not self._chat.get_by_id(payload.item_id):
                self._revision += 1
                self._chat.items.append(
                    llm.ChatMessage(
                        id=payload.item_id,
                        role="user",
                        content=[],
                        created_at=turn_started_at or time.time(),
                    )
                )
        elif isinstance(payload, r.TranscriptFinal):
            self._revision += 1
            self._ids.setdefault(payload.item_id, payload.item_id)
            self._chat._upsert_item(
                llm.ChatMessage(
                    id=payload.item_id,
                    role="user",
                    content=[payload.text],
                    created_at=turn_started_at or time.time(),
                )
            )
        elif isinstance(payload, r.MessageStarted):
            self._ids[payload.output.item_id] = payload.output.item_id
            self._active.setdefault(payload.output.response.response_id, set()).add(
                payload.output.item_id
            )
        elif isinstance(payload, r.ToolCall):
            self._ids[payload.item_id] = payload.item_id
            self._active.setdefault(payload.call.response_id, set()).add(payload.item_id)
        elif isinstance(payload, r.ResponseEnded):
            self._active.pop(payload.response.response_id, None)
        elif isinstance(payload, r.ItemTruncated):
            if payload.kept_text is None or payload.content_index != 0:
                return
            # The acknowledgment can arrive after a history edit. Only update
            # an existing item whose current server identity still matches.
            for local_id, server_id in self._ids.items():
                if server_id != payload.item_id:
                    continue
                item = self._chat.get_by_id(local_id)
                if isinstance(item, llm.ChatMessage) and item.role == "assistant":
                    self._revision += 1
                    self._variants.setdefault(local_id, set()).update(
                        (item.text_content or "", payload.kept_text)
                    )
                    item.content = [payload.kept_text]
                break

    def _shared_target(self, target: llm.ChatContext) -> llm.ChatContext:
        old = {item.id: item for item in self._chat.items}
        additions: list[llm.ChatItem] = []
        seen: set[str] = set()
        positions = {item.id: index for index, item in enumerate(self._chat.items)}
        last = -1
        for item in _items(target):
            if isinstance(item, llm.FunctionCall) and item.extra.get(
                "__livekit_agents_tool_non_blocking"
            ):
                raise HistoryUpdateError("Prism does not support tool progress updates")
            if item.id in seen:
                raise HistoryUpdateError("Prism history item IDs must be unique")
            seen.add(item.id)
            previous = old.get(item.id)
            if previous is None:
                additions.append(item)
                continue
            if additions or positions[item.id] < last:
                raise HistoryUpdateError(
                    "Shared history requires append order; use replace_chat_ctx"
                )
            last = positions[item.id]
            if _wire(previous, validate_text=False) == _wire(item, validate_text=False):
                continue
            if (
                isinstance(previous, llm.ChatMessage)
                and isinstance(item, llm.ChatMessage)
                and previous.role == item.role
                and all(isinstance(c, str) for c in item.content)
                and item.text_content in self._variants.get(item.id, set())
            ):
                continue
            raise HistoryUpdateError(
                "Shared history conflicts with confirmed content; use replace_chat_ctx"
            )
        return llm.ChatContext([*self._chat.items, *additions])

    async def sync(self, target: llm.ChatContext, *, shared: bool = False) -> HistorySyncResult:
        """Journal acknowledgments before the next write; never replay a failed edit."""
        async with self._lock:
            result = HistorySyncResult()
            baseline = {item.id for item in self._chat.items}
            failure: Exception | None = None
            try:
                if self._failure is not None:
                    # Existing live results can settle while local history needs repair.
                    additions = [i for i in _items(target) if self._chat.get_by_id(i.id) is None]
                    if not additions or not all(
                        isinstance(i, llm.FunctionCallOutput) and i.call_id in self._calls
                        for i in additions
                    ):
                        raise self._failure
                    target = llm.ChatContext([*self._chat.items, *additions])
                elif shared:
                    target = self._shared_target(target)
                await self._sync(target, result)
            except Exception as caught_error:
                failure = caught_error
            known = (
                failure is None
                or (isinstance(failure, HistoryUpdateError) and failure.recoverable)
                or (isinstance(failure, r.RimeRealtimeError) and failure.fault.scope == "event")
            )
            arrived = {item.id for item in self._chat.items} - baseline - result.created
            if known and result.created and arrived:
                try:
                    await self._repair_order(baseline, result)
                except Exception:
                    # A missing active item is not deleted. Its order is unknown,
                    # so no guessed projection can be reported as confirmed.
                    known = False
                    failure = HistoryUpdateError(
                        "Prism history order is uncertain", status="uncertain"
                    )
            if failure is not None:
                status: Literal["rejected", "partial", "uncertain"] = (
                    ("partial" if result.committed else "rejected") if known else "uncertain"
                )
                error = HistoryUpdateError(
                    "Prism history update was partially applied; reconcile before continuing"
                    if status == "partial"
                    else str(failure)
                    if isinstance(failure, HistoryUpdateError)
                    else "Prism history update " + status,
                    status=status,
                    confirmed_chat_ctx=self.projection(llm.ChatContext.empty()) if known else None,
                )
                result.failure = error
                self._failure = error
            return result

    async def _repair_order(self, baseline: set[str], result: HistorySyncResult) -> None:
        """Use at most one read when events cross an acknowledgment boundary.

        The SDK's event queue and request futures do not expose a common sequence
        number. A snapshot can establish completed-item order without replaying a
        write. Omitted baseline items are preserved, including active calls.
        """
        if result.history_read:
            raise HistoryUpdateError(
                "History already used its identity repair read", status="uncertain"
            )
        result.history_read = True
        snapshot = await self._sdk.get_history()
        by_call = {i.call_id: i.id for i in snapshot if isinstance(i, r.HistoryFunctionResult)}
        for item in self._chat.items:
            if isinstance(item, llm.FunctionCallOutput) and item.id not in self._ids:
                if server_id := by_call.get(item.call_id):
                    self._ids[item.id] = server_id
        local = {self._ids.get(item.id): item for item in self._chat.items if item.id in self._ids}
        ordered = [local[item.id] for item in snapshot if item.id in local]
        covered = {item.id for item in ordered}
        required = {item.id for item in self._chat.items} - baseline
        if not required <= covered:
            raise HistoryUpdateError(
                "Concurrent history order cannot be confirmed", status="uncertain"
            )
        iterator = iter(ordered)
        self._chat.items[:] = [
            next(iterator) if item.id in covered else item for item in self._chat.items
        ]
        self._revision += 1

    async def _sync(self, target: llm.ChatContext, result: HistorySyncResult) -> None:
        response_revision = self._response_revision
        old = list(self._chat.items)
        desired = _items(target)
        old_by_id = {i.id: i for i in old}
        new_by_id = {i.id: i for i in desired}
        unchanged = {
            key
            for key in old_by_id.keys() & new_by_id.keys()
            if _wire(old_by_id[key], validate_text=False)
            == _wire(new_by_id[key], validate_text=False)
        }
        completed = {i.call_id for i in old if isinstance(i, llm.FunctionCallOutput)}
        pending = {
            i.call_id for i in old if isinstance(i, llm.FunctionCall) and i.call_id not in completed
        }
        pending_ids = {
            i.id for i in old if isinstance(i, llm.FunctionCall) and i.call_id in pending
        }
        self.validate(llm.ChatContext(desired), existing=pending_ids, unchanged=unchanged)
        removed = [i for i in old if i.id not in new_by_id]
        added = [i for i in desired if i.id not in old_by_id]
        retained = [i.id for i in old if i.id in new_by_id]
        desired_retained = [i.id for i in desired if i.id in old_by_id]
        append_only = (
            not removed
            and len(unchanged) == len(old)
            and [i.id for i in desired[: len(old)]] == [i.id for i in old]
        )
        # Prism places an imported result directly after its call, even when
        # newer turns exist. Complete those calls without replacing their history.
        insert_results = (
            not removed
            and len(unchanged) == len(old)
            and retained == desired_retained
            and all(
                isinstance(i, llm.FunctionCallOutput)
                and i.call_id in pending
                and i.call_id not in self._calls
                for i in added
            )
        )
        additions_only = append_only or insert_results
        delete_only = not added and len(unchanged) == len(desired) and retained == desired_retained
        rebuild = not additions_only and not delete_only
        # A live call cannot be imported, replaced, or hidden by a history edit.
        if rebuild and (self._active or pending):
            raise HistoryUpdateError(
                "Finish active replies and tool calls before replacing history"
            )
        if any(isinstance(i, llm.FunctionCall) and i.call_id in pending for i in removed):
            raise HistoryUpdateError("An active tool call must receive its result before removal")
        protected = set().union(*self._active.values()) if self._active else set()
        if any(self._ids.get(i.id, i.id) in protected for i in removed):
            raise HistoryUpdateError("History edits cannot remove an active reply's input")
        # Importing a result has server-defined placement directly after its call.
        # Reject requested orders that would require splitting that pair.
        to_create = desired if rebuild else added
        requested_calls = {
            item.call_id: item for item in desired if isinstance(item, llm.FunctionCall)
        }
        for item in to_create:
            _wire(item)
            if isinstance(item, llm.FunctionCallOutput) and (
                rebuild or item.call_id not in self._calls
            ):
                index = desired.index(item)
                previous = desired[index - 1] if index else None
                if not isinstance(previous, llm.FunctionCall) or previous.call_id != item.call_id:
                    raise HistoryUpdateError(
                        "Imported tool results must immediately follow their calls"
                    )
            if isinstance(item, llm.FunctionCallOutput) and item.call_id in self._calls:
                # LiveKit marks the execution copy when a tool becomes nonblocking.
                # Read that annotation from the requested context, not shared objects.
                if requested_calls[item.call_id].extra.get("__livekit_agents_tool_non_blocking"):
                    raise HistoryUpdateError("Prism does not support tool progress updates")
        deletions = old if rebuild else removed
        if any(item.id not in self._ids for item in deletions):
            # Resolve server IDs for typed turns and live results not returned by
            # the older submit_tool_result interface. Never interpret absent active
            # items as deleted; unknown identity prevents mutation.
            result.history_read = True
            snapshot = await self._sdk.get_history()
            by_call = {i.call_id: i.id for i in snapshot if isinstance(i, r.HistoryFunctionResult)}
            for item in deletions:
                if item.id not in self._ids and isinstance(item, llm.FunctionCallOutput):
                    server_id = by_call.get(item.call_id)
                    if server_id:
                        self._ids[item.id] = server_id
                if item.id not in self._ids:
                    raise HistoryUpdateError("Prism has not published the history item's server ID")
        for item in deletions:
            # Deleting either half retires the complete call/result pair.
            if self._chat.get_by_id(item.id) is None:
                continue
            if rebuild and self._response_revision != response_revision:
                raise HistoryUpdateError("A new reply started during history replacement")
            await self._sdk.delete_item(self._ids[item.id])
            result.committed = True
            retiring = [
                i
                for i in self._chat.items
                if i.id == item.id
                or (
                    isinstance(item, (llm.FunctionCall, llm.FunctionCallOutput))
                    and isinstance(i, (llm.FunctionCall, llm.FunctionCallOutput))
                    and item.call_id == i.call_id
                )
            ]
            for retired in retiring:
                self._chat.items.remove(retired)
                self._ids.pop(retired.id, None)
                self._variants.pop(retired.id, None)
                self._played.pop(retired.id, None)
                if isinstance(retired, llm.FunctionCall):
                    self._calls.pop(retired.call_id, None)
            self._revision += 1
            if rebuild and self._response_revision != response_revision:
                raise HistoryUpdateError("A new reply started during history replacement")
        for item in to_create:
            if rebuild and self._response_revision != response_revision:
                raise HistoryUpdateError("A new reply started during history replacement")
            if (
                not rebuild
                and isinstance(item, llm.FunctionCallOutput)
                and item.call_id in self._calls
            ):
                ref = self._calls[item.call_id][0]
                wire = cast(r.HistoryFunctionResult, _wire(item))
                await self._sdk.submit_tool_result(ref, wire.output)
                result.submitted.append((ref, item.model_copy(deep=True)))
            else:
                ref_item = await self._sdk.create_item(_wire(item))
                self._ids[item.id] = ref_item.item_id
            result.committed = True
            self._revision += 1
            result.created.add(item.id)
            # Provisional order. Interleaved observations require one authoritative
            # read before this update returns or exposes a recoverable failure.
            self._chat.items.insert(
                min(desired.index(item), len(self._chat.items)), item.model_copy(deep=True)
            )
            if rebuild and self._response_revision != response_revision:
                raise HistoryUpdateError("A new reply started during history replacement")
