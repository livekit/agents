"""Graphs: code steps whose edges are known before anything runs.

A node is a dataclass whose fields are its inputs, ``run()`` returns the next node, ``End``, or
``InputRequired`` to wait on the person, and the return annotation is the edge the diagram is
drawn from.

A graph is a class and an ``AgentTask``: ``Triage.run(start, state=)`` runs it inline to
``End`` in the turn; ``await Triage(start, state=)`` is the waiting form, a task that asks the
person from a node and runs that node again on the reply (docs/graphs.md).
"""

from __future__ import annotations

import dataclasses
import sys
import types
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import TypeAdapter, ValidationError
from typing_extensions import Never, Self, TypeVar

from .llm import ChatContext, ChatMessage, Tool
from .llm.tool_context import Completed, Failed, InputRequired, Transition
from .voice.agent import AgentTask, ModelSettings

StateT = TypeVar("StateT", default=None)
DepsT = TypeVar("DepsT", default=None)
RunEndT = TypeVar("RunEndT", default=None)
NodeRunEndT = TypeVar("NodeRunEndT", covariant=True, default=Never)

STALE_GRAPH = "stale-graph"
"""The ``livekit/reason`` of a waiting graph whose saved node or state no longer rebuilds."""


class GraphSetupError(TypeError):
    """The graph's nodes or their edges are wrong, found when the class is defined."""


class GraphRuntimeError(RuntimeError):
    """A node did something the graph does not allow while it ran."""


@dataclass
class GraphRunContext(Generic[StateT, DepsT]):
    """What a node's ``run()`` receives: the run's state and deps, and ``reply``, the person's
    answer when the node runs again after its ``InputRequired``."""

    state: StateT
    deps: DepsT
    reply: str | None = None


@dataclass
class End(Generic[RunEndT]):
    """The typed result a node returns to end the run."""

    value: RunEndT


@dataclass(frozen=True)
class Edge:
    """``Annotated[NextNode, Edge(label="...")]``: a label on the edge in the diagram."""

    label: str | None = None


@dataclass
class NodeDef:
    """What the graph knows of one node class: its id and the edges its ``run()`` declares."""

    node: type[BaseNode[Any, Any, Any]]
    node_id: str
    next_node_edges: dict[str, Edge] = field(default_factory=dict)
    end_edge: Edge | None = None
    asks: bool = False

    @property
    def returns_end(self) -> bool:
        return self.end_edge is not None


class BaseNode(ABC, Generic[StateT, DepsT, NodeRunEndT]):
    """A step of a graph; subclass as a dataclass whose fields are the step's inputs.

    ``BaseNode[State, Deps, Output]``: ``Output`` is what the node may end the run with. The
    return annotation of ``run()`` names the nodes it may return, ``End[Output]``, or
    ``InputRequired``, and is the edge the diagram is drawn from.
    """

    @abstractmethod
    async def run(
        self, ctx: GraphRunContext[StateT, DepsT]
    ) -> BaseNode[StateT, DepsT, Any] | End[NodeRunEndT] | InputRequired: ...

    @classmethod
    def get_node_id(cls) -> str:
        return cls.__name__

    @classmethod
    def get_node_def(cls, local_ns: dict[str, Any] | None = None) -> NodeDef:
        """The node's edges, read from the return annotation of ``run()``."""
        try:
            hints = get_type_hints(cls.run, localns=local_ns, include_extras=True)
        except NameError as e:
            raise GraphSetupError(
                f"cannot resolve the return annotation of {cls.get_node_id()}.run: {e}"
            ) from e

        definition = NodeDef(node=cls, node_id=cls.get_node_id())
        if "return" not in hints:
            raise GraphSetupError(f"{cls.get_node_id()}.run has no return annotation")

        for member in _union_members(hints["return"]):
            edge = Edge()
            if get_origin(member) is Annotated:
                member, *extras = get_args(member)
                edge = next((x for x in extras if isinstance(x, Edge)), edge)
            origin = get_origin(member) or member
            if origin is End:
                definition.end_edge = edge
            elif origin is InputRequired:
                definition.asks = True
            elif isinstance(origin, type) and issubclass(origin, BaseNode):
                definition.next_node_edges[origin.get_node_id()] = edge
            else:
                raise GraphSetupError(
                    f"{cls.get_node_id()}.run returns {member!r}, which is neither a node, End "
                    "nor InputRequired"
                )
        return definition


def _union_members(annotation: Any) -> list[Any]:
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return [m for arg in get_args(annotation) for m in _union_members(arg)]
    return [annotation]


@dataclass
class GraphRunResult(Generic[StateT, RunEndT]):
    """What ``Graph.run()`` returns: the ``End``'s value and the state as the run left it."""

    output: RunEndT
    state: StateT


class GraphRun(Generic[StateT, DepsT, RunEndT]):
    """An inline run of a graph, node by node, from ``Graph.iter()``.

    Iterating yields the start node first, then each node the previous one returned, and the
    ``End``. ``next(node)`` runs one node by hand instead.
    """

    def __init__(
        self,
        graph: type[Graph[RunEndT, StateT, DepsT]],
        start_node: BaseNode[StateT, DepsT, RunEndT],
        *,
        state: StateT,
        deps: DepsT,
    ) -> None:
        self.graph = graph
        self.state = state
        self.deps = deps
        self._next_node: BaseNode[StateT, DepsT, Any] | End[RunEndT] = start_node
        self._started = False

    @property
    def next_node(self) -> BaseNode[StateT, DepsT, Any] | End[RunEndT]:
        """The node that runs next, or the ``End`` once the run is over."""
        return self._next_node

    @property
    def result(self) -> GraphRunResult[StateT, RunEndT] | None:
        """The result once the run reached ``End``, ``None`` before."""
        if not isinstance(self._next_node, End):
            return None
        return GraphRunResult(output=self._next_node.value, state=self.state)

    async def next(
        self, node: BaseNode[StateT, DepsT, Any] | None = None
    ) -> BaseNode[StateT, DepsT, Any] | End[RunEndT]:
        """Runs ``node``, the next one by default, and returns what it returned."""
        if node is None:
            if isinstance(self._next_node, End):
                raise GraphRuntimeError("the run has ended")
            node = self._next_node
        self._started = True
        out = await self.graph._step(node, GraphRunContext(state=self.state, deps=self.deps))
        if isinstance(out, InputRequired):
            raise GraphRuntimeError(
                f"{node.get_node_id()} waits on the person; await {self.graph.__name__}(...) "
                "instead of running it inline"
            )
        self._next_node = out
        return out

    def __aiter__(self) -> AsyncIterator[BaseNode[StateT, DepsT, Any] | End[RunEndT]]:
        return self

    async def __anext__(self) -> BaseNode[StateT, DepsT, Any] | End[RunEndT]:
        if not self._started:
            self._started = True
            return self._next_node
        if isinstance(self._next_node, End):
            raise StopAsyncIteration
        return await self.next(self._next_node)


class Graph(AgentTask[RunEndT, StateT], Generic[RunEndT, StateT, DepsT]):
    """A graph: a class whose ``nodes`` are its node classes, edges read from their ``run()``.

    ``Graph[Output, State]``, or ``Graph[Output, State, Deps]`` for the inline form's deps.
    ``Triage.run(start, state=)`` runs it inline to ``End``; ``await Triage(start, state=)``
    from a tool is the waiting form: an ``AgentTask`` that rides the lane's agent stack while a
    node waits on the person, saved as the graph's name, the asking node and the state.
    """

    nodes: ClassVar[Sequence[type[BaseNode[Any, Any, Any]]]] = ()
    node_defs: ClassVar[dict[str, NodeDef]] = {}
    _graph_state_type: ClassVar[Any] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in cls.__dict__.get("__orig_bases__", ()):
            if get_origin(base) is Graph and len(args := get_args(base)) > 1:
                cls._graph_state_type = args[1]

        # the defining scope resolves forward references in the nodes' annotations
        frame = sys._getframe(1)
        while frame.f_back is not None and frame.f_globals.get("__name__") in (__name__, "abc"):
            frame = frame.f_back
        local_ns: dict[str, Any] = dict(frame.f_locals)

        node_defs: dict[str, NodeDef] = {}
        for node in cls.nodes:
            if not (isinstance(node, type) and issubclass(node, BaseNode)):
                raise GraphSetupError(f"{node!r} is not a BaseNode subclass")
            definition = node.get_node_def(local_ns)
            if definition.node_id in node_defs:
                raise GraphSetupError(f"node {definition.node_id} is listed twice")
            node_defs[definition.node_id] = definition
        for definition in node_defs.values():
            for target in definition.next_node_edges:
                if target not in node_defs:
                    raise GraphSetupError(
                        f"{definition.node_id}.run returns {target}, which is not in the graph "
                        f"{cls.__name__}"
                    )
        cls.node_defs = node_defs

    def __init__(self, start_node: BaseNode[StateT, Any, RunEndT], *, state: StateT) -> None:
        super().__init__(instructions="")
        self.state = state
        self._node: BaseNode[StateT, Any, Any] = start_node
        self._stale: str | None = None

    # -- inline --

    @classmethod
    async def run(  # type: ignore[override]
        cls,
        start_node: BaseNode[StateT, DepsT, RunEndT],
        *,
        state: StateT,
        deps: DepsT = None,  # type: ignore[assignment]
    ) -> GraphRunResult[StateT, RunEndT]:
        """Runs the graph from ``start_node`` to ``End`` in the turn; it never waits."""
        async with cls.iter(start_node, state=state, deps=deps) as graph_run:
            async for _ in graph_run:
                pass
        result = graph_run.result
        assert result is not None
        return result

    @classmethod
    @asynccontextmanager
    async def iter(
        cls,
        start_node: BaseNode[StateT, DepsT, RunEndT],
        *,
        state: StateT,
        deps: DepsT = None,  # type: ignore[assignment]
    ) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]:
        """An inline run to step through node by node; see ``GraphRun``."""
        yield GraphRun(cls, start_node, state=state, deps=deps)

    @classmethod
    def mermaid_code(
        cls, *, start_node: type[BaseNode[Any, Any, Any]] | None = None, title: str | None = None
    ) -> str:
        """The graph as a mermaid state diagram, ``[*]`` for the start and the end; a node that
        waits on the person loops on itself."""
        lines: list[str] = []
        if title:
            lines += ["---", f"title: {title}", "---"]
        lines.append("stateDiagram-v2")
        if start_node is not None:
            lines.append(f"  [*] --> {start_node.get_node_id()}")
        for definition in cls.node_defs.values():
            for target, edge in definition.next_node_edges.items():
                lines.append(f"  {definition.node_id} --> {target}{_label(edge)}")
            if definition.end_edge is not None:
                lines.append(f"  {definition.node_id} --> [*]{_label(definition.end_edge)}")
            if definition.asks:
                lines.append(f"  {definition.node_id} --> {definition.node_id}: asks")
        return "\n".join(lines)

    @classmethod
    async def _step(
        cls, node: BaseNode[Any, Any, Any], ctx: GraphRunContext[Any, Any]
    ) -> BaseNode[Any, Any, Any] | End[Any] | InputRequired:
        if not isinstance(node, BaseNode):
            raise GraphRuntimeError(f"{node!r} is not a node")
        node_id = node.get_node_id()
        if node_id not in cls.node_defs:
            raise GraphRuntimeError(f"node {node_id} is not in graph {cls.__name__}")
        out = await node.run(ctx)
        if not isinstance(out, (BaseNode, End, InputRequired)):
            raise GraphRuntimeError(
                f"{node_id}.run returned {out!r}, neither a node, End nor InputRequired"
            )
        if isinstance(out, BaseNode) and out.get_node_id() not in cls.node_defs:
            raise GraphRuntimeError(
                f"{node_id}.run returned {out.get_node_id()}, which is not in graph {cls.__name__}"
            )
        return out

    # -- waiting: the task --

    async def on_enter(self) -> Transition:
        return await self._advance(None)

    async def _take(self, message: ChatMessage) -> Transition:
        return await self._advance(message.text_content or "")

    async def llm_node(
        self, chat_ctx: ChatContext, tools: list[Tool], model_settings: ModelSettings
    ) -> str:
        """On a call the person's turn reaches the graph here: the asking node runs again."""
        said = next(
            (m.text_content for m in reversed(chat_ctx.messages()) if m.role == "user"), None
        )
        transition = await self._advance(said or "")
        if isinstance(transition, InputRequired):
            return transition.question
        self._get_activity_or_raise()._apply_transition(transition)
        return ""

    async def _advance(self, reply: str | None) -> Transition:
        """Runs from the waiting node to ``End`` or the next question."""
        if self._stale is not None:
            return Failed(f"{STALE_GRAPH}: {self._stale}")
        node = self._node
        ctx = GraphRunContext[StateT, Any](state=self.state, deps=None, reply=reply)
        while True:
            out = await self._step(node, ctx)
            if isinstance(out, End):
                return Completed(out.value)
            if isinstance(out, InputRequired):
                self._node = node
                return out
            node, ctx.reply = out, None

    def _arguments(self) -> dict[str, Any]:
        """The node that asked, its class name and its fields as JSON."""
        node = self._node
        fields = TypeAdapter(type(node)).dump_python(node, mode="json") if _fields(node) else {}
        return {"node": node.get_node_id(), "fields": fields}

    @classmethod
    def _rebuild(cls, args: Any) -> Self:
        node_id = (args or {}).get("node")
        definition = cls.node_defs.get(node_id) if isinstance(node_id, str) else None
        node: BaseNode[Any, Any, Any] | None = None
        stale: str | None = None
        if definition is None:
            stale = f"{cls.__name__} has no node {node_id}"
        else:
            saved = args.get("fields") or {}
            if set(saved) != set(_fields(definition.node)):
                stale = f"the fields of {node_id} changed"
            else:
                try:
                    node = (
                        TypeAdapter(definition.node).validate_python(saved)
                        if dataclasses.is_dataclass(definition.node)
                        else definition.node()
                    )
                except (ValidationError, TypeError) as e:
                    stale = f"{node_id} no longer takes its saved fields: {e}"
        graph = cls(cast(Any, node), state=cast(Any, None))
        graph._stale = stale
        return graph

    def _dump_state(self) -> Any:
        return TypeAdapter(self._graph_state_type).dump_python(self.state, mode="json")

    def _restore(self, state: Any, *, call_id: str | None) -> None:
        self._pending_call_id = call_id
        try:
            self.state = TypeAdapter(self._graph_state_type).validate_python(state)
        except ValidationError as e:
            self._stale = self._stale or f"the state no longer fits: {e}"


def _fields(node: Any) -> list[str]:
    return [f.name for f in dataclasses.fields(node)] if dataclasses.is_dataclass(node) else []


def _label(edge: Edge) -> str:
    return f": {edge.label}" if edge.label else ""


__all__ = [
    "STALE_GRAPH",
    "BaseNode",
    "Edge",
    "End",
    "Graph",
    "GraphRun",
    "GraphRunContext",
    "GraphRunResult",
    "GraphRuntimeError",
    "GraphSetupError",
    "InputRequired",
    "NodeDef",
]
