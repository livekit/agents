"""
- The summarizer that scores agent simulations, as submitted tasks instead of asyncio.
- Each `DigestJob` is a pure code `AgentTask` with `concurrency=16`; it raises when a recording
  is not there yet, and `retry` re-submits it with a backoff.
- `Review` is submitted `after=` the digests and reads each result with `ctx.task(id).result()`;
  the reviewer agent has tools with a read budget and an `@output_validator` that raises
  `ModelRetry`.
- The handler returns the review task's id; the caller follows it with `tasks/resubscribe`. A
  crash after 40 of 60 digests resumes with 40 done.
- `Run` is the lane's state: the jobs only. `Deps` holds the recordings client; each task builds
  it in `__post_init__` and passes `deps=`. `Review` fetches the transcripts it verifies against
  and passes them to the reviewer as `ReviewerDeps`.
"""

from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel

from livekit.agents import Agent, AgentServer, AgentTask, Completed, ModelRetry, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage
from livekit.agents.tasks import Retry

RECORDINGS_LAG = Retry(attempts=5, backoff=[5, 10, 20, 40, 60])


class RecordingNotReady(Exception):
    """Raised by a digest when the export is not there yet; `retry` re-submits the task."""


@dataclass
class Deps:
    http: httpx.AsyncClient


async def fetch_chat_history(http: httpx.AsyncClient, url: str) -> list[dict] | None:
    """The recording service's export. May not exist yet right after a simulation ends."""
    r = await http.get(url)
    return None if r.status_code == 404 else r.json()["items"]


def render_transcript(items: list[dict]) -> str:
    lines = []
    for item in items:
        role = item.get("role", item["type"])
        lines.append(f"[{item['id']}] {role}: {' '.join(item.get('content', []))}")
    return "\n".join(lines)


# ---- the data


class Job(BaseModel):
    job_id: str
    chat_history_url: str
    scenario: str
    expectations: str
    success: bool


class Digest(BaseModel):
    """One conversation, distilled. The transcript itself never reaches the reviewer."""

    job_id: str
    outcome: str
    behaviors: str
    evidence: str
    candidate_issues: list[str]


class Issue(BaseModel):
    job_id: str  # the conversation it was seen in
    description: str
    suggestion: str
    label: str  # cites conversation items by id


class Analysis(BaseModel):
    going_well: str
    to_improve: str
    issues: list[Issue]


@dataclass
class Run:
    """The lane's state: typed, saved with every run."""

    jobs: list[Job] = field(default_factory=list)


@dataclass
class ReviewerDeps:
    """What the reviewer's tools read, and its read budget."""

    transcripts: dict[str, str]
    reads_left: int = 6


# ---- the agents


digester = Agent(
    instructions="Distill one simulated conversation.",
    llm="openai/gpt-5.6-luna",
    output_type=Digest,
)
reviewer = Agent(
    instructions="Review the digests. Verify your top findings. Submit at most five issues.",
    llm="openai/gpt-5.6-luna",
    deps_type=ReviewerDeps,
    output_type=Analysis,
)


@reviewer.tool
async def read_transcript(ctx: RunContext[ReviewerDeps, None], job_id: str) -> str:
    if ctx.deps.reads_left == 0:
        return "Verification read limit reached; rely on the digests."
    ctx.deps.reads_left -= 1
    return ctx.deps.transcripts[job_id][:15_000]


@reviewer.tool
async def search_transcripts(ctx: RunContext[ReviewerDeps, None], text: str) -> str:
    """How many conversations did the same thing, and the lines to cite."""
    hits = []
    for job_id, transcript in ctx.deps.transcripts.items():
        hits += [f"{job_id}: {line}" for line in transcript.splitlines() if text in line]
    return "\n".join(hits)[:5_000]


@reviewer.output_validator
async def cites_real_jobs(ctx: RunContext[ReviewerDeps, None], output: Analysis) -> Analysis:
    """Structured output, checked. ModelRetry sends the model back with the reason."""
    unknown = [i.job_id for i in output.issues if i.job_id not in ctx.deps.transcripts]
    if unknown:
        raise ModelRetry(f"These issues cite job ids that do not exist: {unknown}")
    return output


# ---- the tasks: pure code AgentTasks, submitted; the join is a task submitted after= the others


@dataclass
class DigestJob(AgentTask[Digest]):
    job: Job

    def __post_init__(self) -> None:
        super().__init__(deps=Deps(http=httpx.AsyncClient(timeout=30)))

    retry = RECORDINGS_LAG  # when on_enter raises, the task is re-submitted with a due time
    concurrency = 16

    async def on_enter(self, ctx: RunContext[Deps, Run]) -> Completed[Digest]:
        history = await fetch_chat_history(ctx.deps.http, self.job.chat_history_url)
        if history is None:
            raise RecordingNotReady(self.job.job_id)  # retry handles it; the last attempt fails
        transcript = render_transcript(history)
        prompt = (
            f"{self.job.scenario}\n{self.job.expectations}\nPassed: {self.job.success}\n\n"
            f"{transcript[:200_000]}"
        )
        result = await digester.run(prompt)  # no message_history: a fresh conversation
        return Completed(result.output)


@dataclass
class Review(AgentTask[Analysis]):
    digest_ids: list[str]  # the DigestJob tasks this waits on

    def __post_init__(self) -> None:
        super().__init__(deps=Deps(http=httpx.AsyncClient(timeout=30)))

    async def on_enter(self, ctx: RunContext[Deps, Run]) -> Completed[Analysis]:
        # all terminal: this task ran after= them. result() returns at once on a completed row,
        # and raises for a digest that failed every retry; those jobs are left out
        digests: list[Digest] = []
        for task_id in self.digest_ids:
            try:
                digests.append(await ctx.task(task_id).result())
            except Exception:
                pass
        passed = sum(j.success for j in ctx.state.jobs)
        brief = f"{passed}/{len(ctx.state.jobs)} passed.\n\n"
        brief += "\n\n".join(d.model_dump_json() for d in digests)
        transcripts = {}
        for job in ctx.state.jobs:  # what the reviewer verifies against, fetched for this run
            history = await fetch_chat_history(ctx.deps.http, job.chat_history_url)
            if history is not None:
                transcripts[job.job_id] = render_transcript(history)
        deps = ReviewerDeps(transcripts)
        result = await reviewer.run(brief, deps=deps, max_steps=12)
        return Completed(result.output)


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="summarize", description="Summarizes a run.", public=False, state=Run)
async def handle(ctx: A2ASessionContext[Run]) -> str:
    jobs = [Job(**j) for j in ctx.message.data["jobs"]]
    ctx.state.jobs = jobs
    digests = [await ctx.submit(DigestJob(job)) for job in jobs]  # sixty tasks, sixteen at a time
    # Review runs when every digest is terminal; the caller follows it with tasks/resubscribe and
    # reads the Analysis as its artifact
    review = await ctx.submit(Review([d.id for d in digests]), after=digests)
    return review.id


# Against the asyncio version: a crash after 40 of 60 digests resumes with 40 done, since each
# DigestJob is a task row completed in the thread; the reviewer's tool calls are in the Review
# task's history, so a crash mid-review does not repeat its reads; every task has its own stream,
# so a dashboard shows digests landing live. No task here talks to a person, so none has a model
# of its own; digester and reviewer are the agents they call.
