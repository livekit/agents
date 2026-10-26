"""Digest simulation transcripts and verify findings with a bounded reviewer."""

from dataclasses import dataclass

import httpx
from pydantic import BaseModel

from livekit.agents import Agent, AgentServer, ModelRetry, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage


class RecordingNotReady(Exception):
    """The application can retry after the recording becomes available."""


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
    job_id: str
    description: str
    suggestion: str
    label: str


class Analysis(BaseModel):
    going_well: str
    to_improve: str
    issues: list[Issue]


@dataclass
class ReviewContext:
    """What the reviewer's tools read, and its read budget."""

    transcripts: dict[str, str]
    reads_left: int = 6


digester = Agent(
    instructions="Distill one simulated conversation.",
    llm="openai/gpt-5.6-luna",
    output_type=Digest,
)
reviewer = Agent[ReviewContext](
    instructions="Review the digests. Verify your top findings. Submit at most five issues.",
    llm="openai/gpt-5.6-luna",
    output_type=Analysis,
)


@reviewer.tool
async def read_transcript(ctx: RunContext[ReviewContext], job_id: str) -> str:
    if ctx.context.reads_left == 0:
        return "Verification read limit reached; rely on the digests."
    ctx.context.reads_left -= 1
    return ctx.context.transcripts[job_id][:15000]


@reviewer.tool
async def search_transcripts(ctx: RunContext[ReviewContext], text: str) -> str:
    """How many conversations did the same thing, and the lines to cite."""
    hits = []
    for job_id, transcript in ctx.context.transcripts.items():
        hits += [f"{job_id}: {line}" for line in transcript.splitlines() if text in line]
    return "\n".join(hits)[:5000]


@reviewer.output_validator
async def cites_real_jobs(ctx: RunContext[ReviewContext], output: Analysis) -> Analysis:
    """Structured output, checked. ModelRetry sends the model back with the reason."""
    unknown = [i.job_id for i in output.issues if i.job_id not in ctx.context.transcripts]
    if unknown:
        raise ModelRetry(f"These issues cite job ids that do not exist: {unknown}")
    return output


class BatchRequest(BaseModel):
    jobs: list[Job]


async def summarize_jobs(jobs: list[Job]) -> Analysis:
    digests: list[Digest] = []
    transcripts: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=30) as http:
        for job in jobs:
            history = await fetch_chat_history(http, job.chat_history_url)
            if history is None:
                raise RecordingNotReady(job.job_id)
            transcript = render_transcript(history)
            transcripts[job.job_id] = transcript
            prompt = (
                f"{job.scenario}\n{job.expectations}\nPassed: {job.success}\n{transcript[:200_000]}"
            )
            result = await digester.run(prompt)
            if result.status == "input_required":
                raise ValueError(f"Unattended digest requested human input: {result.question}")
            digests.append(result.output)
    brief = f"{sum(j.success for j in jobs)}/{len(jobs)} passed.\n"
    brief += "\n\n".join(d.model_dump_json() for d in digests)
    result = await reviewer.run(brief, context=ReviewContext(transcripts), max_steps=12)
    if result.status == "input_required":
        raise ValueError(f"Unattended review requested human input: {result.question}")
    return result.output


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(
    endpoint="summarize", description="Summarizes simulation results.", public=False
)
async def handle(ctx: A2ASessionContext) -> Analysis:
    request = BatchRequest.model_validate_json(ctx.message.text_content or "")
    return await summarize_jobs(request.jobs)
