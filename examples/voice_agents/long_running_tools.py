import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Literal  # noqa: F401

from dotenv import load_dotenv
from openai import AsyncOpenAI

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.llm import ToolError, function_tool
from livekit.agents.voice.events import RunContext
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("annotated-tool-args")
logger.setLevel(logging.INFO)

load_dotenv()


client = AsyncOpenAI()

## This example demonstrates how to use async long-running function tools
# with a mock GPU training cluster as an example.

Status = Literal["pending", "running", "completed", "failed", "timeout"]
JobType = Literal["inference", "training"]


@dataclass
class Job:
    job_type: JobType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: Status = field(default="pending")
    progress: float = field(default=0)
    error: str | None = field(default=None)

    def status_message(self):
        if self.status == "pending":
            return {
                "status": "pending",
                "message": f"{self.job_type.capitalize()} job {self.id} is pending, waiting for available GPU resources...",
            }
        elif self.status == "running":
            return {
                "status": "running",
                "message": f"{self.job_type.capitalize()} job {self.id} is running, progress: {self.progress}%",
            }
        elif self.status == "completed":
            return {
                "status": "completed",
                "message": f"{self.job_type.capitalize()} job {self.id} is completed.",
            }
        elif self.status == "failed":
            return {
                "status": "failed",
                "message": f"{self.job_type.capitalize()} job {self.id} failed. Error: {self.error}",
            }
        elif self.status == "timeout":
            return {
                "status": "timeout",
                "message": f"{self.job_type.capitalize()} job {self.id} timed out. No GPU resources available.",
            }
        else:
            raise ValueError(f"Invalid job status: {self.status}")


@dataclass
class Checkpoint:
    id: str
    version: int
    s3_path: str

    def ckpt_message(self):
        return {
            "id": self.id,
            "version": self.version,
            "s3_path": self.s3_path,
        }


@dataclass
class UserData:
    jobs: dict[str, Job] = field(default_factory=dict)
    checkpoints: dict[str, Checkpoint] = field(default_factory=dict)
    num_gpu_available: int = field(default=3)

    def add_job(self, job_type: JobType) -> Job:
        job = Job(job_type=job_type)
        self.jobs[job.id] = job
        return job

    def schedule_job(self, job_id: str):
        job = self.jobs[job_id]

        if self.num_gpu_available == 0:
            return False

        self.num_gpu_available -= 1
        job.status = "running"
        return True

    def finish_job(self, job_id: str):
        job = self.jobs.pop(job_id)
        if job.status in ["failed", "timeout"]:
            return

        if job.status == "running":
            self.num_gpu_available += 1
            job.status = "completed" if job.error is None else "failed"
        elif job.status == "pending":
            job.status = "timeout"


class ModelName(str, Enum):
    LLAMA_3_70B = "llama-3-70b"
    LLAMA_3_8B = "llama-3-8b"
    LLAMA_3_1B = "llama-3-1b"
    QWEN25_3B = "qwen2.5-3b"
    DEEPSEEK_R1 = "deepseek-r1-7b"
    DEEPSEEK_V3 = "deepseek-v3-7b"
    MISTRAL_7B = "mistral-7b"
    OPENAI_GPT_4O = "gpt-4o"
    OPENAI_GPT_4O_MINI = "gpt-4o-mini"


@function_tool
async def check_cluster_status(ctx: RunContext[UserData]):
    """
    Check the status of the cluster.
    """

    result = json.dumps(
        {
            "num_gpu_available": ctx.userdata.num_gpu_available,
            "jobs": [job.status_message() for job in ctx.userdata.jobs.values()],
            "checkpoints": [
                checkpoint.ckpt_message() for checkpoint in ctx.userdata.checkpoints.values()
            ],
        },
        indent=2,
    )
    return result


@function_tool(reply_mode="interrupt")
async def model_inference(
    model_name: Annotated[ModelName, "The model to use for the inference"],
    prompt: Annotated[str, "The prompt to use for the inference"],
    ctx: RunContext[UserData],
):
    """
    Called when the user wants to run inference on a model.
    """

    if model_name not in ctx.userdata.checkpoints:
        raise ToolError(
            f"Model {model_name} has not been trained yet, please use `model_training` to train it first."
        )

    job = ctx.userdata.add_job("inference")
    scheduled = ctx.userdata.schedule_job(job.id)

    yield job.status_message()

    # if not resource available, wait for 10 seconds and try again
    if not scheduled:
        await asyncio.sleep(10)
        logger.info(f"Waiting for resource available for job={job.id}")
        scheduled = ctx.userdata.schedule_job(job.id)

    # if still not resource available, finish the job with timeout
    if not scheduled:
        ctx.userdata.finish_job(job.id)
        yield job.status_message()
        return

    # simulate inference time
    for i in range(5):
        await asyncio.sleep(3)
        job.progress = (i + 1) * 10

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        response_text = response.choices[0].message.content
        job.progress = 100
    except Exception:
        job.error = "Inference failed"
        ctx.userdata.finish_job(job.id)
        yield job.status_message()
        return

    ctx.userdata.finish_job(job.id)
    yield {**job.status_message(), "model_response": response_text}


@function_tool(reply_mode="when_idle")
async def model_training(
    model_name: Annotated[ModelName, "The LLM model to train"],
    technique: Annotated[Literal["sft", "dpo", "ppo"], "The training technique to use"],
    ctx: RunContext[UserData],
):
    """
    Called when the user wants to train a model.
    """

    job = ctx.userdata.add_job("training")
    scheduled = ctx.userdata.schedule_job(job.id)

    yield job.status_message()

    # if not resource available, wait for 10 seconds and try again
    if not scheduled:
        await asyncio.sleep(10)
        scheduled = ctx.userdata.schedule_job(job.id)

    # if still not resource available, finish the job with timeout
    if not scheduled:
        ctx.userdata.finish_job(job.id)
        yield job.status_message()
        return

    # simulate training time
    for i in range(10):
        await asyncio.sleep(4)
        logger.info(
            f"Training progress of job={job.id} model={model_name} technique={technique}: {i + 1} / 10"
        )
        job.progress = (i + 1) * 10

    ctx.userdata.finish_job(job.id)

    if ctx.userdata.checkpoints.get(model_name):
        ctx.userdata.checkpoints[model_name].version += 1
        ctx.userdata.checkpoints[
            model_name
        ].s3_path = f"s3://{model_name}/checkpoint_{model_name}_v{ctx.userdata.checkpoints[model_name].version}.pt"
    else:
        ctx.userdata.checkpoints[model_name] = Checkpoint(
            id=model_name, version=1, s3_path=f"s3://{model_name}/checkpoint_{model_name}_v1.pt"
        )

    yield {
        **job.status_message(),
        "training_technique": technique,
        "checkpoint": ctx.userdata.checkpoints[model_name].ckpt_message(),
    }


class ModelTrainingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful assistant that manages long-running model training and inference jobs on a GPU cluster. "
                "You interact asynchronously with tools and should notify the user once results are available.\n\n"
                "Available tools:\n"
                "- `model_training`: Start training a model with a selected technique. Creates a checkpoint upon success.\n"
                "- `model_inference`: Run inference using a trained model checkpoint.\n"
                "- `check_cluster_status`: View current GPU availability, jobs, and checkpoints.\n\n"
                "Behavior:\n"
                "- You do not wait for job results synchronously; tools return progress or results as they become available.\n"
                "- When a job finishes **while you are speaking**, gracefully pause and say something like: "
                "'Oh, by the way, a job just completed: (details).'\n"
                "- When **idle** and a job finishes, proactively tell the user the result.\n"
                "- If a model hasn't been trained, inform the user to run `model_training` first.\n"
                "- Use `check_cluster_status` for real-time insights before scheduling new jobs.\n\n"
                "Be concise, informative, and always keep the user updated on job progress and system status. "
                "Do not output markdown text since you are talking to a user. Format response to be more conversational."
            ),
            tools=[model_training, model_inference, check_cluster_status],
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(user_input="Greet to user and tell them what you can do")


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    userdata = UserData()
    agent = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        userdata=userdata,
    )

    await agent.start(agent=ModelTrainingAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
