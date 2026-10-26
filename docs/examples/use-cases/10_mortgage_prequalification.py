"""A mortgage questionnaire with persisted answers and a validated final application."""

from dataclasses import dataclass, field

from pydantic import BaseModel

from livekit.agents import Agent, AgentServer, ModelRetry, RunContext, RunResult, persisted
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage


@dataclass
class Question:
    id: str
    text: str
    choices: list[str] = field(default_factory=list)


@dataclass
class Applicant:
    """The applicant details and answers stored by explicit context saves."""

    loan_type: str = "conventional"
    answers: dict[str, str] = field(default_factory=dict)


@dataclass
class ApplicationContext:
    applicant: Applicant = persisted(default_factory=Applicant)


class Application(BaseModel):
    loan_type: str
    answers: dict[str, str]


def questions_for(loan_type: str) -> list[Question]:
    questions = [
        Question("income", "What is your gross annual household income?"),
        Question("employment", "How are you employed?", ["salaried", "self-employed", "retired"]),
        Question("down_payment", "How much do you have for a down payment?"),
    ]
    if loan_type == "fha":
        questions.append(Question("first_home", "Is this your first home?", ["yes", "no"]))
    if loan_type == "va":
        questions.append(Question("service", "Which branch did you serve in, and when?"))
    return questions


loan_officer = Agent[ApplicationContext](
    instructions="Harbor Credit Union home loans. Ask the outstanding questions one at a time, record each answer, and produce the application only when all are answered. Never quote a rate.",
    llm="google/gemma-4-31b-it",
    output_type=Application,
)


@loan_officer.instructions
def questionnaire(ctx: RunContext[ApplicationContext]) -> str:
    applicant = ctx.context.applicant
    return "\n".join(
        f"{q.id}: {q.text}; choices={q.choices}; answer={applicant.answers.get(q.id, 'missing')}"
        for q in questions_for(applicant.loan_type)
    )


@loan_officer.tool
async def record_answer(ctx: RunContext[ApplicationContext], question_id: str, answer: str) -> str:
    """Record an answer to one of the current application's questions."""
    applicant = ctx.context.applicant
    question = next((q for q in questions_for(applicant.loan_type) if q.id == question_id), None)
    if question is None:
        raise ModelRetry("Unknown question ID")
    if question.choices and answer not in question.choices:
        raise ModelRetry(f"Choose one of: {', '.join(question.choices)}")
    applicant.answers[question_id] = answer
    return "answer recorded"


@loan_officer.output_validator
async def complete_application(
    ctx: RunContext[ApplicationContext], output: Application
) -> Application:
    applicant = ctx.context.applicant
    required = {q.id for q in questions_for(applicant.loan_type)}
    if not required <= applicant.answers.keys():
        raise ModelRetry("Collect all outstanding answers first")
    if output.loan_type != applicant.loan_type or output.answers != applicant.answers:
        raise ModelRetry("Return the recorded application exactly")
    return output


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="home-loans", description="Harbor Credit Union home loans.")
async def text(ctx: A2ASessionContext) -> RunResult[Application]:
    app = ApplicationContext()
    await ctx.store.load_context(app)
    result = await loan_officer.run(ctx.message, message_history=ctx.message_history, context=app)
    await ctx.store.save_context(app)
    return result
