"""
- Harbor Credit Union mortgage pre-qualification over text: a list of questions that depends on
  the loan, answered over days.
- `Questionnaire` is an `AgentTask` awaited from a tool: in text mode a handoff plus a deferred
  call, each reply a turn of the task, on any worker.
- The question list is its only argument; answers go into the `Applicant` state, not the object,
  so a rebuild after a deploy loses nothing.
- `record_answer` returns `Completed(answers)` when the last one lands, and the parent's tool call
  gets the value.
"""

from dataclasses import dataclass, field

from livekit.agents import Agent, AgentServer, AgentTask, Completed, RunContext, function_tool
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage


@dataclass
class Question:
    id: str
    text: str
    choices: list[str] = field(default_factory=list)


@dataclass
class Applicant:
    """State: saved with the lane after every run. The questionnaire writes the answers here."""

    loan_type: str = "conventional"
    answers: dict[str, str] = field(default_factory=dict)


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


@dataclass
class Questionnaire(AgentTask[dict[str, str]]):
    """Asks each question in turn, in its own words, and completes with the answers.

    Awaited from a tool in text mode it becomes the current agent, the applicant's replies are its
    turns on any worker, and Completed delivers the value to the parent's pending tool call. Its
    only argument is the question list; the answers so far are in the state, not in the object.
    """

    questions: list[Question]

    def __post_init__(self) -> None:
        lines = []
        for q in self.questions:
            choices = f" (choices: {', '.join(q.choices)})" if q.choices else ""
            lines.append(f"- {q.id}: {q.text}{choices}")
        intro = "Ask these one at a time, conversationally. Record each answer with record_answer."
        super().__init__(instructions=intro + "\n" + "\n".join(lines), llm="google/gemma-4-31b-it")

    @function_tool
    async def record_answer(
        self, ctx: RunContext[None, Applicant], question_id: str, answer: str
    ) -> str | Completed[dict[str, str]]:
        question = next(q for q in self.questions if q.id == question_id)
        if question.choices and answer not in question.choices:
            return f"not one of the choices: {', '.join(question.choices)}"   # the model asks again
        ctx.state.answers[question_id] = answer
        remaining = [q for q in self.questions if q.id not in ctx.state.answers]
        if remaining:
            return f"recorded; next: {remaining[0].text}"
        return Completed(dict(ctx.state.answers))   # the parent's tool call gets this


loan_officer = Agent(
    instructions="Harbor Credit Union home loans. Pre-qualify applicants; never quote a rate.",
    llm="google/gemma-4-31b-it",
    state_type=Applicant,
)


@loan_officer.tool
async def prequalify(ctx: RunContext[None, Applicant]) -> str:
    answers = await Questionnaire(questions_for(ctx.state.loan_type))   # days may pass here
    return f"{len(answers)} answers recorded; hand to underwriting"


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(
    endpoint="home-loans", description="Harbor Credit Union home loans.", state=Applicant
)
async def text(ctx: A2ASessionContext[Applicant]) -> str:
    result = await loan_officer.run(
        ctx.message, message_history=ctx.message_history, state=ctx.state
    )
    return result.input_required or result.output


# In text mode the await is a handoff plus a deferred call: the task is the current agent in the
# journal, each reply is one of its turns, and the Completed value is the pending call's output,
# so the parent resumes on the next turn. Five questions or forty, one class and one tool. The
# same class serves patient intake, claims intake and recruiting screens.
