import logging
from collections.abc import AsyncIterable

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    ModelSettings,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("interview-agent")

load_dotenv()


class InterviewAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a professional HR interviewer named Kelly. "
            "Your role is to conduct job interviews and assess candidates. "
            "Follow this structured interview process:\n\n"
            "1. INTRODUCTION: Start with a warm greeting and introduce yourself\n"
            "2. CANDIDATE BACKGROUND: Ask for their name and current role/background\n"
            "3. TECHNICAL ASSESSMENT: Based on their job profile, ask relevant technical questions\n"
            "4. BEHAVIORAL QUESTIONS: Ask about past experiences, challenges, and achievements\n"
            "5. MOTIVATION: Understand why they're interested in this opportunity\n"
            "6. CLOSING: End professionally and thank them for their time\n\n"
            "INTERVIEW GUIDELINES:\n"
            "- Be professional but friendly and approachable\n"
            "- Ask one question at a time and wait for complete answers\n"
            "- Listen carefully and ask relevant follow-up questions\n"
            "- Keep questions clear and concise\n"
            "- Adapt your questions based on the candidate's responses\n"
            "- For technical roles, ask about specific skills and experience\n"
            "- For managerial roles, focus on leadership and team management\n"
            "- End the interview by asking if they have any questions\n"
            "- Keep the entire conversation under 15-20 minutes"
        )
        
        # Track interview progress
        self.interview_stage = "introduction"
        self.candidate_name = None
        self.candidate_role = None

    async def on_enter(self):
        # Generate the initial greeting when agent joins
        await self.session.generate_reply(
            instructions="Greet the candidate warmly. Introduce yourself as Kelly, an HR interviewer. "
            "Start with: 'Hello! Welcome to our interview session. My name is Kelly and I'll be conducting your interview today. "
            "Could you please start by telling me your name and what you currently do?'"
        )

    @function_tool
    async def assess_technical_skills(self, context: RunContext, skill_area: str, experience_level: str):
        """Called when assessing candidate's technical skills based on their job profile.
        
        Args:
            skill_area: The technical area to assess (e.g., 'Python', 'cloud computing', 'frontend development')
            experience_level: The candidate's experience level (e.g., 'junior', 'mid-level', 'senior')
        """
        
        logger.info(f"Assessing technical skills in {skill_area} for {experience_level} level candidate")
        
        if experience_level.lower() in ["senior", "lead", "principal"]:
            return f"Ask about their experience with {skill_area} in large-scale systems, architectural decisions, and mentoring junior developers."
        elif experience_level.lower() in ["mid-level", "intermediate"]:
            return f"Ask about practical applications of {skill_area}, problem-solving experiences, and collaboration in team environments."
        else:  # junior/entry level
            return f"Ask about their learning journey with {skill_area}, basic concepts understanding, and willingness to learn."

    @function_tool
    async def evaluate_behavioral_competency(self, context: RunContext, competency: str):
        """Evaluate behavioral competencies like teamwork, leadership, problem-solving.
        
        Args:
            competency: The behavioral competency to evaluate
        """
        
        competency_questions = {
            "teamwork": "Tell me about a time you had to work closely with a difficult team member. How did you handle it?",
            "leadership": "Describe a situation where you had to take initiative or lead a project without formal authority.",
            "problem_solving": "Walk me through a complex problem you solved and the process you used to reach the solution.",
            "adaptability": "Give an example of when you had to adapt to significant changes at work.",
            "communication": "Describe a time when you had to explain a complex technical concept to a non-technical audience."
        }
        
        return competency_questions.get(competency.lower(), 
                                       "Tell me about a challenge you faced at work and how you overcame it.")

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        # Use the default TTS processing with markdown/emoji filtering
        return super().tts_node(text, model_settings)


def prewarm(proc: JobProcess):
    # Preload VAD model for faster startup
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "agent_type": "interviewer",
    }

    # Initialize the session with optimized settings for interviews
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(
            model="gpt-4o",  # Using GPT-4o for better interview quality
            temperature=0.7,
          #  max_tokens=150  # Keep responses concise for voice
        ),
        stt=cartesia.STT(
            model="ink-whisper",
            language="en"
        ),
        tts=cartesia.TTS(
           # model="sonic-2",
          #  voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",  # Professional female voice
            language="en",
           # speed=0.95  # Slightly slower for clarity in interviews
        ),
        # Interview-specific settings
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
        min_interruption_duration=0.2,
        turn_detection=MultilingualModel(),
        # Longer pauses for interview responses
        min_endpointing_delay=1.0,
        max_endpointing_delay=5.0,
    )

    # Setup metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Interview session usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the interview session
    await session.start(
        agent=InterviewAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,  # Useful for interview notes
          #  recording_enabled=False      # Disable recording for privacy
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))