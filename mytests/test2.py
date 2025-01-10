from __future__ import annotations

import logging
import uuid
from typing import Annotated

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai

load_dotenv(dotenv_path=".env")


logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

topics = [
    {
        "question": "Why do you think it's important to summarize data using measures like mean, median, and mode instead of listing all data points?",
        "summary": "Descriptive statistics simplify large datasets by summarizing them with key numbers. Measures of central tendency—mean, median, and mode—help identify the central value of a dataset. The mean provides the average, sensitive to outliers; the median is the middle value, unaffected by extreme values; and the mode identifies the most frequent value. These measures are distinct and chosen based on the data's nature and purpose.",
    },
    {
        "question": "How can outliers affect the calculation and interpretation of the mean in a dataset?",
        "summary": "The mean is calculated by summing all data values and dividing by their count. While it is a valuable measure of central tendency, it is sensitive to outliers, which can skew results. For instance, adding a single high value to a dataset can disproportionately increase the mean, making it less representative of the majority.",
    },
    {
        "question": "When comparing datasets with outliers, why might the median be a better representation of the center than the mean?",
        "summary": "The median represents the middle value when data is sorted. It splits the dataset into two equal halves and remains unaffected by extreme values. This makes it a reliable measure for datasets with significant outliers, providing a clearer view of the central tendency.",
    },
    {
        "question": "Can you think of situations where the mode is more useful than the mean or median?",
        "summary": "The mode identifies the most frequently occurring value in a dataset. It is particularly useful in qualitative data or when assessing popularity. Data can have multiple modes (bimodal, multimodal), showing several frequently occurring values, offering insights into distribution patterns not revealed by the mean or median.",
    },
    {
        "question": "What does it mean for a dataset to have a low or high standard deviation, and why is this important?",
        "summary": "Standard deviation measures how much data values deviate from the mean. A low standard deviation indicates tightly clustered data, while a high standard deviation suggests more spread. This measure is crucial for understanding variability and consistency within a dataset, impacting interpretations in areas like quality control and risk assessment.",
    },
    {
        "question": "How does the five number summary provide a comprehensive overview of a dataset?",
        "summary": "The five number summary includes the minimum, first quartile, median, third quartile, and maximum. It divides data into four parts, each containing 25% of the data points, and provides insights into the dataset’s spread, central tendency, and distribution. Metrics like the interquartile range (IQR) highlight data concentration and outliers.",
    },
    {
        "question": "What advantages do boxplots offer when comparing multiple datasets?",
        "summary": "Boxplots visually represent the five number summary, including the interquartile range and median. Whiskers highlight variability, while outliers are easily identified. They enable quick comparisons between datasets, showing differences in spread, skewness, and central tendency.",
    },
    {
        "question": "How do histograms make frequency distributions easier to interpret?",
        "summary": "Histograms transform frequency distributions into visual bar charts, with bars representing data frequencies within ranges. This allows for intuitive insights into data concentration, spread, and patterns like skewness, aiding quick analysis and decision-making.",
    },
    {
        "question": "How does skewness affect the relationship between the mean and median in a dataset?",
        "summary": "Skewness describes asymmetry in data distribution. In right-skewed data, the mean is pulled higher by a long tail of large values, while in left-skewed data, it is pulled lower. The median remains central, often closer to the dataset’s true middle, making it a more robust measure in skewed datasets.",
    },
    {
        "question": "Can you think of a real-world scenario where understanding both the mean and standard deviation is critical?",
        "summary": "Combining measures of central tendency with variation offers a deeper understanding of datasets. For example, in quality control, the mean ensures products meet specifications, while the standard deviation ensures consistency. Together, they provide a full picture of data behavior, guiding practical decisions.",
    },
]


class VoiceCoursePracticeFunctionContext(llm.FunctionContext):
    def __init__(self):
        # map topics, adding a reviewed boolean to each entry, and initializing review to False
        self.topics = [{"reviewed": False, **topic} for topic in topics]

        logger.info(f"Initialized with {len(self.topics)} topics.")
        super().__init__()

    @llm.ai_callable()
    async def get_topic(self):
        """Called when you need a new topic to review with the student."""
        # find the first topic that hasn't been reviewed yet
        topic = next((t for t in self.topics if not t["reviewed"]), None)
        if topic:
            topic["reviewed"] = True
            val = f"""Topic starting question: "{topic['question']}". Topic summary: {topic['summary']}"""
            logger.info(f"New topic: {val}")
            return val

        logger.info("Everything is reviewed. The student is all done.")
        return "Everything is reviewed. The student is all done."


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    await run_multimodal_agent(ctx, participant)

    logger.info("agent started")


# Custom serializer for JSON encoding. FIXME move
def __custom_serializer(obj):
    if isinstance(obj, uuid.UUID):
        return str(obj)  # Convert UUID to string
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


async def run_multimodal_agent(ctx: JobContext, participant: rtc.RemoteParticipant):
    logger.info(
        "***** starting multimodal agent %s %s %s",
        participant.name,
        participant.attributes,
        participant.metadata,
    )

    instructions = (
        "You are a tutor talking to a student in a graduate degree program. The student wants to practice for "
        "an exam. You can use the provided functions to get information about a list of topics the student should review."
        "Once the student has demonstrated that they are comfortable with a topic, you can move on to the next one."
        "Don't talk too much. As long as the student seems confident with the material, ask questions more than just "
        "repeating information that they likely already know."
    )

    model = openai.realtime.RealtimeModel(
        instructions=instructions,
        modalities=["audio", "text"],
    )
    assistant = MultimodalAgent(
        model=model,
        fnc_ctx=VoiceCoursePracticeFunctionContext(),
    )
    assistant.start(ctx.room, participant)

    session = model.sessions[0]
    session.conversation.item.create(
        llm.ChatMessage(
            role="assistant",
            content="""Start off with a short intro by saying something like "Let's get started reviewing." 
            Then use the provided function to select a topic and start reviewing.""",
        )
    )
    session.response.create()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
