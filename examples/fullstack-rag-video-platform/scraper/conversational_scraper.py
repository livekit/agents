"""
Conversational Scraper
Talk to the scraper using natural language (chat or voice).
100% open-source with Ollama/LlamaCPP + Whisper + Coqui TTS.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from beast_scraper import BeastScraper, ScrapeResult
from learning.pattern_learner import PatternLearner
from memory.user_memory import UserMemory
from scraper_config import ScraperConfig

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """A message in the conversation"""
    role: str  # user or assistant
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class ConversationalScraper:
    """
    Scraper you can talk to with natural language.

    Features:
    - Natural language scraping requests
    - Remembers who you are across sessions
    - Learns your preferences
    - Plans multi-step scraping workflows
    - 100% open-source LLM (Ollama)
    """

    def __init__(
        self,
        user_id: str,
        config: Optional[ScraperConfig] = None
    ):
        """
        Initialize conversational scraper.

        Args:
            user_id: Unique user identifier
            config: Scraper configuration
        """
        self.user_id = user_id
        self.config = config or ScraperConfig()

        # Core components
        self.scraper = BeastScraper(config)
        self.user_memory = UserMemory(user_id)
        self.pattern_learner = PatternLearner()

        # LLM setup (open-source)
        self.llm = Ollama(
            model=self.config.llm.model,
            base_url=self.config.llm.base_url,
            temperature=self.config.llm.temperature
        )

        # Memory for conversation
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Agent and tools
        self.agent: Optional[AgentExecutor] = None
        self.tools: List[Tool] = []

        # Conversation history
        self.messages: List[ConversationMessage] = []

        logger.info(f"🗣️ Conversational Scraper ready for user: {user_id}")

    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing conversational scraper...")

        # Initialize scraper
        await self.scraper.initialize()

        # Load user memory
        await self.user_memory.load()

        # Setup tools
        self._setup_tools()

        # Create agent
        self._create_agent()

        logger.info("✓ Conversational scraper ready!")

    def _setup_tools(self):
        """Setup tools available to the LLM agent"""

        # Scraping tool
        scrape_tool = Tool(
            name="scrape_url",
            func=self._scrape_url_sync,
            coroutine=self._scrape_url,
            description="""
            Scrape data from a URL.

            Args:
                url (str): The URL to scrape
                extract (str): Comma-separated fields to extract (e.g., "title,price,description")
                mode (str): Scraping mode - "auto", "fast", "browser", or "stealth"

            Returns:
                Dict with scraped data

            Example: scrape_url("https://example.com", "title,price", "browser")
            """
        )

        # Multi-URL scraping
        scrape_many_tool = Tool(
            name="scrape_multiple_urls",
            func=self._scrape_many_sync,
            coroutine=self._scrape_many,
            description="""
            Scrape multiple URLs in parallel.

            Args:
                urls (str): Comma-separated list of URLs
                extract (str): Comma-separated fields to extract

            Returns:
                List of results from each URL

            Example: scrape_multiple_urls("https://site1.com,https://site2.com", "title,price")
            """
        )

        # Search learned patterns
        pattern_tool = Tool(
            name="get_scraping_patterns",
            func=self._get_patterns_sync,
            coroutine=self._get_patterns,
            description="""
            Get learned scraping patterns for a domain.

            Args:
                domain (str): Domain to get patterns for (e.g., "amazon.com")

            Returns:
                Known patterns and selectors for that domain
            """
        )

        # User preferences
        preference_tool = Tool(
            name="remember_preference",
            func=self._remember_preference_sync,
            coroutine=self._remember_preference,
            description="""
            Remember a user preference.

            Args:
                key (str): Preference name
                value (str): Preference value

            Example: remember_preference("favorite_format", "CSV")
            """
        )

        self.tools = [
            scrape_tool,
            scrape_many_tool,
            pattern_tool,
            preference_tool
        ]

    def _create_agent(self):
        """Create the LLM agent with tools"""

        # Agent prompt
        template = """You are an expert web scraping assistant. You help users scrape websites efficiently and intelligently.

You have access to these tools:
{tools}

You also know about the user:
User ID: {user_id}
Preferences: {user_preferences}
Previous conversations: {chat_history}

When the user asks you to scrape something:
1. Understand what they want to extract
2. Determine the best scraping mode (fast for simple sites, browser for JavaScript-heavy sites)
3. Use your tools to execute the scrape
4. Present results clearly
5. Learn from the interaction to improve future scrapes

Always be helpful, efficient, and remember user preferences.

User: {input}

Thought: {agent_scratchpad}
"""

        prompt = PromptTemplate(
            input_variables=["input", "user_id", "user_preferences", "chat_history", "tools", "agent_scratchpad"],
            template=template
        )

        # Create ReAct agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.conversation_memory,
            verbose=True,
            handle_parsing_errors=True
        )

    async def chat(self, message: str) -> str:
        """
        Chat with the scraper using natural language.

        Args:
            message: User's message/request

        Returns:
            Assistant's response
        """
        logger.info(f"User ({self.user_id}): {message}")

        # Save user message
        self.messages.append(ConversationMessage(
            role="user",
            content=message,
            timestamp=datetime.utcnow()
        ))

        # Get user preferences
        preferences = await self.user_memory.get_preferences()

        # Run agent
        try:
            response = await self.agent_executor.ainvoke({
                "input": message,
                "user_id": self.user_id,
                "user_preferences": str(preferences)
            })

            assistant_message = response['output']

        except Exception as e:
            logger.error(f"Error in agent execution: {e}")
            assistant_message = f"I encountered an error: {str(e)}. Let me try a different approach."

        # Save assistant message
        self.messages.append(ConversationMessage(
            role="assistant",
            content=assistant_message,
            timestamp=datetime.utcnow()
        ))

        # Save to user memory
        await self.user_memory.add_conversation(message, assistant_message)

        logger.info(f"Assistant: {assistant_message}")

        return assistant_message

    # Tool implementations
    async def _scrape_url(self, url: str, extract: str = "", mode: str = "auto") -> Dict[str, Any]:
        """Scrape a single URL"""
        extract_list = [e.strip() for e in extract.split(",")] if extract else None

        result = await self.scraper.scrape(
            url=url,
            extract=extract_list,
            mode=mode
        )

        if result.success:
            return {
                "success": True,
                "url": url,
                "data": result.data,
                "response_time": result.response_time
            }
        else:
            return {
                "success": False,
                "url": url,
                "error": result.error
            }

    def _scrape_url_sync(self, *args, **kwargs):
        """Sync wrapper for scrape_url"""
        return asyncio.run(self._scrape_url(*args, **kwargs))

    async def _scrape_many(self, urls: str, extract: str = "") -> List[Dict[str, Any]]:
        """Scrape multiple URLs"""
        url_list = [u.strip() for u in urls.split(",")]
        extract_list = [e.strip() for e in extract.split(",")] if extract else None

        results = await self.scraper.scrape_many(
            urls=url_list,
            extract=extract_list
        )

        return [
            {
                "url": r.url,
                "success": r.success,
                "data": r.data if r.success else None,
                "error": r.error if not r.success else None
            }
            for r in results
        ]

    def _scrape_many_sync(self, *args, **kwargs):
        """Sync wrapper for scrape_many"""
        return asyncio.run(self._scrape_many(*args, **kwargs))

    async def _get_patterns(self, domain: str) -> Dict[str, Any]:
        """Get learned patterns for a domain"""
        pattern = await self.pattern_learner.get_pattern(domain)
        return pattern or {"message": f"No patterns learned yet for {domain}"}

    def _get_patterns_sync(self, *args, **kwargs):
        """Sync wrapper for get_patterns"""
        return asyncio.run(self._get_patterns(*args, **kwargs))

    async def _remember_preference(self, key: str, value: str) -> Dict[str, Any]:
        """Remember a user preference"""
        await self.user_memory.set_preference(key, value)
        return {"success": True, "message": f"I'll remember that your {key} is {value}"}

    def _remember_preference_sync(self, *args, **kwargs):
        """Sync wrapper for remember_preference"""
        return asyncio.run(self._remember_preference(*args, **kwargs))

    def get_conversation_history(self) -> List[ConversationMessage]:
        """Get conversation history"""
        return self.messages

    async def close(self):
        """Clean up resources"""
        await self.scraper.close()
        await self.user_memory.save()

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Example usage
async def main():
    """Example usage of conversational scraper"""

    async with ConversationalScraper(user_id="john_doe") as scraper:
        # Natural language requests
        response = await scraper.chat(
            "Scrape the top 5 articles from news.ycombinator.com and get the titles"
        )
        print(f"Assistant: {response}\n")

        # Follow-up (it remembers context!)
        response = await scraper.chat(
            "Now do the same for techcrunch.com"
        )
        print(f"Assistant: {response}\n")

        # Complex multi-step
        response = await scraper.chat(
            "Monitor prices for 'gaming laptop' on amazon and bestbuy, "
            "then tell me which has the best deal"
        )
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
