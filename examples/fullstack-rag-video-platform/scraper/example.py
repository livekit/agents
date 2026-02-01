"""
BEAST Scraper - Example Usage
Demonstrates all major features.
"""

import asyncio
import logging
from beast_scraper import BeastScraper, scrape, scrape_many
from conversational_scraper import ConversationalScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_1_simple_scrape():
    """Example 1: Simple single URL scraping"""
    print("\n" + "="*60)
    print("Example 1: Simple Scraping")
    print("="*60)

    result = await scrape("https://news.ycombinator.com")

    print(f"✓ Scraped {result.url}")
    print(f"  Response time: {result.response_time:.2f}s")
    print(f"  Engine used: {result.engine_used}")
    print(f"  Success: {result.success}")


async def example_2_extract_fields():
    """Example 2: Extract specific fields"""
    print("\n" + "="*60)
    print("Example 2: Extract Specific Fields")
    print("="*60)

    result = await scrape(
        "https://news.ycombinator.com",
        extract=["title", "links"],
        mode="browser"
    )

    print(f"Extracted data from {result.url}:")
    print(f"  Fields: {list(result.data.keys())}")


async def example_3_parallel_scraping():
    """Example 3: Scrape multiple URLs in parallel"""
    print("\n" + "="*60)
    print("Example 3: Parallel Scraping")
    print("="*60)

    urls = [
        "https://news.ycombinator.com",
        "https://example.com",
        "https://httpbin.org/html"
    ]

    print(f"Scraping {len(urls)} URLs in parallel...")

    results = await scrape_many(urls, max_concurrent=5)

    print(f"\nResults:")
    for result in results:
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.url} - {result.response_time:.2f}s")


async def example_4_self_improvement():
    """Example 4: Demonstrate self-improvement"""
    print("\n" + "="*60)
    print("Example 4: Self-Improvement")
    print("="*60)

    async with BeastScraper() as scraper:
        url = "https://news.ycombinator.com"

        # First scrape - no learned patterns
        print(f"First scrape (learning)...")
        result1 = await scraper.scrape(url, learn=True)
        print(f"  Time: {result1.response_time:.2f}s")

        # Second scrape - uses learned patterns
        print(f"Second scrape (using learned patterns)...")
        result2 = await scraper.scrape(url, learn=True)
        print(f"  Time: {result2.response_time:.2f}s")

        # View learned pattern
        pattern = await scraper.pattern_learner.get_pattern(url)
        if pattern:
            print(f"\nLearned pattern:")
            print(f"  Best engine: {pattern['best_engine']}")
            print(f"  Success rate: {pattern['success_rate']:.1%}")
            print(f"  Requires JS: {pattern['requires_javascript']}")


async def example_5_conversational():
    """Example 5: Conversational interface"""
    print("\n" + "="*60)
    print("Example 5: Conversational Scraping")
    print("="*60)

    async with ConversationalScraper(user_id="example_user") as scraper:
        # Natural language request
        print("\nYou: Scrape the title from example.com")
        response = await scraper.chat(
            "Scrape the title from example.com"
        )
        print(f"Assistant: {response}")

        # Follow-up (remembers context)
        print("\nYou: What did you find?")
        response = await scraper.chat(
            "What did you find?"
        )
        print(f"Assistant: {response}")


async def example_6_user_memory():
    """Example 6: User memory and preferences"""
    print("\n" + "="*60)
    print("Example 6: User Memory")
    print("="*60)

    async with ConversationalScraper(user_id="john_doe") as scraper:
        # Set preferences
        await scraper.user_memory.set_preference("favorite_format", "JSON")
        await scraper.user_memory.set_preference("scraping_mode", "fast")

        # Chat - it remembers you!
        response = await scraper.chat(
            "Hello! What do you know about me?"
        )
        print(f"Assistant: {response}")

        # Get profile
        profile = await scraper.user_memory.get_profile()
        print(f"\nUser profile:")
        print(f"  User ID: {profile['user_id']}")
        print(f"  Created: {profile['created_at']}")

        # Get preferences
        prefs = await scraper.user_memory.get_preferences()
        print(f"\nPreferences: {prefs}")


async def example_7_statistics():
    """Example 7: View learning statistics"""
    print("\n" + "="*60)
    print("Example 7: Learning Statistics")
    print("="*60)

    async with BeastScraper() as scraper:
        # Perform some scrapes
        await scraper.scrape("https://example.com", learn=True)
        await scraper.scrape("https://httpbin.org/html", learn=True)

        # Get statistics
        stats = await scraper.pattern_learner.get_statistics()

        print(f"\nLearning Statistics:")
        print(f"  Total attempts: {stats['total_attempts']}")
        print(f"  Successful: {stats['successful_attempts']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Avg response time: {stats['avg_response_time']:.2f}s")
        print(f"  Learned domains: {stats['learned_domains']}")


async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("🦾 BEAST Scraper - Examples")
    print("="*60)

    try:
        await example_1_simple_scrape()
        await example_2_extract_fields()
        await example_3_parallel_scraping()
        await example_4_self_improvement()
        await example_5_conversational()
        await example_6_user_memory()
        await example_7_statistics()

        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
