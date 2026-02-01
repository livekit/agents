"""
Simple BEAST Scraper Demo - No LLM Required
Fast scraping demo that works without Ollama
"""

import asyncio
import logging
from pathlib import Path

# Setup simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)

print("\n" + "="*60)
print("🦾 BEAST SCRAPER - SIMPLE DEMO")
print("="*60 + "\n")

async def demo_basic_scraping():
    """Demo basic web scraping"""
    print("📡 Testing basic HTTP scraping...")

    try:
        import httpx
        from bs4 import BeautifulSoup

        # Simple HTTP scrape
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("https://example.com")

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.find('title')

                print(f"✓ Successfully scraped example.com")
                print(f"  Title: {title.string if title else 'N/A'}")
                print(f"  Response time: {response.elapsed.total_seconds():.2f}s")
                print(f"  Content length: {len(response.text)} bytes")
            else:
                print(f"✗ Failed with status: {response.status_code}")

    except Exception as e:
        print(f"✗ Error: {e}")

async def demo_parallel_scraping():
    """Demo parallel scraping"""
    print("\n⚡ Testing parallel scraping...")

    try:
        import httpx
        from datetime import datetime

        urls = [
            "https://example.com",
            "https://httpbin.org/html",
        ]

        start_time = datetime.now()

        async with httpx.AsyncClient(timeout=10) as client:
            tasks = [client.get(url) for url in urls]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = (datetime.now() - start_time).total_seconds()

        success_count = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)

        print(f"✓ Scraped {len(urls)} URLs in parallel")
        print(f"  Success: {success_count}/{len(urls)}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average: {elapsed/len(urls):.2f}s per URL")

    except Exception as e:
        print(f"✗ Error: {e}")

async def demo_playwright():
    """Demo Playwright browser scraping"""
    print("\n🌐 Testing Playwright (browser scraping)...")

    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto("https://example.com", wait_until="load")

            title = await page.title()
            content = await page.content()

            await browser.close()

            print(f"✓ Successfully scraped with Playwright")
            print(f"  Title: {title}")
            print(f"  Page size: {len(content)} bytes")

    except Exception as e:
        print(f"✗ Playwright error: {e}")
        print("  (Playwright browser may need installation)")

async def main():
    """Run all demos"""
    print("Starting BEAST Scraper demos...\n")

    await demo_basic_scraping()
    await demo_parallel_scraping()
    await demo_playwright()

    print("\n" + "="*60)
    print("✓ DEMO COMPLETE!")
    print("="*60)
    print("\n📚 Next steps:")
    print("  1. Install Ollama for conversational features")
    print("  2. Run 'python example.py' for advanced demos")
    print("  3. Check QUICKSTART.md for usage guide")
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())
