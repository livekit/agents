#!/usr/bin/env python3
"""
YOLO Mode Demo - Non-Interactive DealMachine SENSEI
Runs in demo mode without credentials, analyzes site structure
"""

import asyncio
import logging
from dealmachine_sensei import DealMachineSensei
from dealmachine_rag_integration import DealMachineRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-demo")


async def run_demo():
    """Run DealMachine SENSEI in demo mode"""
    print("\n" + "="*70)
    print("🔥 DEALMACHINE SENSEI - YOLO DEMO MODE")
    print("="*70 + "\n")

    print("⚡ Running in YOLO mode (no credentials needed)")
    print("📊 Will analyze DealMachine.com structure\n")

    # Initialize scraper
    sensei = DealMachineSensei(
        documents_dir="/home/user/Documents/dealmachine_data",
        headless=True  # Run in background
    )

    try:
        await sensei.initialize()

        print("🌐 Visiting DealMachine.com...")
        await sensei.page.goto("https://www.dealmachine.com", wait_until="networkidle")

        # Get page title
        title = await sensei.page.title()
        print(f"✓ Page loaded: {title}")

        # Analyze structure
        print("🔍 Analyzing site structure...")

        # Get all headings
        headings = await sensei.page.query_selector_all("h1, h2, h3")
        print(f"✓ Found {len(headings)} headings")

        # Get all links
        links = await sensei.page.query_selector_all("a")
        print(f"✓ Found {len(links)} links")

        # Get all buttons
        buttons = await sensei.page.query_selector_all("button")
        print(f"✓ Found {len(buttons)} buttons")

        # Take a screenshot
        screenshot_path = sensei.documents_dir / "dealmachine_homepage.png"
        await sensei.page.screenshot(path=str(screenshot_path))
        print(f"📸 Screenshot saved: {screenshot_path}")

        # Save structure analysis
        analysis = {
            "url": "https://www.dealmachine.com",
            "title": title,
            "headings_count": len(headings),
            "links_count": len(links),
            "buttons_count": len(buttons),
            "analyzed_at": str(asyncio.get_event_loop().time())
        }

        analysis_path = sensei.documents_dir / "site_analysis.json"
        import json
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"💾 Analysis saved: {analysis_path}")

        await sensei.close()

    except Exception as e:
        logger.error(f"Error during demo: {e}")
        await sensei.close()
        raise

    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print(f"📁 Results saved to: /home/user/Documents/dealmachine_data/")
    print("="*70 + "\n")

    # Run RAG integration
    print("\n" + "="*70)
    print("🧠 RUNNING RAG INTEGRATION...")
    print("="*70 + "\n")

    rag = DealMachineRAG()
    total = rag.load_knowledge_base()

    if total > 0:
        rag.index_properties()
        rag.get_insights()
        report = rag.generate_report()
        print(report)
    else:
        print("ℹ️  No scraped properties yet. Run with credentials to scrape real data.")

    print("\n✅ ALL DONE!\n")


if __name__ == "__main__":
    asyncio.run(run_demo())
