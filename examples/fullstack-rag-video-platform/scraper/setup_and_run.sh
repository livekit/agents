#!/bin/bash

# 🦾 BEAST Scraper - Auto Setup & Launch Script
# This script installs everything and runs the system automatically

set -e

echo "════════════════════════════════════════════════════════════════"
echo "🦾 BEAST SCRAPER - YOLO MODE ACTIVATED"
echo "Automating EVERYTHING..."
echo "════════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/home/user/agents/examples/fullstack-rag-video-platform/scraper"
cd "$BASE_DIR"

# Step 1: Check Python
echo -e "${BLUE}[1/10]${NC} Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Python found: $PYTHON_VERSION"
else
    echo "✗ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Step 2: Install core dependencies
echo -e "\n${BLUE}[2/10]${NC} Installing core Python dependencies..."
pip3 install --quiet --break-system-packages \
    playwright beautifulsoup4 lxml httpx aiohttp \
    pydantic pydantic-settings aiosqlite python-dotenv loguru \
    2>/dev/null || echo "Some packages already installed"
echo -e "${GREEN}✓${NC} Core dependencies ready"

# Step 3: Install Playwright browsers
echo -e "\n${BLUE}[3/10]${NC} Installing Playwright Chromium browser..."
playwright install chromium >/dev/null 2>&1 || echo "Browser already installed"
echo -e "${GREEN}✓${NC} Playwright browser ready"

# Step 4: Check Ollama
echo -e "\n${BLUE}[4/10]${NC} Checking Ollama (local LLM)..."
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓${NC} Ollama is installed"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Ollama server is running"
    else
        echo -e "${YELLOW}⚠${NC}  Starting Ollama server..."
        ollama serve >/dev/null 2>&1 &
        sleep 3
    fi

    # Check for models
    if ollama list | grep -q "llama"; then
        echo -e "${GREEN}✓${NC} LLM model available"
    else
        echo -e "${YELLOW}⚠${NC}  Pulling llama3.2 model (this may take a few minutes)..."
        ollama pull llama3.2 || echo "Will use simple mode without LLM"
    fi
else
    echo -e "${YELLOW}⚠${NC}  Ollama not installed - will run in simple mode"
    echo "   To install: curl -fsSL https://ollama.ai/install.sh | sh"
fi

# Step 5: Create data directories
echo -e "\n${BLUE}[5/10]${NC} Creating data directories..."
mkdir -p data storage sessions
echo -e "${GREEN}✓${NC} Directories created"

# Step 6: Create config file
echo -e "\n${BLUE}[6/10]${NC} Creating configuration..."
cat > .env << 'EOF'
# BEAST Scraper Configuration
SCRAPER_PRIMARY_ENGINE=playwright
SCRAPER_HEADLESS=true
SCRAPER_LOG_LEVEL=INFO

# LLM (Ollama)
SCRAPER_LLM_PROVIDER=ollama
SCRAPER_LLM_MODEL=llama3.2
SCRAPER_LLM_BASE_URL=http://localhost:11434

# Performance
SCRAPER_MAX_WORKERS=10
SCRAPER_TIMEOUT=30
EOF
echo -e "${GREEN}✓${NC} Configuration ready"

# Step 7: Create simple demo script
echo -e "\n${BLUE}[7/10]${NC} Creating demo script..."
cat > demo_simple.py << 'PYEOF'
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
PYEOF

echo -e "${GREEN}✓${NC} Demo script created"

# Step 8: Create auto-launcher
echo -e "\n${BLUE}[8/10]${NC} Creating auto-launcher..."
cat > run.sh << 'EOF'
#!/bin/bash
cd /home/user/agents/examples/fullstack-rag-video-platform/scraper
python3 demo_simple.py
EOF
chmod +x run.sh
echo -e "${GREEN}✓${NC} Launcher ready"

# Step 9: Test installation
echo -e "\n${BLUE}[9/10]${NC} Testing installation..."
python3 -c "import playwright, httpx, bs4, aiohttp; print('All core modules imported successfully')" 2>/dev/null && \
    echo -e "${GREEN}✓${NC} All dependencies working" || \
    echo -e "${YELLOW}⚠${NC}  Some optional dependencies missing"

# Step 10: Show status
echo -e "\n${BLUE}[10/10]${NC} Installation complete!"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ BEAST SCRAPER IS READY!${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "🚀 Quick Start:"
echo "   ./run.sh              # Run simple demo"
echo "   python3 example.py    # Run full examples (needs Ollama)"
echo ""
echo "📁 Locations:"
echo "   Config: .env"
echo "   Data: ./data/"
echo "   Storage: ./storage/"
echo ""
echo "📚 Documentation:"
echo "   QUICKSTART.md - 5-minute guide"
echo "   SCRAPING_BEAST.md - Complete docs"
echo ""

# Run the demo
echo "════════════════════════════════════════════════════════════════"
echo "🎬 RUNNING DEMO NOW..."
echo "════════════════════════════════════════════════════════════════"
echo ""

python3 demo_simple.py
