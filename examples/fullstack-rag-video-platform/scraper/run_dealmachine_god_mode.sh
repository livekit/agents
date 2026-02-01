#!/bin/bash

# DealMachine GOD MODE - Ultimate Automated Scraper
# This script runs everything automatically

set -e

echo "════════════════════════════════════════════════════════════════════"
echo "🔥 DEALMACHINE GOD MODE ACTIVATED 🔥"
echo "The Ultimate Specialized Scraper with RAG Intelligence"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/home/user/agents/examples/fullstack-rag-video-platform/scraper"
DOCS_DIR="/home/user/Documents/dealmachine_data"

cd "$BASE_DIR"

# Create documents directory
echo -e "${BLUE}[1/6]${NC} Creating documents directory..."
mkdir -p "$DOCS_DIR"
echo -e "${GREEN}✓${NC} Documents directory ready: $DOCS_DIR"
echo ""

# Check dependencies
echo -e "${BLUE}[2/6]${NC} Checking dependencies..."
python3 -c "import playwright, bs4, aiohttp" 2>/dev/null && \
    echo -e "${GREEN}✓${NC} All dependencies installed" || \
    (echo -e "${YELLOW}⚠${NC}  Installing missing dependencies..." && \
     pip3 install --quiet --break-system-packages playwright beautifulsoup4 aiohttp)
echo ""

# Ensure Playwright browsers are installed
echo -e "${BLUE}[3/6]${NC} Checking Playwright browsers..."
if [ ! -d "/root/.cache/ms-playwright" ]; then
    echo -e "${YELLOW}⚠${NC}  Installing Playwright browsers..."
    playwright install chromium >/dev/null 2>&1
fi
echo -e "${GREEN}✓${NC} Playwright browsers ready"
echo ""

# Show script info
echo -e "${BLUE}[4/6]${NC} Preparing DealMachine SENSEI..."
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Features:${NC}"
echo "  🥋 Expert knowledge of DealMachine.com structure"
echo "  🧠 AI-powered pattern recognition and learning"
echo "  🔐 Auto-login and session management"
echo "  📊 RAG integration for intelligent data analysis"
echo "  💾 Auto-save to your Documents folder"
echo "  📈 Generates insights and reports"
echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Ask for credentials
echo -e "${BLUE}[5/6]${NC} Setup credentials (optional)..."
echo ""
echo -e "${YELLOW}NOTE:${NC} For full scraping, you need DealMachine credentials."
echo "Press ENTER to skip and run in demo mode (site analysis only)"
echo ""
read -p "DealMachine Email (or press Enter to skip): " DEALMACHINE_EMAIL

if [ -n "$DEALMACHINE_EMAIL" ]; then
    read -s -p "DealMachine Password: " DEALMACHINE_PASSWORD
    echo ""
    echo -e "${GREEN}✓${NC} Credentials configured"

    # Create credentials file
    cat > "$BASE_DIR/dealmachine_credentials.json" << EOF
{
    "email": "$DEALMACHINE_EMAIL",
    "password": "$DEALMACHINE_PASSWORD"
}
EOF
    chmod 600 "$BASE_DIR/dealmachine_credentials.json"
else
    echo -e "${YELLOW}⚠${NC}  Running in DEMO mode (no login)"
fi
echo ""

# Run the scraper
echo -e "${BLUE}[6/6]${NC} Launching DealMachine SENSEI..."
echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 STARTING SCRAPE...${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Run Python scraper
python3 dealmachine_sensei.py

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🧠 RUNNING RAG INTEGRATION & ANALYSIS...${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Run RAG integration
python3 dealmachine_rag_integration.py

echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ GOD MODE COMPLETE!${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}📁 Results Location:${NC}"
echo "   $DOCS_DIR"
echo ""
echo -e "${CYAN}📄 Files Created:${NC}"
ls -1 "$DOCS_DIR" 2>/dev/null | while read file; do
    echo "   ✓ $file"
done
echo ""
echo -e "${CYAN}📊 Quick Stats:${NC}"
if [ -f "$DOCS_DIR/insights.json" ]; then
    python3 << 'PYEOF'
import json
with open('/home/user/Documents/dealmachine_data/insights.json') as f:
    insights = json.load(f)
    print(f"   • Total Properties: {insights.get('total_properties', 0)}")
    if insights.get('cities'):
        top_city = max(insights['cities'].items(), key=lambda x: x[1])
        print(f"   • Top City: {top_city[0]} ({top_city[1]} properties)")
    if insights.get('price_stats'):
        avg_price = insights['price_stats'].get('avg', 0)
        print(f"   • Average Price: ${avg_price:,}")
PYEOF
fi
echo ""
echo -e "${PURPLE}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Check your Documents folder for all scraped data"
echo "  2. Read knowledge_base_report.txt for full analysis"
echo "  3. Open CSV files in Excel/Google Sheets"
echo "  4. Run again to scrape more properties!"
echo ""
echo -e "${GREEN}The SENSEI has spoken. 🥋${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Clean up credentials file
if [ -f "$BASE_DIR/dealmachine_credentials.json" ]; then
    rm "$BASE_DIR/dealmachine_credentials.json"
fi
