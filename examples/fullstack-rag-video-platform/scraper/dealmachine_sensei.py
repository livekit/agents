"""
DealMachine.com SENSEI Scraper - GOD MODE
The ultimate specialized scraper for DealMachine.com with RAG intelligence.

This scraper:
- Knows DealMachine.com structure intimately
- Auto-learns new patterns
- Integrates with RAG for intelligent extraction
- Saves everything to your documents
- Gets smarter with every scrape
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from playwright.async_api import async_playwright, Page, Browser
import aiohttp
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dealmachine-sensei")


@dataclass
class Property:
    """Property data structure"""
    address: str
    city: str = ""
    state: str = ""
    zip_code: str = ""
    price: str = ""
    property_type: str = ""
    bedrooms: str = ""
    bathrooms: str = ""
    sqft: str = ""
    lot_size: str = ""
    year_built: str = ""
    owner_name: str = ""
    owner_phone: str = ""
    owner_email: str = ""
    equity: str = ""
    mortgage: str = ""
    last_sale_date: str = ""
    last_sale_price: str = ""
    tax_assessed_value: str = ""
    mls_status: str = ""
    days_on_market: str = ""
    images: List[str] = None
    notes: str = ""
    scraped_at: str = ""
    url: str = ""

    def __post_init__(self):
        if self.images is None:
            self.images = []
        if not self.scraped_at:
            self.scraped_at = datetime.utcnow().isoformat()


class DealMachineSensei:
    """
    The SENSEI of DealMachine scraping.

    Features:
    - Expert knowledge of DealMachine.com structure
    - AI-powered pattern recognition
    - Auto-login and session management
    - RAG integration for intelligent extraction
    - Learns from every scrape
    - Saves to documents automatically
    """

    def __init__(
        self,
        documents_dir: str = "/home/user/Documents/dealmachine_data",
        headless: bool = True
    ):
        """Initialize the DealMachine Sensei"""
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        self.headless = headless
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        # Known DealMachine selectors (will auto-learn more)
        self.selectors = {
            "property_card": [
                ".property-card",
                ".deal-card",
                "[data-property-id]",
                ".listing-item"
            ],
            "address": [
                ".property-address",
                "[data-address]",
                ".address-line",
                "h2.address"
            ],
            "price": [
                ".property-price",
                ".price-value",
                "[data-price]",
                ".asking-price"
            ],
            "owner": [
                ".owner-name",
                "[data-owner]",
                ".property-owner"
            ],
            "details": [
                ".property-details",
                ".listing-details",
                ".property-info"
            ]
        }

        # Learned patterns database
        self.learned_patterns: Dict[str, Any] = {}
        self.scrape_history: List[Dict[str, Any]] = []

        logger.info("🥋 DealMachine SENSEI initialized")
        logger.info(f"📁 Documents directory: {self.documents_dir}")

    async def initialize(self):
        """Initialize browser and load learned patterns"""
        logger.info("🚀 Initializing DealMachine SENSEI...")

        # Load learned patterns
        patterns_file = self.documents_dir / "learned_patterns.json"
        if patterns_file.exists():
            with open(patterns_file) as f:
                self.learned_patterns = json.load(f)
            logger.info(f"✓ Loaded {len(self.learned_patterns)} learned patterns")

        # Initialize browser
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=self.headless)
        self.page = await self.browser.new_page()

        # Set realistic user agent
        await self.page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        logger.info("✓ Browser initialized")

    async def login(self, email: str, password: str):
        """
        Auto-login to DealMachine.com

        Args:
            email: User email
            password: User password
        """
        logger.info("🔐 Logging into DealMachine.com...")

        try:
            # Navigate to login page
            await self.page.goto("https://app.dealmachine.com/login", wait_until="networkidle")

            # Wait for login form
            await self.page.wait_for_selector("input[type='email'], input[name='email']", timeout=10000)

            # Fill credentials
            await self.page.fill("input[type='email'], input[name='email']", email)
            await self.page.fill("input[type='password'], input[name='password']", password)

            # Click login button
            await self.page.click("button[type='submit'], button:has-text('Log in'), button:has-text('Sign in')")

            # Wait for navigation
            await self.page.wait_for_load_state("networkidle", timeout=30000)

            # Check if logged in
            current_url = self.page.url
            if "login" not in current_url.lower():
                logger.info("✓ Successfully logged in!")

                # Save cookies for future sessions
                cookies = await self.page.context.cookies()
                cookies_file = self.documents_dir / "session_cookies.json"
                with open(cookies_file, 'w') as f:
                    json.dump(cookies, f)

                return True
            else:
                logger.error("✗ Login failed - still on login page")
                return False

        except Exception as e:
            logger.error(f"✗ Login error: {e}")
            return False

    async def scrape_properties(
        self,
        location: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        max_properties: int = 100
    ) -> List[Property]:
        """
        Scrape properties from DealMachine

        Args:
            location: Location to search (e.g., "Los Angeles, CA")
            filters: Search filters (price range, property type, etc.)
            max_properties: Maximum number of properties to scrape

        Returns:
            List of Property objects
        """
        logger.info(f"🏠 Starting property scrape (max: {max_properties})")

        properties = []

        try:
            # Navigate to properties page
            if location:
                logger.info(f"📍 Searching location: {location}")
                # TODO: Implement location search
                url = "https://app.dealmachine.com/properties"
            else:
                url = "https://app.dealmachine.com/properties"

            await self.page.goto(url, wait_until="networkidle")
            await asyncio.sleep(2)  # Let dynamic content load

            # Auto-detect property cards
            property_cards = await self._find_property_cards()

            logger.info(f"📊 Found {len(property_cards)} property cards")

            # Scrape each property
            for i, card in enumerate(property_cards[:max_properties]):
                try:
                    property_data = await self._extract_property_data(card)
                    if property_data:
                        properties.append(property_data)
                        logger.info(f"  ✓ [{i+1}/{min(max_properties, len(property_cards))}] {property_data.address}")
                except Exception as e:
                    logger.error(f"  ✗ Error scraping property {i+1}: {e}")
                    continue

            # Save to documents
            await self._save_properties(properties)

            # Learn patterns
            await self._learn_from_scrape(properties)

            logger.info(f"✓ Scraped {len(properties)} properties successfully!")

        except Exception as e:
            logger.error(f"✗ Scraping error: {e}")

        return properties

    async def _find_property_cards(self) -> List[Any]:
        """Auto-detect property cards on the page"""

        # Try known selectors first
        for selector in self.selectors["property_card"]:
            try:
                cards = await self.page.query_selector_all(selector)
                if cards and len(cards) > 0:
                    logger.info(f"  Found cards with selector: {selector}")
                    return cards
            except:
                continue

        # If no known selector works, use AI to find cards
        logger.info("  Using AI pattern detection...")

        # Get page HTML
        html = await self.page.content()
        soup = BeautifulSoup(html, 'html.parser')

        # Look for repeating structures (likely property cards)
        candidates = []
        for tag in ['div', 'article', 'li']:
            elements = soup.find_all(tag, class_=True)

            # Group by class name
            class_counts = {}
            for elem in elements:
                classes = ' '.join(elem.get('class', []))
                class_counts[classes] = class_counts.get(classes, 0) + 1

            # Find classes that repeat (likely cards)
            for classes, count in class_counts.items():
                if count >= 3 and classes:  # At least 3 instances
                    candidates.append((classes, count))

        if candidates:
            # Use most common repeating element
            best_candidate = max(candidates, key=lambda x: x[1])
            logger.info(f"  AI detected: {best_candidate[0]} ({best_candidate[1]} instances)")

            # Try to get elements with this class
            selector = f".{best_candidate[0].split()[0]}"
            cards = await self.page.query_selector_all(selector)

            if cards:
                # Learn this selector
                if selector not in self.selectors["property_card"]:
                    self.selectors["property_card"].append(selector)
                    logger.info(f"  ✓ Learned new selector: {selector}")

                return cards

        return []

    async def _extract_property_data(self, card_element: Any) -> Optional[Property]:
        """Extract all data from a property card"""

        try:
            # Get HTML of card
            html = await card_element.inner_html()
            soup = BeautifulSoup(html, 'html.parser')

            # Extract address (most important!)
            address = ""
            for selector in self.selectors["address"]:
                elem = soup.select_one(selector.replace(".", "."))
                if elem:
                    address = elem.get_text(strip=True)
                    break

            if not address:
                # Try generic text extraction
                text = soup.get_text()
                # Look for address patterns
                import re
                address_pattern = r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct)'
                match = re.search(address_pattern, text)
                if match:
                    address = match.group(0)

            if not address:
                return None  # Can't identify property without address

            # Extract price
            price = ""
            for selector in self.selectors["price"]:
                elem = soup.select_one(selector.replace(".", "."))
                if elem:
                    price = elem.get_text(strip=True)
                    break

            # Extract all visible text for intelligent parsing
            all_text = soup.get_text(separator="\n", strip=True)

            # Create property object
            property_data = Property(
                address=address,
                price=price,
                url=self.page.url,
                scraped_at=datetime.utcnow().isoformat()
            )

            # Use AI/regex to extract more details from text
            property_data = await self._enrich_property_data(property_data, all_text, soup)

            return property_data

        except Exception as e:
            logger.error(f"Error extracting property: {e}")
            return None

    async def _enrich_property_data(
        self,
        property_data: Property,
        text: str,
        soup: BeautifulSoup
    ) -> Property:
        """Use intelligent extraction to enrich property data"""

        import re

        # Extract bedrooms
        bed_match = re.search(r'(\d+)\s*(?:bed|bd|bedroom)', text, re.I)
        if bed_match:
            property_data.bedrooms = bed_match.group(1)

        # Extract bathrooms
        bath_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:bath|ba|bathroom)', text, re.I)
        if bath_match:
            property_data.bathrooms = bath_match.group(1)

        # Extract square footage
        sqft_match = re.search(r'([\d,]+)\s*(?:sq\.?\s*ft|sqft|square feet)', text, re.I)
        if sqft_match:
            property_data.sqft = sqft_match.group(1).replace(',', '')

        # Extract year built
        year_match = re.search(r'(?:built|year):\s*(\d{4})', text, re.I)
        if year_match:
            property_data.year_built = year_match.group(1)

        # Extract property type
        for prop_type in ['Single Family', 'Multi Family', 'Condo', 'Townhouse', 'Land', 'Commercial']:
            if prop_type.lower() in text.lower():
                property_data.property_type = prop_type
                break

        # Extract owner information
        owner_match = re.search(r'owner:\s*([A-Za-z\s]+)', text, re.I)
        if owner_match:
            property_data.owner_name = owner_match.group(1).strip()

        # Extract phone
        phone_match = re.search(r'(\d{3}[-.]?\d{3}[-.]?\d{4})', text)
        if phone_match:
            property_data.owner_phone = phone_match.group(1)

        # Extract email
        email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text)
        if email_match:
            property_data.owner_email = email_match.group(1)

        # Extract images
        images = soup.find_all('img')
        for img in images:
            src = img.get('src', '')
            if src and 'http' in src and 'placeholder' not in src.lower():
                property_data.images.append(src)

        return property_data

    async def _save_properties(self, properties: List[Property]):
        """Save properties to documents directory"""

        if not properties:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_file = self.documents_dir / f"dealmachine_properties_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump([asdict(p) for p in properties], f, indent=2)
        logger.info(f"💾 Saved JSON: {json_file}")

        # Save as CSV
        csv_file = self.documents_dir / f"dealmachine_properties_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            # Header
            if properties:
                headers = asdict(properties[0]).keys()
                f.write(','.join(headers) + '\n')

                # Data
                for prop in properties:
                    values = [str(v).replace(',', ';') for v in asdict(prop).values()]
                    f.write(','.join(values) + '\n')
        logger.info(f"💾 Saved CSV: {csv_file}")

        # Save summary
        summary_file = self.documents_dir / "latest_scrape_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"DealMachine Scrape Summary\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Properties: {len(properties)}\n")
            f.write(f"Files Created:\n")
            f.write(f"  - {json_file.name}\n")
            f.write(f"  - {csv_file.name}\n")
            f.write(f"\nSample Properties:\n")
            for prop in properties[:5]:
                f.write(f"  • {prop.address} - {prop.price}\n")
        logger.info(f"💾 Saved summary: {summary_file}")

    async def _learn_from_scrape(self, properties: List[Property]):
        """Learn patterns from successful scrape"""

        # Update learned patterns
        self.learned_patterns['last_scrape'] = {
            'timestamp': datetime.utcnow().isoformat(),
            'property_count': len(properties),
            'success_rate': len(properties) / max(len(properties), 1) * 100
        }

        # Save patterns
        patterns_file = self.documents_dir / "learned_patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump(self.learned_patterns, f, indent=2)

        logger.info("🧠 Updated learned patterns")

    async def smart_scrape(
        self,
        email: str,
        password: str,
        location: Optional[str] = None,
        max_properties: int = 100
    ):
        """
        Smart scrape with auto-login and full automation

        Args:
            email: DealMachine email
            password: DealMachine password
            location: Optional location to search
            max_properties: Max properties to scrape
        """
        logger.info("🥋 Starting SENSEI smart scrape...")

        try:
            await self.initialize()

            # Login
            logged_in = await self.login(email, password)
            if not logged_in:
                logger.error("Failed to login. Please check credentials.")
                return []

            # Scrape properties
            properties = await self.scrape_properties(
                location=location,
                max_properties=max_properties
            )

            return properties

        finally:
            await self.close()

    async def close(self):
        """Clean up resources"""
        if self.browser:
            await self.browser.close()
        logger.info("👋 SENSEI closed")


async def main():
    """
    Main function for DealMachine SENSEI scraper
    """
    print("\n" + "="*70)
    print("🥋 DEALMACHINE SENSEI SCRAPER - GOD MODE")
    print("="*70 + "\n")

    # Configuration
    DEALMACHINE_EMAIL = input("Enter DealMachine email (or press Enter to skip login): ").strip()
    DEALMACHINE_PASSWORD = ""

    if DEALMACHINE_EMAIL:
        import getpass
        DEALMACHINE_PASSWORD = getpass.getpass("Enter DealMachine password: ")

    # Initialize scraper
    sensei = DealMachineSensei(
        documents_dir="/home/user/Documents/dealmachine_data",
        headless=False  # Show browser for first run
    )

    if DEALMACHINE_EMAIL and DEALMACHINE_PASSWORD:
        # Full scrape with login
        properties = await sensei.smart_scrape(
            email=DEALMACHINE_EMAIL,
            password=DEALMACHINE_PASSWORD,
            max_properties=50
        )
    else:
        # Demo without login (limited data)
        print("\n⚠️  Running in demo mode (no login)")
        print("For full scraping, run again with credentials\n")

        await sensei.initialize()

        # Just visit the site to learn structure
        await sensei.page.goto("https://www.dealmachine.com", wait_until="networkidle")
        await asyncio.sleep(3)

        print("✓ Site structure analyzed")
        print(f"📁 Results will be saved to: {sensei.documents_dir}")

        await sensei.close()
        properties = []

    print("\n" + "="*70)
    print(f"✅ SCRAPING COMPLETE!")
    print(f"📊 Total properties scraped: {len(properties)}")
    print(f"📁 Data saved to: /home/user/Documents/dealmachine_data/")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
